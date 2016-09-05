package org.skytruth.feature_pipeline

import io.github.karols.units._
import io.github.karols.units.SI._
import io.github.karols.units.defining._
import com.google.common.geometry.{S2LatLng}
import com.google.protobuf.MessageLite
import com.spotify.scio._
import com.spotify.scio.values.SCollection
import com.spotify.scio.bigquery._
import com.typesafe.scalalogging.LazyLogging
import com.google.cloud.dataflow.sdk.options.{DataflowPipelineOptions, PipelineOptionsFactory}
import com.google.cloud.dataflow.sdk.runners.{DataflowPipelineRunner}
import com.google.cloud.dataflow.sdk.io.{Write}
import com.opencsv.CSVReader
import java.io.FileReader
import org.joda.time.{Instant, Duration}
import org.skytruth.dataflow.{TFRecordSink}
import org.tensorflow.example.{
  Example,
  Feature,
  Features,
  FeatureList,
  FeatureLists,
  SequenceExample,
  Int64List,
  FloatList,
  BytesList
}
import scala.collection.{mutable, immutable}
import scala.collection.JavaConversions._

object Parameters {
  val minRequiredPositions = 1000
  val minTimeBetweenPoints = Duration.standardMinutes(5)

  val stationaryPeriodMaxDistance = 800.0.of[meter]
  val stationaryPeriodMinDuration = Duration.standardHours(2 * 24)

  val inputMeasuresPath =
    "gs://new-benthos-pipeline/data-production/measures-pipeline/st-segment/*/*"
  val outputFeaturesPath =
    "gs://alex-dataflow-scratch/features-scratch"
  val gceProject = "world-fishing-827"
  val dataflowStaging = "gs://alex-dataflow-scratch/dataflow-staging"

  // Sourced from: https://github.com/GlobalFishingWatch
  val vesselClassificationPath = "feature-pipeline/src/data/combined_classification_list.csv"
}

object AdditionalUnits {
  type knots = DefineUnit[_k ~: _n ~: _o ~: _t]
  type meters_per_second = meter / second

  implicit val knots_to_mps = one[knots].contains(0.514444)[meters_per_second]

  type degrees = DefineUnit[_d ~: _e ~: _g]
}

import AdditionalUnits._

case class LatLon(val lat: DoubleU[degrees], val lon: DoubleU[degrees]) {
  def getDistance(other: LatLon): DoubleU[meter] = {
    val p1 = S2LatLng.fromDegrees(lat.value, lon.value)
    val p2 = S2LatLng.fromDegrees(other.lat.value, other.lon.value)

    p1.getEarthDistance(p2).of[meter]
  }
}

case class VesselMetadata(
    val mmsi: Int,
    // Training, Test, Unclassified
    val dataset: String = "Unclassified",
    val vesselType: String = "Unclassified"
)

case class VesselLocationRecord(
    val timestamp: Instant,
    val location: LatLon,
    val distanceToShore: DoubleU[kilometer],
    val distanceToPort: DoubleU[kilometer],
    val speed: DoubleU[knots],
    val course: DoubleU[degrees],
    val heading: DoubleU[degrees]
)

object Pipeline extends LazyLogging {
  lazy val blacklistedMmsis = Set(0, 12345)

  // Reads JSON vessel records, filters to only location records, groups by MMSI and sorts
  // by ascending timestamp.
  def readJsonRecords(input: SCollection[TableRow], metadata: SCollection[(Int, VesselMetadata)])
    : SCollection[(VesselMetadata, Seq[VesselLocationRecord])] = {

    val vesselMetadataMap = metadata.asMapSideInput
    // Keep only records with a location.
    input
      .withSideInputs(vesselMetadataMap)
      .filter((json, _) => json.containsKey("lat") && json.containsKey("lon"))
      // Build a typed location record with units of measure.
      .map((json, s) => {
        val mmsi = json.getLong("mmsi").toInt
        val metadata = s(vesselMetadataMap).getOrElse(mmsi, VesselMetadata(mmsi))
        val record =
          VesselLocationRecord(Instant.parse(json.getString("timestamp")),
                               LatLon(json.getDouble("lat").of[degrees],
                                      json.getDouble("lon").of[degrees]),
                               json.getDouble("distance_to_shore").of[kilometer],
                               json.getDouble("distance_to_port").of[kilometer],
                               json.getDouble("speed").of[knots],
                               json.getDouble("course").of[degrees],
                               json.getDouble("heading").of[degrees])
        (metadata, record)
      })
      .toSCollection
      .filter { case (metadata, records) => !blacklistedMmsis.contains(metadata.mmsi) }
      .groupByKey
      .map { case (metadata, records) => (metadata, records.toSeq.sortBy(_.timestamp.getMillis)) }
  }

  def thinPoints(records: Iterable[VesselLocationRecord]): Iterable[VesselLocationRecord] = {
    val thinnedPoints = mutable.ListBuffer.empty[VesselLocationRecord]

    // Thin locations down to a minimum time between each.
    records.foreach { vr =>
      if (thinnedPoints.isEmpty || !vr.timestamp.isBefore(
            thinnedPoints.last.timestamp.plus(Parameters.minTimeBetweenPoints))) {
        thinnedPoints.append(vr)
      }
    }

    thinnedPoints
  }

  def removeStationaryPeriods(
      records: Iterable[VesselLocationRecord]): Iterable[VesselLocationRecord] = {
    // Remove long stationary periods from the record.
    val withoutLongStationaryPeriods = mutable.ListBuffer.empty[VesselLocationRecord]
    val currentPeriod = mutable.Queue.empty[VesselLocationRecord]
    records.foreach { vr =>
      if (!currentPeriod.isEmpty) {
        val periodFirst = currentPeriod.front
        val speed = vr.speed
        val distanceDelta = vr.location.getDistance(periodFirst.location)
        if (distanceDelta > Parameters.stationaryPeriodMaxDistance) {
          if (vr.timestamp.isAfter(
                periodFirst.timestamp.plus(Parameters.stationaryPeriodMinDuration))) {
            withoutLongStationaryPeriods.append(periodFirst)
            withoutLongStationaryPeriods.append(currentPeriod.last)
          } else {
            withoutLongStationaryPeriods ++= currentPeriod
          }

          currentPeriod.clear
        }
      }

      currentPeriod.enqueue(vr)
    }
    withoutLongStationaryPeriods ++= currentPeriod

    withoutLongStationaryPeriods
  }

  def filterVesselRecords(
      input: SCollection[(VesselMetadata, Seq[VesselLocationRecord])],
      minRequiredPositions: Int): SCollection[(VesselMetadata, Seq[VesselLocationRecord])] = {
    val vesselsWithSufficientData = input.filter {
      case (_, records) => records.length > minRequiredPositions
    }

    // Remove vessels with insufficient locations.
    vesselsWithSufficientData.map {
      case (metadata, records) =>
        val thinnedPoints = thinPoints(records)
        val withoutLongStationaryPeriods = removeStationaryPeriods(thinnedPoints)

        (metadata, withoutLongStationaryPeriods.toSeq)
    }
  }

  def angleNormalize(angle: DoubleU[degrees]) = {
    angle
  }

  def buildSingleVesselFeatures(input: Seq[VesselLocationRecord]): Seq[Array[Double]] =
    input
      .sliding(3)
      .map {
        case Seq(p0, p1, p2) =>
          // Timestamp delta seconds, distance delta meters, cog delta degrees, sog mps,
          // integrated cog delta degrees, integrated sog mps, local tod, local month of year,
          // # neighbours, # closest neighbours.
          val timestampDeltaSeconds = p1.timestamp.getMillis / 1000 - p0.timestamp.getMillis / 1000
          val distanceDeltaMeters = p1.location.getDistance(p0.location).value
          val speedMps = p1.speed.convert[meters_per_second].value
          val integratedSpeedMps = distanceDeltaMeters / timestampDeltaSeconds
          val cogDeltaDegrees = angleNormalize((p1.course - p0.course)).value

          // TODO(alexwilson): Expand to full feature set.
          Array(timestampDeltaSeconds,
                distanceDeltaMeters,
                speedMps,
                integratedSpeedMps,
                cogDeltaDegrees)
      }
      .toSeq

  def buildTFExampleProto(metadata: VesselMetadata, data: Seq[Array[Double]]) = {
    val example = SequenceExample.newBuilder()
    val contextBuilder = example.getContextBuilder()

    // TODO(alexwilson): bring this in from vessel metadata when present.
    val vessel_type = 0

    // Add the mmsi and vessel type to the builder
    contextBuilder.putFeature(
      "mmsi",
      Feature.newBuilder().setInt64List(Int64List.newBuilder().addValue(metadata.mmsi)).build())
    contextBuilder.putFeature(
      "vessel_type",
      Feature.newBuilder().setInt64List(Int64List.newBuilder().addValue(vessel_type)).build())

    // TODO(alexwilson): Add timestamps for each for later subsequence selection.
    // Add all the sequence data as a feature.
    val sequenceData = FeatureList.newBuilder()
    data.foreach { datum =>
      val floatData = FloatList.newBuilder()
      datum.foreach { v =>
        floatData.addValue(v.toFloat)
      }
      sequenceData.addFeature(Feature.newBuilder().setFloatList(floatData.build()))
    }
    val featureLists = FeatureLists.newBuilder()
    featureLists.putFeatureList("movement features", sequenceData.build())
    example.setFeatureLists(featureLists)

    example.build()
  }

  def buildVesselFeatures(input: SCollection[(VesselMetadata, Seq[VesselLocationRecord])]) =
    input.map {
      case (metadata, records) =>
        val features = buildSingleVesselFeatures(records)
        val featuresAsTFExample = buildTFExampleProto(metadata, features)
        featuresAsTFExample.asInstanceOf[MessageLite]
    }

  def main(argArray: Array[String]) {
    val (options, remaining_args) = ScioContext.parseArguments[DataflowPipelineOptions](argArray)
    options.setRunner(classOf[DataflowPipelineRunner])
    options.setProject(Parameters.gceProject)
    options.setStagingLocation(Parameters.dataflowStaging)

    val sc = ScioContext(options)

    // TODO(alexwilson): Load up the vessel metadata as a side input and join
    // with the JSON records.
    // Read the vessel metadata for known vessels.
    val vesselMetadataCsv =
      remaining_args.getOrElse("vessel_metadata", Parameters.vesselClassificationPath)
    val metadataReader = new CSVReader(new FileReader(vesselMetadataCsv))
    val allLines = metadataReader.readAll()
    val vesselMetadata = allLines.toList
      .drop(1)
      .map { l =>
        val mmsi = l(0).toInt
        val dataset = l(1)
        val vesselType = l(2)

        (mmsi, VesselMetadata(mmsi, dataset, vesselType))
      }
      .toMap

    // Read, filter and build location records.
    val locationRecords =
      readJsonRecords(sc.tableRowJsonFile(Parameters.inputMeasuresPath),
                      sc.parallelize(vesselMetadata))

    val filtered = filterVesselRecords(locationRecords, Parameters.minRequiredPositions)
    val features = buildVesselFeatures(filtered)

    // TODO(alexwilson): Filter out the features into Train, Test, Unclassified.
    features.internal.apply(
      "WriteTFRecords",
      Write.to(new TFRecordSink(Parameters.outputFeaturesPath, "tfrecord", "shard")))

    sc.close()
  }
}
