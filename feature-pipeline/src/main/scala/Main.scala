package org.skytruth.feature_pipeline

import io.github.karols.units._
import io.github.karols.units.SI._
import io.github.karols.units.defining._

import com.google.common.geometry.{S2LatLng}
import com.google.protobuf.{ByteString, MessageLite}
import com.spotify.scio._
import com.spotify.scio.values.SCollection
import com.spotify.scio.bigquery._
import com.typesafe.scalalogging.LazyLogging
import com.google.cloud.dataflow.sdk.options.{DataflowPipelineOptions, PipelineOptionsFactory}
import com.google.cloud.dataflow.sdk.runners.{DataflowPipelineRunner}
import com.google.cloud.dataflow.sdk.io.{Write}
import com.opencsv.CSVReader
import java.io.{File, FileReader, InputStream, OutputStream}
import java.nio.charset.StandardCharsets
import org.apache.commons.math3.util.MathUtils
import org.joda.time.{DateTime, DateTimeZone, Instant, Duration}
import org.joda.time.format.ISODateTimeFormat
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
    //"gs://new-benthos-pipeline/data-production/measures-pipeline/st-segment/*/*"
    "gs://new-benthos-pipeline/data-production/measures-pipeline/st-segment/2015-*/*"
  val outputFeaturesPath =
    "gs://alex-dataflow-scratch/features-scratch"
  val gceProject = "world-fishing-827"
  val dataflowStaging = "gs://alex-dataflow-scratch/dataflow-staging"

  // Sourced from: https://github.com/GlobalFishingWatch
  val vesselClassificationPath = "feature-pipeline/src/data/combined_classification_list.csv"

  val splits = Seq("Training", "Test", "Unclassified")
}

object AdditionalUnits {
  type knots = DefineUnit[_k ~: _n ~: _o ~: _t]
  type meters_per_second = meter / second

  type degrees = DefineUnit[_d ~: _e ~: _g]
  type radians = DefineUnit[_r ~: _a ~: _d]

  implicit val knots_to_mps = one[knots].contains(0.514444)[meters_per_second]
  implicit val radians_to_degrees = one[degrees].contains(MathUtils.TWO_PI / 360.0)[radians]
}

import AdditionalUnits._

case class LatLon(lat: DoubleU[degrees], lon: DoubleU[degrees]) {
  def getDistance(other: LatLon): DoubleU[meter] = {
    val p1 = S2LatLng.fromDegrees(lat.value, lon.value)
    val p2 = S2LatLng.fromDegrees(other.lat.value, other.lon.value)

    p1.getEarthDistance(p2).of[meter]
  }
}

object VesselMetadata {
  val vesselTypeToIndexMap = Seq(
    "Purse seine",
    "Longliner",
    "Trawler",
    "Pots and traps",
    "Passenger",
    "Tug",
    "Cargo/Tanker",
    "Supply"
  ).zipWithIndex.map { case (name, index) => (name, index + 1) }.toMap
}

case class VesselMetadata(
    mmsi: Int,
    dataset: String = "Unclassified",
    vesselType: String = "Unknown"
) {
  def vesselTypeIndex = VesselMetadata.vesselTypeToIndexMap.getOrElse(vesselType, 0)
}

// TODO(alexwilson): For now build simple coder for this class. :-(
case class VesselLocationRecord(
    timestamp: Instant,
    location: LatLon,
    distanceToShore: DoubleU[kilometer],
    distanceToPort: DoubleU[kilometer],
    speed: DoubleU[knots],
    course: DoubleU[degrees],
    heading: DoubleU[degrees]
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
          // TODO(alexwilson): Double-check all these units are correct.
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
      .filter { case (metadata, record) => !blacklistedMmsis.contains(metadata.mmsi) }
      .groupByKey
      .map {
        case (metadata, records) => (metadata, records.toIndexedSeq.sortBy(_.timestamp.getMillis))
      }
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
    // Remove long stationary periods from the record: anything over the threshold
    // time will be reduced to just the start and end points of the period.
    // TODO(alexwilson): Tim points out that leaves vessels sitting around for t - delta looking
    // significantly different from those sitting around for t + delta. Consider his scheme of just
    // cropping all excess time over the threshold instead.
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

          currentPeriod.clear()
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

        (metadata, withoutLongStationaryPeriods.toIndexedSeq)
    }
  }

  // TODO(alexwilson): Implement
  def angleNormalize(angle: DoubleU[degrees]) =
    MathUtils
      .normalizeAngle(angle.convert[radians].value, MathUtils.TWO_PI / 2.0)
      .of[radians]
      .convert[degrees]

  def buildSingleVesselFeatures(input: Seq[VesselLocationRecord]): Seq[Array[Double]] =
    input
      .sliding(3)
      .map {
        case Seq(p0, p1, p2) =>
          val timestampSeconds = p1.timestamp.getMillis / 1000
          val timestampDeltaSeconds = p1.timestamp.getMillis / 1000 - p0.timestamp.getMillis / 1000
          val distanceDeltaMeters = p1.location.getDistance(p0.location).value
          val speedMps = p1.speed.convert[meters_per_second].value
          val integratedSpeedMps = distanceDeltaMeters / timestampDeltaSeconds
          val cogDeltaDegrees = angleNormalize((p1.course - p0.course)).value
          val distanceToShoreKm = p1.distanceToShore.convert[kilometer].value

          // TODO: Integrated cog, local tod, local month of year.
          // TODO(alexwilson): #neighbours, distance to closest neighbour.

          // We include the absolute time not as a feature, but to make it easy
          // to binary-search for the start and end of time ranges when running
          // under TensorFlow.
          Array(timestampSeconds,
                timestampDeltaSeconds,
                distanceDeltaMeters,
                speedMps,
                integratedSpeedMps,
                cogDeltaDegrees,
                distanceToShoreKm)
      }
      .toIndexedSeq

  def buildTFExampleProto(metadata: VesselMetadata, data: Seq[Array[Double]]): SequenceExample = {
    val example = SequenceExample.newBuilder()
    val contextBuilder = example.getContextBuilder()

    val vessel_type = metadata.vesselTypeIndex

    // Add the mmsi and vessel type to the builder
    contextBuilder.putFeature(
      "mmsi",
      Feature.newBuilder().setInt64List(Int64List.newBuilder().addValue(metadata.mmsi)).build())
    contextBuilder.putFeature(
      "vessel_type_index",
      Feature.newBuilder().setInt64List(Int64List.newBuilder().addValue(vessel_type)).build())

    contextBuilder.putFeature(
      "vessel_type_name",
      Feature
        .newBuilder()
        .setBytesList(
          BytesList.newBuilder().addValue(ByteString.copyFromUtf8(metadata.vesselType)))
        .build())

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
    featureLists.putFeatureList("movement_features", sequenceData.build())
    example.setFeatureLists(featureLists)

    example.build()
  }

  def buildVesselFeatures(input: SCollection[(VesselMetadata, Seq[VesselLocationRecord])])
    : SCollection[(VesselMetadata, SequenceExample)] =
    input.filter { case (metadata, records) => records.size() >= 3 }.map {
      case (metadata, records) =>
        val features = buildSingleVesselFeatures(records)
        val featuresAsTFExample = buildTFExampleProto(metadata, features)
        (metadata, featuresAsTFExample)
    }

  def main(argArray: Array[String]) {
    val now = new DateTime(DateTimeZone.UTC)
    val (options, remaining_args) = ScioContext.parseArguments[DataflowPipelineOptions](argArray)
    options.setRunner(classOf[DataflowPipelineRunner])
    options.setProject(Parameters.gceProject)
    options.setStagingLocation(Parameters.dataflowStaging)

    val sc = ScioContext(options)

    // Load up the vessel metadata as a side input and join
    // with the JSON records for known vessels.
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

    Parameters.splits.foreach { split =>
      val outputPath =
        Parameters.outputFeaturesPath + "/" +
          ISODateTimeFormat.basicDateTimeNoMillis().print(now) + "/" +
          split

      val filteredFeatures = features.filter { case (md, _) => md.dataset == split }.map {
        case (_, example) => example.asInstanceOf[MessageLite]
      }

      filteredFeatures.internal.apply(
        "WriteTFRecords",
        Write.to(new TFRecordSink(outputPath + "/", "tfrecord", "shard-SSSSS-of-NNNNN")))
    }

    sc.close()
  }
}
