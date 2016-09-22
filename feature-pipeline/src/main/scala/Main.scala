package org.skytruth.feature_pipeline

import io.github.karols.units._
import io.github.karols.units.SI._
import io.github.karols.units.defining._

import com.google.common.geometry.{S2, S2LatLng}
import com.google.protobuf.{ByteString, MessageLite}
import com.spotify.scio._
import com.spotify.scio.values.SCollection
import com.spotify.scio.bigquery._
import com.typesafe.scalalogging.{LazyLogging, Logger}
import com.google.cloud.dataflow.sdk.options.{DataflowPipelineOptions, PipelineOptionsFactory}
import com.google.cloud.dataflow.sdk.runners.{DataflowPipelineRunner}
import com.google.cloud.dataflow.sdk.io.{Write}
import com.opencsv.CSVReader
import java.io.{File, FileReader, InputStream, OutputStream}
import java.lang.{RuntimeException}
import java.nio.charset.StandardCharsets
import org.apache.commons.math3.util.MathUtils
import org.joda.time.{DateTime, DateTimeZone, Duration, Instant, LocalDateTime}
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
import scala.math

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

  val minValidTime = Instant.parse("2012-01-01T00:00:00Z")
  lazy val maxValidTime = Instant.now()

  val trainingSplit = "Training"
  val testSplit = "Test"
  val unclassifiedSplit = "Unclassified"
  val splits = Seq(trainingSplit, testSplit, unclassifiedSplit)
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
    weight: Float = 1.0f,
    dataset: String = Parameters.unclassifiedSplit,
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

case class RangeValidator(valid: Boolean) extends AnyVal {
  def inRange[T <: Ordered[T]](value: T, min: T, max: T) =
    RangeValidator(valid && (value >= min && value < max))
  // TODO(alexwilson): Both of these could be removed if Instant and MUnit can be
  // made to be Ordered, with appropriate use of implicits.
  def inRange(value: Instant, min: Instant, max: Instant) =
    RangeValidator(valid && (value.getMillis >= min.getMillis && value.getMillis < max.getMillis))
  def inRange[T <: MUnit](value: DoubleU[T], min: DoubleU[T], max: DoubleU[T]) =
    RangeValidator(valid && (value.value >= min.value && value.value < max.value))
}

object RangeValidator {
  def apply() = new RangeValidator(true)
}

object Pipeline extends LazyLogging {
  implicit class RichLogger(val logger: Logger) {
    def fatal(message: String) = {
      logger.error(message)
      throw new RuntimeException(s"Fatal error: $message")
    }
  }

  lazy val blacklistedMmsis = Set(0, 12345)

  // Normalize from -180 to + 180
  def angleNormalize(angle: DoubleU[degrees]) =
    MathUtils.normalizeAngle(angle.convert[radians].value, 0.0).of[radians].convert[degrees]

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
                               LatLon(angleNormalize(json.getDouble("lat").of[degrees]),
                                      angleNormalize(json.getDouble("lon").of[degrees])),
                               json.getDouble("distance_to_shore").of[kilometer],
                               json.getDouble("distance_to_port").of[kilometer],
                               json.getDouble("speed").of[knots],
                               angleNormalize(json.getDouble("course").of[degrees]),
                               angleNormalize(json.getDouble("heading").of[degrees]))
        (metadata, record)
      })
      .toSCollection
      .filter { case (metadata, _) => !blacklistedMmsis.contains(metadata.mmsi) }
      .filter {
        case (_, record) =>
          RangeValidator()
            .inRange(record.timestamp, Parameters.minValidTime, Parameters.maxValidTime)
            .inRange(record.location.lat, -90.0.of[degrees], 90.0.of[degrees])
            .inRange(record.location.lon, -180.0.of[degrees], 180.of[degrees])
            .inRange(record.distanceToShore, 0.0.of[kilometer], 20000.0.of[kilometer])
            .inRange(record.distanceToPort, 0.0.of[kilometer], 20000.0.of[kilometer])
            .inRange(record.speed, 0.0.of[knots], 100.0.of[knots])
            .inRange(record.course, -180.0.of[degrees], 180.of[degrees])
            .inRange(record.heading, -180.0.of[degrees], 180.of[degrees])
            .valid
      }
      .groupByKey
      .map {
        case (metadata, records) =>
          (metadata,
           records.toIndexedSeq
           // On occasion the same message seems to appear twice in the record. Remove.
           .distinct.sortBy(_.timestamp.getMillis))
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
            if (currentPeriod.last != periodFirst) {
              withoutLongStationaryPeriods.append(currentPeriod.last)
            }
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

  def buildSingleVesselFeatures(input: Seq[VesselLocationRecord]): Seq[Array[Double]] =
    input
      .sliding(3)
      .map {
        case Seq(p0, p1, p2) =>
          if (p0 == p1) {
            logger.fatal(s"p0 and p1 are the same: $p0, $p1")
          }
          val ll0 = S2LatLng.fromDegrees(p0.location.lat.value, p0.location.lon.value)
          val ll1 = S2LatLng.fromDegrees(p1.location.lat.value, p1.location.lon.value)
          val ll2 = S2LatLng.fromDegrees(p2.location.lat.value, p2.location.lon.value)
          val timestampSeconds = p1.timestamp.getMillis / 1000
          val timestampDeltaSeconds = p1.timestamp.getMillis / 1000 - p0.timestamp.getMillis / 1000
          val distanceDeltaMeters = p1.location.getDistance(p0.location).value
          val speedMps = p1.speed.convert[meters_per_second].value
          val integratedSpeedMps = distanceDeltaMeters / timestampDeltaSeconds
          val cogDeltaDegrees = angleNormalize((p1.course - p0.course)).value
          val integratedCogDeltaDegrees = S2
            .turnAngle(ll0.normalized().toPoint(),
                       ll1.normalized().toPoint(),
                       ll2.normalized().toPoint())
            .of[radians]
            .convert[degrees]
            .value
          val distanceToShoreKm = p1.distanceToShore.convert[kilometer].value

          // Calculate time features. TODO(alexwilson): Add tests.
          val longitudeTzOffsetSeconds = 60 * 60 * 12 * (p1.location.lon
              .convert[degrees]
              .value / 180.0)
          val offsetTimezone = DateTimeZone.forOffsetMillis(longitudeTzOffsetSeconds.toInt * 1000)
          val localTime = new LocalDateTime(timestampSeconds, offsetTimezone)
          val localTodFeature = ((localTime
              .getHourOfDay() * (localTime.getMinuteOfHour() / 60.0)) - 12.0) / 12.0
          val localMonthOfYearFeature = (localTime.getMonthOfYear() - 6.0) / 6.0

          // TODO(alexwilson): #neighbours, distance to closest neighbour, is_dark.

          // We include the absolute time not as a feature, but to make it easy
          // to binary-search for the start and end of time ranges when running
          // under TensorFlow.
          val feature = Array[Double](timestampSeconds,
                                      math.log(1.0 + timestampDeltaSeconds),
                                      math.log(1.0 + distanceDeltaMeters),
                                      math.log(1.0 + speedMps),
                                      math.log(1.0 + integratedSpeedMps),
                                      cogDeltaDegrees / 180.0,
                                      localTodFeature,
                                      localMonthOfYearFeature,
                                      integratedCogDeltaDegrees / 180.0,
                                      math.log(1.0 + distanceToShoreKm))

          feature.foreach { v =>
            if (v.isNaN || v.isInfinite) {
              logger.fatal(s"Malformed feature: ${feature.toList}, $p0, $p1, $p2")
            }
          }

          feature
      }
      .toIndexedSeq

  def buildTFExampleProto(metadata: VesselMetadata, data: Seq[Array[Double]]): SequenceExample = {
    val example = SequenceExample.newBuilder()
    val contextBuilder = example.getContextBuilder()

    val vessel_type = metadata.vesselTypeIndex

    // Add the mmsi, weight and vessel type to the builder
    contextBuilder.putFeature(
      "mmsi",
      Feature.newBuilder().setInt64List(Int64List.newBuilder().addValue(metadata.mmsi)).build())
    contextBuilder.putFeature(
      "weight",
      Feature.newBuilder().setFloatList(FloatList.newBuilder().addValue(metadata.weight)).build())
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

  def readVesselMetadata(vesselMetadataCsv: String) = {
    val metadataReader = new CSVReader(new FileReader(vesselMetadataCsv))
    val allLines = metadataReader.readAll()
    val unweightedVesselMetadata = allLines.toList
      .drop(1)
      .map { l =>
        val mmsi = l(0).toInt
        val dataset = l(1)
        val vesselType = l(2)

        (mmsi, VesselMetadata(mmsi, 0.0f, dataset, vesselType))
      }
      .toMap

    // Count the number of training vessel types per vessel type.
    val counts = mutable.Map[String, Int]()
    unweightedVesselMetadata.foreach {
      case (_, vm) =>
        if (vm.dataset == Parameters.trainingSplit) {
          counts(vm.vesselType) = counts.getOrElse(vm.vesselType, 0) + 1
        }
    }

    val rarestVesselTypeCount = counts.map(_._2).min
    val vesselTypeSampleProbability = counts.map {
      case (vt, c) =>
        val sampleProbability = rarestVesselTypeCount.toFloat / c.toFloat
        logger.info(s"Vessel type $vt has sample probability $sampleProbability")
        (vt, sampleProbability)
    }.toMap

    val vesselMetadata = unweightedVesselMetadata.map {
      case (mmsi, vm) =>
        val weight = if (vm.dataset == Parameters.trainingSplit) {
          vesselTypeSampleProbability(vm.vesselType)
        } else {
          1.0f
        }
        (mmsi, VesselMetadata(mmsi, weight, vm.dataset, vm.vesselType))
    }

    vesselMetadata
  }

  def main(argArray: Array[String]) {
    val now = new DateTime(DateTimeZone.UTC)
    val (options, remaining_args) = ScioContext.parseArguments[DataflowPipelineOptions](argArray)

    // Load up the vessel metadata as a side input and join
    // with the JSON records for known vessels.
    val vesselMetadataCsv =
      remaining_args.getOrElse("vessel_metadata", Parameters.vesselClassificationPath)
    val vesselMetadata = readVesselMetadata(vesselMetadataCsv)

    options.setRunner(classOf[DataflowPipelineRunner])
    options.setProject(Parameters.gceProject)
    options.setStagingLocation(Parameters.dataflowStaging)

    val sc = ScioContext(options)

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
