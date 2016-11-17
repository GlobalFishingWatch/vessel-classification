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
import com.google.cloud.dataflow.sdk.options.{
  DataflowPipelineOptions,
  PipelineOptions,
  PipelineOptionsFactory
}
import com.google.cloud.dataflow.sdk.runners.{DataflowPipelineRunner}
import com.google.cloud.dataflow.sdk.io.{FileBasedSink, Write}
import com.google.cloud.dataflow.sdk.util.{GcsUtil}
import com.google.cloud.dataflow.sdk.util.gcsfs.{GcsPath}
import com.opencsv.CSVReader
import java.io.{File, FileOutputStream, FileReader, InputStream, OutputStream}
import java.lang.{RuntimeException}
import java.nio.channels.Channels
import java.nio.charset.StandardCharsets
import org.apache.commons.math3.util.MathUtils
import org.joda.time.{DateTime, DateTimeZone, Duration, Instant, LocalDateTime}
import org.joda.time.format.ISODateTimeFormat
import org.json4s._
import org.json4s.JsonDSL.WithDouble._
import org.json4s.native.JsonMethods._
import org.skytruth.common.AdditionalUnits._
import org.skytruth.common.{GcpConfig, LatLon, ValueCache}
import org.skytruth.common.Implicits._
import org.skytruth.common.ScioContextResource._
import org.skytruth.dataflow.{TFRecordSink, TFRecordUtils}

import scala.collection.{mutable, immutable}
import scala.collection.JavaConversions._
import scala.math

import resource._

import org.apache.commons.lang3.builder.ToStringBuilder._

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

  lazy val blacklistedMmsis = Set(0, 12345)

  // Reads JSON vessel records, filters to only location records, groups by MMSI and sorts
  // by ascending timestamp.
  def readJsonRecords(
      inputs: Seq[SCollection[TableRow]],
      knownFishingMMSIs: Set[Int],
      minRequiredPositions: Long): SCollection[(VesselMetadata, Seq[VesselLocationRecord])] = {

    val input = SCollection.unionAll(inputs)
    // Keep only records with a location.
    val validRecords = input
      .filter(json => json.containsKey("lat") && json.containsKey("lon"))
      // Build a typed location record with units of measure.
      .map(json => {
        val mmsi = json.getLong("mmsi").toInt
        val metadata = VesselMetadata(mmsi, knownFishingMMSIs.contains(mmsi))
        val record =
          // TODO(alexwilson): Double-check all these units are correct.
          VesselLocationRecord(Instant.parse(json.getString("timestamp")),
                               LatLon(Utility.angleNormalize(json.getDouble("lat").of[degrees]),
                                      Utility.angleNormalize(json.getDouble("lon").of[degrees])),
                               (json.getDouble("distance_from_shore") / 1000.0).of[kilometer],
                               json.getDouble("speed").of[knots],
                               Utility.angleNormalize(json.getDouble("course").of[degrees]),
                               Utility.angleNormalize(json.getDouble("heading").of[degrees]))
        (metadata, record)
      })
      .filter { case (metadata, _) => !blacklistedMmsis.contains(metadata.mmsi) }
      .filter {
        case (_, record) =>
          RangeValidator()
            .inRange(record.timestamp, Parameters.minValidTime, Parameters.maxValidTime)
            .inRange(record.location.lat, -90.0.of[degrees], 90.0.of[degrees])
            .inRange(record.location.lon, -180.0.of[degrees], 180.of[degrees])
            .inRange(record.distanceToShore, 0.0.of[kilometer], 20000.0.of[kilometer])
            .inRange(record.speed, 0.0.of[knots], 100.0.of[knots])
            .inRange(record.course, -180.0.of[degrees], 180.of[degrees])
            .inRange(record.heading, -180.0.of[degrees], 180.of[degrees])
            .valid
      }

    // Do an early filtering of MMSIs with insufficient points, to improve the downstream
    // performance of the pipeline. Doing this with countByKey is much faster than
    // doing this via a groupBy.
    val mmsisWithSufficientPoints =
      validRecords.countByKey.filter(_._2 >= minRequiredPositions).map(_._1).asListSideInput

    val allowedMMSIs = ValueCache[Set[VesselMetadata]]()
    var cachedMap: Option[Set[VesselMetadata]] = None
    val filteredValidRecords = validRecords
      .withSideInputs(mmsisWithSufficientPoints)
      .filter {
        case ((vmd, _), ctx) =>
          val mmsiSet = allowedMMSIs.get(() => ctx(mmsisWithSufficientPoints).toSet)

          mmsiSet.contains(vmd)
      }
      .toSCollection

    filteredValidRecords.groupByKey.map {
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

  def removeStationaryPeriods(records: Iterable[VesselLocationRecord]): ProcessedLocations = {
    // Remove long stationary periods from the record: anything over the threshold
    // time will be reduced to just the start and end points of the period.
    // TODO(alexwilson): Tim points out that leaves vessels sitting around for t - delta looking
    // significantly different from those sitting around for t + delta. Consider his scheme of just
    // cropping all excess time over the threshold instead.
    val withoutLongStationaryPeriods = mutable.ListBuffer.empty[VesselLocationRecord]
    val stationaryPeriods = mutable.ListBuffer.empty[StationaryPeriod]
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
            val numPoints = currentPeriod.length.toDouble
            val duration = new Duration(periodFirst.timestamp, currentPeriod.last.timestamp)
            val aveLatLon = LatLon.mean(currentPeriod.map { _.location })
            val meanDistanceToShore = currentPeriod.map { _.distanceToShore }.mean
            val meanDriftRadius = currentPeriod.map { _.location.getDistance(aveLatLon) }.mean

            stationaryPeriods.append(
              StationaryPeriod(aveLatLon, duration, meanDistanceToShore, meanDriftRadius))
          } else {
            withoutLongStationaryPeriods ++= currentPeriod
          }

          currentPeriod.clear()
        }
      }

      currentPeriod.enqueue(vr)
    }
    withoutLongStationaryPeriods ++= currentPeriod

    ProcessedLocations(withoutLongStationaryPeriods.toIndexedSeq, stationaryPeriods.toIndexedSeq)
  }

  def filterAndProcessVesselRecords(
      input: SCollection[(VesselMetadata, Seq[VesselLocationRecord])])
    : SCollection[(VesselMetadata, ProcessedLocations)] = {
    input.map {
      case (metadata, records) =>
        val thinnedPoints = thinPoints(records)
        val processedLocations = removeStationaryPeriods(thinnedPoints)

        (metadata, processedLocations)
    }
  }

  def loadFishingMMSIs(): Set[Int] = {
    val fishingMMSIreader = new CSVReader(new FileReader(Parameters.knownFishingMMSIs))
    fishingMMSIreader
      .readAll()
      .map { l =>
        l(0).toInt
      }
      .toSet
  }

  import Utility._

  def main(argArray: Array[String]) {
    val (options, remaining_args) = ScioContext.parseArguments[DataflowPipelineOptions](argArray)

    val environment = remaining_args.required("env")
    val jobName = remaining_args.required("job-name")
    val generateModelFeatures = remaining_args.boolean("generate-model-features", true)
    val generateAnchorages = remaining_args.boolean("generate-anchorages", true)
    val generateAnchorageVisits = remaining_args.boolean("generate-anchorage-visits", true)
    val generateEncounters = remaining_args.boolean("generate-encounters", true)

    val config = GcpConfig.makeConfig(environment, jobName)

    logger.info(s"Pipeline output path: ${config.pipelineOutputPath}")

    options.setRunner(classOf[DataflowPipelineRunner])
    options.setProject(config.projectId)
    options.setStagingLocation(config.dataflowStagingPath)

    managed(ScioContext(options)).acquireAndGet((sc) => {

      // Read, filter and build location records. We build a set of matches for all
      // relevant years, as a single Cloud Dataflow text reader currently can't yet
      // handle the sheer volume of matching files.
      val matches = (Parameters.allDataYears).map { year =>
        val path = Parameters.measuresPathPattern(year)

        sc.tableRowJsonFile(path)
      }

      val knownFishingMMSIs = loadFishingMMSIs()

      val minValidLocations = 200
      val locationRecords: SCollection[(VesselMetadata, Seq[VesselLocationRecord])] =
        readJsonRecords(matches, knownFishingMMSIs, Parameters.minRequiredPositions)

      val adjacencies =
        Encounters.calculateAdjacency(Parameters.adjacencyResamplePeriod, locationRecords)

      val processed =
        filterAndProcessVesselRecords(locationRecords)
      val processedWithAdjecency = Encounters.annotateAdjacency(processed, adjacencies)

      val anchoragePoints =
        Anchorages.findAnchoragePointCells(processed)
      val anchorages = Anchorages.buildAnchoragesFromAnchoragePoints(anchoragePoints)

      val anchorageVisitsPath = config.pipelineOutputPath + "/anchorage_group_visits"

      val anchorageVisits =
        Anchorages
          .findAnchorageVisits(locationRecords, anchorages, Parameters.minAnchorageVisitDuration)

      anchorageVisits.flatMap {
        case (metadata, visits) =>
          visits.map((visit) => {
            compact(
              render(("mmsi" -> metadata.mmsi) ~
                ("visit" -> visit.toJson)))
          })
      }.saveAsTextFile(anchorageVisitsPath)

      if (generateModelFeatures) {
        val features = ModelFeatures.buildVesselFeatures(processedWithAdjecency, anchorages).map {
          case (md, feature) =>
            (s"${md.mmsi}", feature)
        }
        // Output vessel classifier features.
        val outputFeaturePath = config.pipelineOutputPath + "/features"
        val res = Utility.oneFilePerTFRecordSink(outputFeaturePath, features)
      }

      if (generateAnchorages) {
        // Output anchorages points.
        val anchoragePointsPath = config.pipelineOutputPath + "/anchorage_points"
        anchoragePoints.map { anchoragePoint =>
          compact(render(anchoragePoint.toJson))
        }.saveAsTextFile(anchoragePointsPath)

        // And anchorages.
        val anchoragesPath = config.pipelineOutputPath + "/anchorages"
        anchorages.map { anchorage =>
          compact(render(anchorage.toJson))
        }.saveAsTextFile(anchoragesPath)

        val anchorageVisitsPath = config.pipelineOutputPath + "/anchorage_group_visits"

        if (generateAnchorageVisits) {
          val anchorageVisits =
            Anchorages.findAnchorageVisits(locationRecords,
                                           anchorages,
                                           Parameters.minAnchorageVisitDuration)

          anchorageVisits.map {
            case (metadata, visits) =>
              compact(
                render(("mmsi" -> metadata.mmsi) ~
                  ("visits" -> visits.map(_.toJson))))
          }.saveAsTextFile(anchorageVisitsPath)
        }
      }

      if (generateEncounters) {
        val adjacencyAnnotated =
          Encounters.annotateAdjacency(Parameters.adjacencyResamplePeriod, locationRecords)

        // Build and output suspected encounters.
        val suspectedEncountersPath = config.pipelineOutputPath + "/encounters"
        val encounters =
          Encounters.calculateEncounters(Parameters.minDurationForEncounter, adjacencies)
        encounters.map(ec => compact(render(ec.toJson))).saveAsTextFile(suspectedEncountersPath)
      }

      // Get a list of all MMSIs to save to disk to speed up TF training startup.
      val mmsiListPath = config.pipelineOutputPath + "/mmsis"
      processed.keys.saveAsTextFile(mmsiListPath)
    })
  }
}
