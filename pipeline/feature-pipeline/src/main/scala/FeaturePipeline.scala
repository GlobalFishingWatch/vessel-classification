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
import org.skytruth.anchorages._
import org.skytruth.common._
import org.skytruth.common.AdditionalUnits._
import org.skytruth.common.Implicits._
import org.skytruth.common.ScioContextResource._
import org.skytruth.dataflow.{TFRecordSink, TFRecordUtils}

import scala.collection.{mutable, immutable}
import scala.collection.JavaConversions._
import scala.math

import resource._

import org.apache.commons.lang3.builder.ToStringBuilder._

object Pipeline extends LazyLogging {

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
      val matches = (InputDataParameters.allDataYears).map { year =>
        val path = InputDataParameters.measuresPathPattern(year)

        sc.tableRowJsonFile(path)
      }

      val knownFishingMMSIs = AISDataProcessing.loadFishingMMSIs()

      val minValidLocations = 200
      val locationRecords: SCollection[(VesselMetadata, Seq[VesselLocationRecord])] =
        AISDataProcessing
          .readJsonRecords(matches, knownFishingMMSIs, InputDataParameters.minRequiredPositions)

      val processed =
        AISDataProcessing.filterAndProcessVesselRecords(locationRecords)

      val anchorages = if (generateAnchorages) {
        val anchoragePoints =
          Anchorages.findAnchoragePointCells(processed)
        val anchorages = Anchorages.buildAnchoragesFromAnchoragePoints(anchoragePoints)

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

        if (generateAnchorageVisits) {
          val anchorageVisitsPath = config.pipelineOutputPath + "/anchorage_visits"
          val anchorageVisits =
            Anchorages.findAnchorageVisits(locationRecords,
                                           anchorages,
                                           AnchorageParameters.minAnchorageVisitDuration)

          anchorageVisits.map {
            case (metadata, visits) =>
              compact(
                render(("mmsi" -> metadata.mmsi) ~
                  ("visits" -> visits.map(_.toJson))))
          }.saveAsTextFile(anchorageVisitsPath)
        }

        anchorages
      } else {
        sc.parallelize(Seq.empty[Anchorage])
      }

      val locationsWithAdjacency = if (generateEncounters) {
        val adjacencies =
          Encounters.calculateAdjacency(Parameters.adjacencyResamplePeriod, locationRecords)

        // Build and output suspected encounters.
        val suspectedEncountersPath = config.pipelineOutputPath + "/encounters"
        val encounters =
          Encounters.calculateEncounters(Parameters.minDurationForEncounter, adjacencies)
        encounters.map(ec => compact(render(ec.toJson))).saveAsTextFile(suspectedEncountersPath)

        Encounters.annotateAdjacency(processed, adjacencies)
      } else {
        processed.map {
          case (vmd, pl) =>
            val locationsWithEmptyAdjacency =
              pl.locations.map(vlr => VesselLocationRecordWithAdjacency(vlr, Adjacency(0, None)))

            (vmd, locationsWithEmptyAdjacency)
        }
      }

      if (generateModelFeatures) {
        val features = ModelFeatures.buildVesselFeatures(locationsWithAdjacency, anchorages).map {
          case (md, feature) =>
            (s"${md.mmsi}", feature)
        }
        // Output vessel classifier features.
        val outputFeaturePath = config.pipelineOutputPath + "/features"
        val res = Utility.oneFilePerTFRecordSink(outputFeaturePath, features)
      }

      // Get a list of all MMSIs to save to disk to speed up TF training startup.
      val mmsiListPath = config.pipelineOutputPath + "/mmsis"
      processed.keys.saveAsTextFile(mmsiListPath)
    })
  }
}
