package org.skytruth.feature_pipeline

import io.github.karols.units._
import io.github.karols.units.SI._
import io.github.karols.units.defining._

import com.spotify.scio._
import com.spotify.scio.values.SCollection
import com.typesafe.scalalogging.{LazyLogging, Logger}
import com.google.cloud.dataflow.sdk.options.{
  DataflowPipelineOptions,
  PipelineOptions,
  PipelineOptionsFactory
}
import com.google.cloud.dataflow.sdk.runners.{DataflowPipelineRunner}
import org.json4s._
import org.json4s.JsonDSL.WithDouble._
import org.json4s.native.JsonMethods._
import org.skytruth.anchorages._
import org.skytruth.common._
import org.skytruth.common.AdditionalUnits._
import org.skytruth.common.Implicits._
import org.skytruth.common.ScioContextResource._

import scala.collection.{mutable, immutable}

import resource._

object Pipeline extends LazyLogging {

  def main(argArray: Array[String]) {
    val (options, remaining_args) = ScioContext.parseArguments[DataflowPipelineOptions](argArray)

    val environment = remaining_args.required("env")
    val jobName = remaining_args.required("job-name")
    val generateModelFeatures = remaining_args.boolean("generate-model-features", true)
    val anchoragesRootPath = remaining_args("anchorages-root-path")
    val generateEncounters = remaining_args.boolean("generate-encounters", true)
    val dataYears = remaining_args.getOrElse("data-years", "2012,2013,2014,2015,2016")

    val config = GcpConfig.makeConfig(environment, jobName)

    logger.info(s"Pipeline output path: ${config.pipelineOutputPath}")

    options.setRunner(classOf[DataflowPipelineRunner])
    options.setProject(config.projectId)
    options.setStagingLocation(config.dataflowStagingPath)

    managed(ScioContext(options)).acquireAndGet((sc) => {

      // Read, filter and build location records. We build a set of matches for all
      // relevant years, as a single Cloud Dataflow text reader currently can't yet
      // handle the sheer volume of matching files.
      val matches = dataYears.split(",").map { year =>
        val path = InputDataParameters.measuresPathPattern(year)

        sc.textFile(path)
      }

      val anchorages = if (!anchoragesRootPath.isEmpty) {
        Anchorage.readAnchorages(anchoragesRootPath)
      } else {
        Seq.empty[Anchorage]
      }

      val knownFishingMMSIs = AISDataProcessing.loadFishingMMSIs()

      val minValidLocations = 200
      val locationRecords: SCollection[(VesselMetadata, Seq[VesselLocationRecord])] =
        AISDataProcessing
          .readJsonRecords(matches, knownFishingMMSIs, InputDataParameters.minRequiredPositions)

      val processed =
        AISDataProcessing.filterAndProcessVesselRecords(
          locationRecords,
          InputDataParameters.stationaryPeriodMinDuration)

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
      processed.keys.groupAll.flatMap(_.map(md => s"${md.mmsi}")).saveAsTextFile(mmsiListPath)
    })
  }
}
