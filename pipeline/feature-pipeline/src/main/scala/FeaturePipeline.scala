// Copyright 2017 Google Inc. and Skytruth Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
import org.joda.time.{Duration, Instant}
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
    val dataYearsArg = remaining_args.list("data-years")
    val dataFileGlob =
      remaining_args.getOrElse("data-file-glob", InputDataParameters.defaultDataFileGlob)
    val PUBSUB_TIMESTAMP_LABEL_KEY = "timestamp"


    val config = GcpConfig.makeConfig(environment, jobName)

    logger.info(s"Pipeline output path: ${config.pipelineOutputPath}")

    options.setRunner(classOf[DataflowPipelineRunner])
    options.setProject(config.projectId)
    options.setStagingLocation(config.dataflowStagingPath)

    managed(ScioContext(options)).acquireAndGet((sc) => {
      logger.info("Finding matching files.")
      // Read, filter and build location records. We build a set of matches for all
      // relevant years, as a single Cloud Dataflow text reader currently can't yet
      // handle the sheer volume of matching files.
      // val aisInputData = InputDataParameters
      //   .dataFileGlobPerYear(dataYearsArg, dataFileGlob)
      //   .map(glob => sc.textFile(glob))

      logger.info("Building pipeline.")
      val knownFishingMMSIs = AISDataProcessing.loadFishingMMSIs()

      val minValidLocations = 200

      // TODO -- add to opts
      // val input = sc.pubsubTopic("projects/aju-vtests2/topics/shipping")
      val input = sc.pubsubTopic("projects/aju-vtests2/topics/shipping",
        timestampLabel = PUBSUB_TIMESTAMP_LABEL_KEY)
      val wstream = input
        .withSlidingWindows(
                Duration.standardMinutes(60),
                Duration.standardMinutes(10))

      val locationRecords: SCollection[(VesselMetadata, Seq[VesselLocationRecord])] =
        AISDataProcessing.readJsonRecordsStreaming(wstream,
                                          knownFishingMMSIs,
                                          InputDataParameters.minRequiredPositions)

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
        val features =
          ModelFeatures.buildVesselFeatures(locationsWithAdjacency, anchoragesRootPath).map {
            case (md, feature) =>
              (s"${md.mmsi}", feature)
          }
        val featuress =
          ModelFeatures.buildVesselFeaturesStreaming(locationsWithAdjacency, anchoragesRootPath).map {
            case (md, feature) =>
              logger.info("got feature...")
              // TODO: This is not correct; just experimenting. Need to generate json.
              var hmmm = ""
              feature.foreach { elt =>
                val estring = elt.mkString(", ")
                hmmm = hmmm + ", [" + estring + "]"
              }
              // (s"${md.mmsi}", feature)
              // feature.mkString(", ")
              hmmm
          }
        featuress
          .saveAsPubsub("projects/aju-vtests2/topics/gfwfeatures")

        // Also output vessel classifier features.
        val outputFeaturePath = config.pipelineOutputPath + "/features"
        val res = Utility.oneFilePerTFRecordSink(outputFeaturePath, features)
      }

      // aju TODO - the write gives an error in streaming mode.  Not clear what it should be
      // replaced by.
      // Get a list of all MMSIs to save to disk to speed up TF training startup.
      // val mmsiListPath = config.pipelineOutputPath + "/mmsis"
      // processed.keys.groupAll.flatMap(_.map(md => s"${md.mmsi}")).saveAsTextFile(mmsiListPath)

      logger.info("Launching pipeline.")
    })
  }
}
