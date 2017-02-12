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
import com.spotify.scio.values.WindowOptions
import com.typesafe.scalalogging.{LazyLogging, Logger}
import com.google.cloud.dataflow.sdk.options.{
  DataflowPipelineOptions,
  PipelineOptions,
  PipelineOptionsFactory
}
import com.google.cloud.dataflow.sdk.runners.{DataflowPipelineRunner}
import com.google.cloud.dataflow.sdk.transforms.windowing._
import org.joda.time.{Duration, Instant}
import org.json4s._
import org.json4s.JsonAST.JValue
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

      logger.info("Building pipeline.")
      val knownFishingMMSIs = AISDataProcessing.loadFishingMMSIs()

      val minValidLocations = 200
      val anchoragesLookupCache = ValueCache[AdjacencyLookup[Anchorage]]()
      val anchoragesLookup = anchoragesLookupCache.get { () =>
          Anchorages.getAnchoragesLookup(anchoragesRootPath)
        }


      // TODO -- add to opts
      // val input = sc.pubsubTopic("projects/aju-vtests2/topics/shipping",
      val input = sc.pubsubTopic("projects/earth-outreach/topics/shipping",
        timestampLabel = PUBSUB_TIMESTAMP_LABEL_KEY)
      // This is not the final window/trigger def'n that we want.
      // aju TODO: window params
      val wstream = input
        .withFixedWindows(Duration.standardHours(12),
        // .withSlidingWindows(Duration.standardHours(2),
          // Duration.standardMinutes(15),
          options = WindowOptions(
            trigger = AfterWatermark.pastEndOfWindow()
              .withLateFirings(AfterProcessingTime.pastFirstElementInPane()
                .plusDelayOf(Duration.standardMinutes(10))),
            // accumulationMode = ACCUMULATING_FIRED_PANES,
            accumulationMode = DISCARDING_FIRED_PANES,
            allowedLateness = Duration.standardDays(60)  // aju TODO: fix this
            )
          )

      val locationRecords: SCollection[(VesselMetadata, Seq[VesselLocationRecord])] =
        AISDataProcessing.readJsonRecordsStreaming(wstream,
                                          knownFishingMMSIs,
                                          InputDataParameters.minRequiredPositions)

      val processed =
        AISDataProcessing.filterAndProcessVesselRecords(
          locationRecords,
          InputDataParameters.stationaryPeriodMinDuration)

      // aju  - deleted the "encounters" processing for this pipeline
      val locationsWithEmptyAdjacencyx = processed.map {
        case (vmd, pl) =>
          val locationsWithEmptyAdjacency =
            pl.locations.map(vlr => VesselLocationRecordWithAdjacency(vlr, Adjacency(0, None)))

          (vmd, locationsWithEmptyAdjacency)
      }

      val s2level = 13  // ~1 km
      val featuresStreaming =
        ModelFeatures.buildVesselFeaturesStreaming(locationsWithEmptyAdjacencyx, anchoragesLookup)
         .withTimestamp
         .map {
          case ((md, firstLoc, feature), ts) =>
            val flist =  feature.map { f => f.toList }
            val json = ("mmsi" -> s"${md.mmsi}") ~
                        ("s2CellId" -> firstLoc.getS2CellId(s2level).id) ~
                        ("firstTimestamp" -> Math.round(flist(0)(0)) * 1000) ~
                        ("firstTimestampStr" -> new Instant(Math.round(flist(0)(0)) * 1000).toString) ~
                        ("windowTimestampStr" -> ts.toString) ~
                        ("windowTimestamp" -> ts.getMillis) ~
                        ("feature" -> flist)
            compact(render(json))
        }
      // aju TODO: add timestamp attr to pubsub element ('firstTimestamp'??)
      featuresStreaming
        // .saveAsPubsub("projects/aju-vtests2/topics/gfwfeatures")   // TODO - add to opts
        .saveAsPubsub("projects/earth-outreach/topics/gfwfeatures")   // TODO - add to opts

      // aju - temp removing the other output branch. TODO: add back in properly.
      // val features =
      //   ModelFeatures.buildVesselFeatures(locationsWithEmptyAdjacencyx, anchoragesRootPath).map {
      //     case (md, feature) =>
      //       (s"${md.mmsi}", feature)
      //   }
        // Array[Double](timestampSeconds,
        //                             math.log(1.0 + timestampDeltaSeconds),
        //                             math.log(1.0 + distanceDeltaMeters),
        //                             math.log(1.0 + speedMps),
        //                             math.log(1.0 + integratedSpeedMps),
        //                             cogDeltaDegrees / 180.0,
        //                             localTodFeature,
        //                             localMonthOfYearFeature,
        //                             integratedCogDeltaDegrees / 180.0,
        //                             math.log(1.0 + distanceToShoreKm),
        //                             math.log(1.0 + distanceToBoundingAnchorageKm),
        //                             math.log(1.0 + timeToBoundingAnchorageS),
        //                             a0.numNeighbours)


      // Also output vessel classifier features.
      // val outputFeaturePath = config.pipelineOutputPath + "/features"
      // val res = Utility.oneFilePerTFRecordSink(outputFeaturePath, features)
      // }

      // aju TODO - the write gives an error in streaming mode.  Not clear what it should be
      // replaced by.
      // Get a list of all MMSIs to save to disk to speed up TF training startup.
      // val mmsiListPath = config.pipelineOutputPath + "/mmsis"
      // processed.keys.groupAll.flatMap(_.map(md => s"${md.mmsi}")).saveAsTextFile(mmsiListPath)

      logger.info("Launching pipeline.")
    })
  }
}
