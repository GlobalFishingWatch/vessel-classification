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
      val input = sc.pubsubTopic("projects/earth-outreach/topics/shipping",
        timestampLabel = PUBSUB_TIMESTAMP_LABEL_KEY)
      val wstream = input
        .withFixedWindows(Duration.standardHours(6), // TODO: unhardwire
          options = WindowOptions(
            trigger = AfterWatermark.pastEndOfWindow()
              .withLateFirings(AfterProcessingTime.pastFirstElementInPane()
                .plusDelayOf(Duration.standardMinutes(10))), // TODO: unhardwire
            accumulationMode = DISCARDING_FIRED_PANES,
            allowedLateness = Duration.standardDays(60)  // aju TODO: unhardwire
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
          case ((md, firstLoc, timestampsS2Ids, feature), ts) =>
            val flist =  feature.map { f => f.toList }
            val json = ("mmsi" -> s"${md.mmsi}") ~
                        // ("s2CellId" -> firstLoc.getS2CellId(s2level).id) ~
                        ("firstTimestamp" -> Math.round(flist(0)(0)) * 1000) ~
                        ("firstTimestampStr" -> new Instant(Math.round(flist(0)(0)) * 1000).toString) ~
                        ("windowTimestampStr" -> ts.toString) ~
                        ("windowTimestamp" -> ts.getMillis) ~
                        ("feature" -> flist) ~
                        ("timestampsS2Ids" -> timestampsS2Ids)
            compact(render(json))
        }
      // aju TODO: would be nice to add timestamp attr to pubsub element, apparently not exposed currently.
      // Supporting this via separate "connector" script instead.
      featuresStreaming
        // .saveAsPubsub("projects/aju-vtests2/topics/gfwfeatures")   // TODO - add to opts
        .saveAsPubsub("projects/earth-outreach/topics/gfwfeatures")   // TODO - add to opts

      // TODO -- adding this branch back in seems to trigger OOM issues...
      // val features =
      //   ModelFeatures.buildVesselFeatures(locationsWithEmptyAdjacencyx, anchoragesLookup).map {
      //     case (md, feature) =>
      //       (s"${md.mmsi}", feature)
      //   }

      // // Also output vessel classifier features.
      // val outputFeaturePath = config.pipelineOutputPath + "/features"
      // val res = Utility.oneFilePerTFRecordSink(outputFeaturePath, features)

      // aju TODO - this partic write gives an error in streaming mode. Do we
      // still want this?  Not necess just for demo, would be longer-term.
      // Get a list of all MMSIs to save to disk to speed up TF training startup.
      // val mmsiListPath = config.pipelineOutputPath + "/mmsis"
      // processed.keys.groupAll.flatMap(_.map(md => s"${md.mmsi}")).saveAsTextFile(mmsiListPath)

      logger.info("Launching pipeline.")
    })
  }
}
