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

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory
import com.fasterxml.jackson.module.scala.DefaultScalaModule

import io.github.karols.units._
import io.github.karols.units.SI._
import io.github.karols.units.defining._
import org.joda.time.{Duration, Instant}

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


case class PipelineConfig(
    inputFilePatterns: Seq[String],
    knownFishingMMSIs: String,
    anchoragesRootPath: String,
    minRequiredPositions: Long,
    encounterMinHours: Long,
    encounterMaxKilometers: Double
)


object Pipeline extends LazyLogging {

  def readYamlConfig(config: String): PipelineConfig = {
    val mapper: ObjectMapper = new ObjectMapper(new YAMLFactory())
    mapper.registerModule(DefaultScalaModule)

    mapper.readValue(config, classOf[PipelineConfig])
  }

  def main(argArray: Array[String]) {
    val (options, remaining_args) = ScioContext.parseArguments[DataflowPipelineOptions](argArray)

    val environment = remaining_args.required("env")
    val jobName = remaining_args.required("job-name")
    val generateModelFeatures = remaining_args.boolean("generate-model-features", true)
    val generateEncounters = remaining_args.boolean("generate-encounters", true)
    val jobConfigurationFile = remaining_args.required("job-config")

    val config = GcpConfig.makeConfig(environment, jobName)

    logger.info(s"Pipeline output path: ${config.pipelineOutputPath}")

    options.setRunner(classOf[DataflowPipelineRunner])
    options.setProject(config.projectId)
    options.setStagingLocation(config.dataflowStagingPath)

    val pipelineConfig = managed(scala.io.Source.fromFile(jobConfigurationFile)).acquireAndGet {
      s => readYamlConfig(s.mkString)
    }

    managed(ScioContext(options)).acquireAndGet((sc) => {
      logger.info("Finding matching files.")
      // Read, filter and build location records. We build a set of matches for all
      // relevant years, as a single Cloud Dataflow text reader currently can't yet
      // handle the sheer volume of matching files.
      val aisInputData = pipelineConfig.inputFilePatterns.map(glob => sc.textFile(glob))

      logger.info("Building pipeline.")
      val knownFishingMMSIs = AISDataProcessing.loadFishingMMSIs(pipelineConfig.knownFishingMMSIs)

      val locationRecords: SCollection[(VesselMetadata, Seq[VesselLocationRecord])] =
        AISDataProcessing.readJsonRecords(aisInputData,
                                          knownFishingMMSIs,
                                          pipelineConfig.minRequiredPositions)


      val processed =
        AISDataProcessing.filterAndProcessVesselRecords(
          locationRecords,
          InputDataParameters.stationaryPeriodMinDuration)

      val locationsWithAdjacency = if (generateEncounters) {
        val maxEncounterRadius = 2 * pipelineConfig.encounterMaxKilometers.of[kilometer]
        val adjacencies =
          Encounters.calculateAdjacency(Parameters.adjacencyResamplePeriod, locationRecords, 
                      maxEncounterRadius)

        // Build and output suspected encounters.
        val suspectedEncountersPath = config.pipelineOutputPath + "/encounters"
        val encounters =
          Encounters.calculateEncounters(Duration.standardHours(pipelineConfig.encounterMinHours), adjacencies, 
              pipelineConfig.encounterMaxKilometers.of[kilometer])
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
          ModelFeatures.buildVesselFeatures(locationsWithAdjacency, pipelineConfig.anchoragesRootPath).map {
            case (md, feature) =>
              (s"${md.mmsi}", feature)
          }
        // Output vessel classifier features.
        val outputFeaturePath = config.pipelineOutputPath + "/features"
        val res = Utility.oneFilePerTFRecordSink(outputFeaturePath, features)

        // Get a list of all MMSIs to save to disk to speed up TF training startup.
        val mmsiListPath = config.pipelineOutputPath + "/mmsis"
        features.keys.groupAll.flatMap(_.map(md => md)).saveAsTextFile(mmsiListPath)

      }

      logger.info("Launching pipeline.")
    })
  }
}
