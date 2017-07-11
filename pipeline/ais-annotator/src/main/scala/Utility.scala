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

package org.skytruth.ais_annotator

import io.github.karols.units._
import io.github.karols.units.SI._
import io.github.karols.units.defining._

import com.google.protobuf.{ByteString, MessageLite}
import com.google.cloud.dataflow.sdk.options.{
  DataflowPipelineOptions,
  PipelineOptions,
  PipelineOptionsFactory
}
import com.google.cloud.dataflow.sdk.util.GcsUtil
import com.google.cloud.dataflow.sdk.util.gcsfs.GcsPath
import com.typesafe.scalalogging.{LazyLogging, Logger}
import com.spotify.scio._
import com.spotify.scio.values.SCollection
import java.io.{File, FileOutputStream, FileReader, InputStream, OutputStream}
import java.nio.channels.Channels

import org.joda.time.{DateTime, DateTimeZone, Duration, Instant, LocalDateTime}
import org.json4s._
import org.json4s.JsonDSL.WithDouble._
import org.json4s.native.JsonMethods._
import org.skytruth.common.AdditionalUnits._
import org.skytruth.common.Implicits._
import org.skytruth.common.ScioContextResource._
import org.skytruth.common._

import scala.collection.{mutable, immutable}
import scala.collection.JavaConversions._

import com.spotify.scio._
import resource._


object Utility extends LazyLogging {
  // TODO(alexwilson): Rolling this ourselves isn't nice. Explore how to do this with existing cloud dataflow sinks.
  def CustomShardedTFRecordSink(basePath: String,
                                               tagged_values: SCollection[(String, Seq[Iterable[String]])]) = {
    // Write data to temporary files, one per value.
    val tempFiles = tagged_values.flatMap {
      case (shardname, values) =>
        val count = values.length
        values.zipWithIndex.map {
          case (value, i) => {
            val suffix = scala.util.Random.nextInt
            val tempPath = s"$basePath/$shardname/tmp-$suffix-$i-of-$count"
            val finalPath = s"$basePath/$shardname/$i-of-$count"
            val pipelineOptions = PipelineOptionsFactory.create()
            val gcsUtil = new GcsUtil.GcsUtilFactory().create(pipelineOptions)

            val channel = gcsUtil.create(GcsPath.fromUri(tempPath), "application/octet-stream")

            val outFs = Channels.newOutputStream(channel)

            outFs.write(value.mkString("\n").getBytes)
            outFs.close()

            (tempPath, finalPath)            
          }
        }
    }

    // Copy the files to their final destinations, then delete.
    tempFiles.map {
      case (tempPath, finalPath) =>
        val pipelineOptions = PipelineOptionsFactory.create()
        val gcsUtil = new GcsUtil.GcsUtilFactory().create(pipelineOptions)

        // TODO(alexwilson): This API is designed for batching copies. It might
        // be better to do this multiple-filenames at a time.
        gcsUtil.copy(List(tempPath), List(finalPath))

        gcsUtil.remove(List(tempPath))

        finalPath
    }
  }
}
