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

package org.skytruth.common

import com.google.cloud.storage.{Storage, StorageOptions}
import com.google.common.io.LineReader

import java.nio.channels.Channels
import java.nio.charset.StandardCharsets.UTF_8

import scala.collection.{mutable, immutable}
import scala.collection.JavaConversions._
import scala.collection.JavaConverters._

case class GoogleCloudStorage(private val storage: Storage) {
  private def parseGcsPath(path: String) = {
    assert(path.take(5) == "gs://")
    val els = path.drop(5).split("/", 2)
    assert(els.size == 2)
    (els(0), els(1))
  }

  private def buildGcsPath(bucket: String, path: String) =
    s"gs://$bucket/$path"

  def list(fullPath: String): Iterator[String] = {
    val (bucket, path) = parseGcsPath(fullPath)
    val result = storage.list(bucket, Storage.BlobListOption.prefix(path))
    result.iterateAll().toIterator.map(b => buildGcsPath(bucket, b.getName))
  }

  def get(fullPath: String): Iterator[String] = {
    val (bucket, path) = parseGcsPath(fullPath)
    var lineReader: Option[LineReader] = Some(
      new LineReader(Channels.newReader(storage.get(bucket, path).reader(), UTF_8.name())))

    new Iterator[String] {
      private def readLine(): Option[String] = {
        lineReader.flatMap { lr =>
          val next = Option(lr.readLine())
          if (!next.isDefined) {
            lineReader = None
          }

          next
        }
      }
      var currentVal: Option[String] = None
      var nextVal = readLine()

      override def hasNext = nextVal.isDefined
      override def next() = {
        currentVal = nextVal
        nextVal = readLine()
        currentVal.get
      }
    }
  }
}

object GoogleCloudStorage {
  def apply() = new GoogleCloudStorage(StorageOptions.defaultInstance().service())
}
