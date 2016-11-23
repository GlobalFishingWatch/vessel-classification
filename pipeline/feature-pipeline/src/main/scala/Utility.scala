package org.skytruth.feature_pipeline

import io.github.karols.units._
import io.github.karols.units.SI._
import io.github.karols.units.defining._

import com.google.protobuf.{ByteString, MessageLite}
import com.google.cloud.dataflow.sdk.runners.{DataflowPipelineRunner}
import com.google.cloud.dataflow.sdk.io.{FileBasedSink, Write}
import com.google.cloud.dataflow.sdk.options.{
  DataflowPipelineOptions,
  PipelineOptions,
  PipelineOptionsFactory
}
import com.google.cloud.dataflow.sdk.util.{GcsUtil}
import com.google.cloud.dataflow.sdk.util.gcsfs.{GcsPath}
import com.typesafe.scalalogging.{LazyLogging, Logger}
import com.spotify.scio._
import com.spotify.scio.values.SCollection
import java.io.{File, FileOutputStream, FileReader, InputStream, OutputStream}
import java.lang.{RuntimeException}
import java.nio.channels.Channels

import org.apache.commons.math3.util.MathUtils
import org.joda.time.{DateTime, DateTimeZone, Duration, Instant, LocalDateTime}
import org.json4s._
import org.json4s.JsonDSL.WithDouble._
import org.json4s.native.JsonMethods._
import org.skytruth.common.AdditionalUnits._
import org.skytruth.common.Implicits._
import org.skytruth.common.ScioContextResource._
import org.skytruth.common._
import org.skytruth.dataflow.{TFRecordSink, TFRecordUtils}

import scala.collection.{mutable, immutable}
import scala.collection.JavaConversions._

import com.spotify.scio._
import resource._

case class Adjacency(
    numNeighbours: Int,
    closestNeighbour: Option[(VesselMetadata, DoubleU[kilometer], ResampledVesselLocation)])

case class VesselLocationRecordWithAdjacency(location: VesselLocationRecord, adjacency: Adjacency)

case class ResampledVesselLocation(timestamp: Instant,
                                   location: LatLon,
                                   distanceToShore: DoubleU[kilometer],
                                   pointDensity: Double)

case class ResampledVesselLocationWithAdjacency(locationRecord: ResampledVesselLocation,
                                                adjacency: Adjacency)

case class SingleEncounter(startTime: Instant,
                           endTime: Instant,
                           meanLocation: LatLon,
                           medianDistance: DoubleU[kilometer],
                           medianSpeed: DoubleU[knots],
                           vessel1PointCount: Int,
                           vessel2PointCount: Int) {
  def toJson =
    ("duration_seconds" -> (new Duration(startTime, endTime)).getStandardSeconds) ~
      ("start_time" -> startTime.toString) ~
      ("end_time" -> endTime.toString) ~
      ("mean_latitude" -> meanLocation.lat.value) ~
      ("mean_longitude" -> meanLocation.lon.value) ~
      ("median_distance" -> medianDistance.value) ~
      ("median_speed" -> medianSpeed.value) ~
      ("vessel1_point_count" -> vessel1PointCount) ~
      ("vessel2_point_count" -> vessel2PointCount)
}

case class VesselEncounters(vessel1: VesselMetadata,
                            vessel2: VesselMetadata,
                            encounters: Seq[SingleEncounter]) {
  def toJson =
    ("vessel1_mmsi" -> vessel1.mmsi) ~
      ("vessel2_mmsi" -> vessel2.mmsi) ~
      ("vessel1_flag_state" -> vessel1.flagState) ~
      ("vessel2_flag_state" -> vessel2.flagState) ~
      ("encounters" -> encounters.map(_.toJson))

}

object Utility extends LazyLogging {
  // TODO(alexwilson): Rolling this ourselves isn't nice. Explore how to do this with existing cloud dataflow sinks.
  def oneFilePerTFRecordSink[T <: MessageLite](basePath: String,
                                               values: SCollection[(String, T)]) = {
    // Write data to temporary files, one per mmsi.
    val tempFiles = values.map {
      case (name, value) =>
        val suffix = scala.util.Random.nextInt
        val tempPath = s"$basePath/$name.tfrecord-tmp-$suffix"
        val finalPath = s"$basePath/$name.tfrecord"

        val pipelineOptions = PipelineOptionsFactory.create()
        val gcsUtil = new GcsUtil.GcsUtilFactory().create(pipelineOptions)

        val channel = gcsUtil.create(GcsPath.fromUri(tempPath), "application/octet-stream")
        val outFs = Channels.newOutputStream(channel)
        TFRecordUtils.write(outFs, value)
        outFs.close()

        (tempPath, finalPath)
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
