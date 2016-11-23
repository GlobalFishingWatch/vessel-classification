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
import org.skytruth.common.{LatLon}
import org.skytruth.dataflow.{TFRecordSink, TFRecordUtils}

import scala.collection.{mutable, immutable}
import scala.collection.JavaConversions._

import com.spotify.scio._
import resource._


case class Adjacency(
    numNeighbours: Int,
    closestNeighbour: Option[(VesselMetadata, DoubleU[kilometer], ResampledVesselLocation)])

case class VesselLocationRecordWithAdjacency(location: VesselLocationRecord, adjacency: Adjacency)

case class ProcessedAdjacencyLocations(locations: Seq[VesselLocationRecordWithAdjacency],
                                       stationaryPeriods: Seq[StationaryPeriod])

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
  // Normalize from -180 to + 180
  def angleNormalize(angle: DoubleU[degrees]) =
    MathUtils.normalizeAngle(angle.convert[radians].value, 0.0).of[radians].convert[degrees]

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

  def resampleVesselSeries(increment: Duration,
                           input: Seq[VesselLocationRecord]): Seq[ResampledVesselLocation] = {
    val incrementSeconds = increment.getStandardSeconds()
    val maxInterpolateGapSeconds = Parameters.maxInterpolateGap.getStandardSeconds()
    def tsToUnixSeconds(timestamp: Instant): Long = (timestamp.getMillis / 1000L)
    def roundToIncrement(timestamp: Instant): Long =
      (tsToUnixSeconds(timestamp) / incrementSeconds) * incrementSeconds

    var iterTime = roundToIncrement(input.head.timestamp)
    val endTime = roundToIncrement(input.last.timestamp)

    var iterLocation = input.iterator

    val interpolatedSeries = mutable.ListBuffer.empty[ResampledVesselLocation]
    var lastLocationRecord: Option[VesselLocationRecord] = None
    var currentLocationRecord = iterLocation.next()
    while (iterTime <= endTime) {
      while (tsToUnixSeconds(currentLocationRecord.timestamp) < iterTime && iterLocation.hasNext) {
        lastLocationRecord = Some(currentLocationRecord)
        currentLocationRecord = iterLocation.next()
      }

      lastLocationRecord.foreach { llr =>
        val firstTimeSeconds = tsToUnixSeconds(llr.timestamp)
        val secondTimeSeconds = tsToUnixSeconds(currentLocationRecord.timestamp)
        val timeDeltaSeconds = secondTimeSeconds - firstTimeSeconds

        val pointDensity = math.min(1.0, incrementSeconds.toDouble / timeDeltaSeconds.toDouble)

        if (firstTimeSeconds <= iterTime && secondTimeSeconds >= iterTime &&
            timeDeltaSeconds < maxInterpolateGapSeconds) {
          val mix = (iterTime - firstTimeSeconds).toDouble / (secondTimeSeconds - firstTimeSeconds).toDouble

          val interpLat = currentLocationRecord.location.lat.value * mix +
              llr.location.lat.value * (1.0 - mix)
          val interpLon = currentLocationRecord.location.lon.value * mix +
              llr.location.lon.value * (1.0 - mix)

          val interpDistFromShore = currentLocationRecord.distanceToShore.value * mix +
              llr.distanceToShore.value * (1.0 - mix)

          interpolatedSeries.append(
            ResampledVesselLocation(new Instant(iterTime * 1000),
                                    LatLon(interpLat.of[degrees], interpLon.of[degrees]),
                                    interpDistFromShore.of[kilometer],
                                    pointDensity))
        }
      }

      iterTime += incrementSeconds
    }

    interpolatedSeries.toIndexedSeq
  }
}
