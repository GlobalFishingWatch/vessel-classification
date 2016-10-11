package org.skytruth.feature_pipeline

import io.github.karols.units._
import io.github.karols.units.SI._
import io.github.karols.units.defining._

import com.google.common.geometry.{S2, S2Cap, S2CellId, S2LatLng, S2RegionCoverer}
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
import org.skytruth.dataflow.{TFRecordSink, TFRecordUtils}

import scala.collection.{mutable, immutable}
import scala.collection.JavaConversions._

object AdditionalUnits {
  type knots = DefineUnit[_k ~: _n ~: _o ~: _t]
  type meters_per_second = meter / second

  type degrees = DefineUnit[_d ~: _e ~: _g]
  type radians = DefineUnit[_r ~: _a ~: _d]

  implicit val knots_to_mps = one[knots].contains(0.514444)[meters_per_second]
  implicit val radians_to_degrees = one[degrees].contains(MathUtils.TWO_PI / 360.0)[radians]
}

import AdditionalUnits._

case class LatLon(lat: DoubleU[degrees], lon: DoubleU[degrees]) {
  def getS2LatLng() = S2LatLng.fromDegrees(lat.value, lon.value)

  def getDistance(other: LatLon): DoubleU[kilometer] =
    getS2LatLng().getEarthDistance(other.getS2LatLng()).toDouble.of[meter].convert[kilometer]

  def getS2CellId(level: Int): S2CellId = {
    val cell = S2CellId.fromLatLng(getS2LatLng())
    cell.parent(level)
  }
}

object LatLon {
  def fromS2CellId(cell: S2CellId) = {
    val loc = cell.toLatLng()
    LatLon(loc.latDegrees().of[degrees], loc.lngDegrees().of[degrees])
  }

  def mean(locations: Iterable[LatLon]): LatLon = {
    var lat = 0.0
    var lon = 0.0
    var count = 0
    locations.foreach { l =>
      lat += l.lat.value
      lon += l.lon.value
      count += 1
    }
    LatLon((lat / count.toDouble).of[degrees], (lon / count.toDouble).of[degrees])
  }
}

case class VesselMetadata(
    mmsi: Int
)

case class StationaryPeriod(location: LatLon, duration: Duration)

case class ProcessedLocations(locations: Seq[VesselLocationRecord],
                              stationaryPeriods: Seq[StationaryPeriod])

// TODO(alexwilson): For now build simple coder for this class. :-(
case class VesselLocationRecord(timestamp: Instant,
                                location: LatLon,
                                distanceToShore: DoubleU[kilometer],
                                distanceToPort: DoubleU[kilometer],
                                speed: DoubleU[knots],
                                course: DoubleU[degrees],
                                heading: DoubleU[degrees])

case class ResampledVesselLocation(timestamp: Instant,
                                   location: LatLon,
                                   distanceToShore: DoubleU[kilometer])

case class ResampledVesselLocationWithAdjacency(
    timestamp: Instant,
    location: LatLon,
    distanceToShore: DoubleU[kilometer],
    numNeighbours: Int,
    closestNeighbour: Option[(VesselMetadata, DoubleU[kilometer])])

case class VesselEncounter(vessel1: VesselMetadata,
                           vessel2: VesselMetadata,
                           startTime: Instant,
                           endTime: Instant,
                           meanLocation: LatLon,
                           medianDistance: DoubleU[kilometer]) {
  def toCsvLine =
    s"${vessel1.mmsi},${vessel2.mmsi},$startTime,$endTime,${meanLocation.lat},${meanLocation.lon},${medianDistance}"
}

case class SuspectedPort(location: LatLon, vessels: Seq[VesselMetadata])

object Utility extends LazyLogging {
  implicit class RichLogger(val logger: Logger) {
    def fatal(message: String) = {
      logger.error(message)
      throw new RuntimeException(s"Fatal error: $message")
    }
  }

  implicit class RicherIterable[T](val iterable: Iterable[T]) {
    def countBy[K](fn: T => K): Map[K, Int] = {
      val counts = mutable.Map[K, Int]()
      iterable.foreach { el =>
        val k = fn(el)
        counts(k) = counts.getOrElse(k, 0) + 1
      }
      // Converts to immutable map.
      counts.toMap
    }
  }

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

  // TODO(alexwilson): For each resampled vessel point, send it out keyed against each constituent
  // cell (a choice of level 13 seems reasonable) and the quantised time (in 10 minute buckets). Then
  // for each cell and time point, do an N^2 comparison between all vessel points, re-broadcast, group
  // by time and first mmsi and de-dupe. Then extract the count and the top 10 from each.

  // For a given radius of cap on the sphere, a given location and a given S2 cell level, return
  // all the S2 cells required to cover the cap.
  def getCapCoveringCells(location: LatLon,
                          radius: DoubleU[kilometer],
                          level: Int): Seq[S2CellId] = {
    val earthRadiusKm = S2LatLng.EARTH_RADIUS_METERS / 1000.0
    val capRadiusOnUnitSphere = radius.value / earthRadiusKm
    val coverer = new S2RegionCoverer()
    coverer.setMinLevel(level)
    coverer.setMaxLevel(level)

    // S2 cap requires an axis (location on unit sphere) and the height of the cap (the cap is
    // a planar cut on the unit sphere). The cap height is 1 - (sqrt(r^2 - a^2)/r) where r is
    // the radius of the circle (1.0 after we've normalized) and a is the radius of the cap itself.
    val axis = location.getS2LatLng().normalized().toPoint()
    val capHeight = 1.0 - (math.sqrt(1.0 - capRadiusOnUnitSphere * capRadiusOnUnitSphere))
    val cap = S2Cap.fromAxisHeight(axis, capHeight)

    val coverCells = new java.util.ArrayList[S2CellId]()
    coverer.getCovering(cap, coverCells)

    coverCells.foreach { cc =>
      assert(cc.level() == level)
    }

    coverCells.toList
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
                                    interpDistFromShore.of[kilometer]))
        }
      }

      iterTime += incrementSeconds
    }

    interpolatedSeries.toIndexedSeq
  }
}
