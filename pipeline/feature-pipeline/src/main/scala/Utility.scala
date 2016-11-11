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
import org.json4s._
import org.json4s.JsonDSL.WithDouble._
import org.json4s.native.JsonMethods._
import org.skytruth.common.{Implicits => STImplicits}
import org.skytruth.dataflow.{TFRecordSink, TFRecordUtils}

import scala.collection.{mutable, immutable}
import scala.collection.JavaConversions._

import com.spotify.scio._
import resource._

object ScioContextResource {
  implicit def scioContextResource[A <: ScioContext] = new Resource[A] {
    override def close(r: A) = r.close()
  }
}

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

  def weightedMean(locations: Iterable[LatLon], weights: Iterable[Double]): LatLon = {
    var lat = 0.0
    var lon = 0.0
    var weight = 0.0
    (locations zip weights).foreach {
      case (l, w) =>
        lat += l.lat.value * w
        lon += l.lon.value * w
        weight += w
    }
    LatLon((lat / weight).of[degrees], (lon / weight).of[degrees])
  }
}

case class VesselMetadata(mmsi: Int) {
  def flagState = CountryCodes.fromMmsi(mmsi)
}

case class StationaryPeriod(location: LatLon, duration: Duration)

case class ProcessedLocations(locations: Seq[VesselLocationRecord],
                              stationaryPeriods: Seq[StationaryPeriod])

case class VesselLocationRecord(timestamp: Instant,
                                location: LatLon,
                                distanceToShore: DoubleU[kilometer],
                                speed: DoubleU[knots],
                                course: DoubleU[degrees],
                                heading: DoubleU[degrees])

case class ResampledVesselLocation(timestamp: Instant,
                                   location: LatLon,
                                   distanceToShore: DoubleU[kilometer],
                                   pointDensity: Double)

case class ResampledVesselLocationWithAdjacency(
    locationRecord: ResampledVesselLocation,
    numNeighbours: Int,
    closestNeighbour: Option[(VesselMetadata, DoubleU[kilometer], ResampledVesselLocation)])

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

case class AnchoragePoint(meanLocation: LatLon,
                          vessels: Seq[VesselMetadata],
                          knownFishingVesselCount: Int) {
  import STImplicits._

  def toJson = {
    val flagStateDistribution = vessels.countBy(_.flagState).toSeq.sortBy(c => -c._2)
    ("id" -> id) ~
      ("latitude" -> meanLocation.lat.value) ~
      ("longitude" -> meanLocation.lon.value) ~
      ("unique_vessel_count" -> vessels.size) ~
      ("known_fishing_vessel_count" -> knownFishingVesselCount) ~
      ("flag_state_distribution" -> flagStateDistribution) ~
      ("mmsis" -> vessels.map(_.mmsi))
  }

  def id: String =
    meanLocation.getS2CellId(Parameters.anchoragesS2Scale).toToken
}

case class AnchorageGroup(meanLocation: LatLon, anchoragePoints: Set[AnchoragePoint]) {
  import STImplicits._

  def id: String =
    meanLocation.getS2CellId(Parameters.anchoragesS2Scale).toToken

  def toJson = {
    ("id" -> id) ~
      ("latitude" -> meanLocation.lat.value) ~
      ("longitude" -> meanLocation.lon.value) ~
      ("anchorage_points" -> anchoragePoints.toSeq.sortBy(_.id).map(_.id))
  }
}

object AnchorageGroup {
  def fromAnchorages(anchoragePoints: Iterable[AnchoragePoint]) =
    AnchorageGroup(LatLon.weightedMean(anchoragePoints.map(_.meanLocation),
                                       anchoragePoints.map(_.vessels.length.toDouble)),
                   anchoragePoints.toSet)
}

case class AnchorageGroupVisit(anchorageGroup: AnchorageGroup,
                               arrival: Instant,
                               departure: Instant) {
  def extend(other: AnchorageGroupVisit): immutable.Seq[AnchorageGroupVisit] = {
    if (anchorageGroup eq other.anchorageGroup) {
      Vector(AnchorageGroupVisit(anchorageGroup, arrival, other.departure))
    } else {
      Vector(this, other)
    }
  }

  def duration = new Duration(arrival, departure)

  def toJson =
    ("anchorageGroup" -> anchorageGroup.id) ~
      ("arrival" -> arrival.toString()) ~
      ("departure" -> departure.toString())
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

case class AdjacencyLookup[T](values: Seq[T],
                              locFn: T => LatLon,
                              maxRadius: DoubleU[kilometer],
                              level: Int) {
  private val cellMap = values
    .flatMap(v =>
      Utility.getCapCoveringCells(locFn(v), maxRadius, level).map(cellid => (cellid, v)))
    .groupBy(_._1)
    .map { case (cellid, vs) => (cellid, vs.map(_._2)) }

  def nearby(location: LatLon) = {
    val queryCells = Utility.getCapCoveringCells(location, maxRadius, level)
    val allValues = queryCells.flatMap { cellid =>
      cellMap.getOrElse(cellid, Seq())
    }

    allValues.map(v => (locFn(v).getDistance(location), v)).toIndexedSeq.distinct.sortBy(_._1)
  }
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
