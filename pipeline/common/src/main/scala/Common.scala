package org.skytruth.common

import io.github.karols.units._
import io.github.karols.units.SI._
import io.github.karols.units.defining._

import com.google.common.geometry.{S2, S2Cap, S2CellId, S2LatLng, S2RegionCoverer}
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import com.spotify.scio._
import com.typesafe.scalalogging.{LazyLogging, Logger}

import org.apache.commons.math3.util.MathUtils
import org.json4s._
import org.json4s.JsonDSL.WithDouble._
import org.json4s.native.JsonMethods._
import org.joda.time.{DateTime, DateTimeZone, Duration, Instant}
import org.joda.time.format.ISODateTimeFormat

import scala.collection.{mutable, immutable}
import scala.collection.JavaConversions._

import resource._

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

  // For a given radius of cap on the sphere, a given location and a given S2 cell level, return
  // all the S2 cells required to cover the cap.
  def getCapCoveringCells(radius: DoubleU[kilometer], level: Int): Seq[S2CellId] = {
    val earthRadiusKm = S2LatLng.EARTH_RADIUS_METERS / 1000.0
    val capRadiusOnUnitSphere = radius.value / earthRadiusKm
    val coverer = new S2RegionCoverer()
    coverer.setMinLevel(level)
    coverer.setMaxLevel(level)

    // S2 cap requires an axis (location on unit sphere) and the height of the cap (the cap is
    // a planar cut on the unit sphere). The cap height is 1 - (sqrt(r^2 - a^2)/r) where r is
    // the radius of the circle (1.0 after we've normalized) and a is the radius of the cap itself.
    val axis = getS2LatLng().normalized().toPoint()
    val capHeight = 1.0 - (math.sqrt(1.0 - capRadiusOnUnitSphere * capRadiusOnUnitSphere))
    val cap = S2Cap.fromAxisHeight(axis, capHeight)

    val coverCells = new java.util.ArrayList[S2CellId]()
    coverer.getCovering(cap, coverCells)

    coverCells.foreach { cc =>
      assert(cc.level() == level)
    }

    coverCells.toList
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

case class VesselMetadata(mmsi: Int, isFishingVessel: Boolean = false) {
  def flagState = CountryCodes.fromMmsi(mmsi)

  def toJson = {
    ("mmsi" -> mmsi) ~
    ("is_fishing" -> isFishingVessel)
  }
}

object VesselMetadata {
  implicit val formats = DefaultFormats

  def fromJson(json: JValue) = VesselMetadata(
    (json \ "mmsi").extract[Int],
    (json \ "is_fishing").extract[Boolean])
}

case class VesselLocationRecord(timestamp: Instant,
                                location: LatLon,
                                distanceToShore: DoubleU[kilometer],
                                speed: DoubleU[knots],
                                course: DoubleU[degrees],
                                heading: DoubleU[degrees])

case class StationaryPeriod(location: LatLon,
                            duration: Duration,
                            meanDistanceToShore: DoubleU[kilometer],
                            meanDriftRadius: DoubleU[kilometer])

case class ProcessedLocations(locations: Seq[VesselLocationRecord],
                              stationaryPeriods: Seq[StationaryPeriod])

case class AdjacencyLookup[T](values: Seq[T],
                              locFn: T => LatLon,
                              maxRadius: DoubleU[kilometer],
                              level: Int) {
  private val cellMap = values
    .flatMap(v => locFn(v).getCapCoveringCells(maxRadius, level).map(cellid => (cellid, v)))
    .groupBy(_._1)
    .map { case (cellid, vs) => (cellid, vs.map(_._2)) }

  def nearby(location: LatLon) = {
    val queryCells = location.getCapCoveringCells(maxRadius, level)
    val allValues = queryCells.flatMap { cellid =>
      cellMap.getOrElse(cellid, Seq())
    }

    allValues.map(v => (locFn(v).getDistance(location), v)).toIndexedSeq.distinct.sortBy(_._1)
  }
}

case class IteratorWithCurrent[T](private val iterator: Iterator[T]) {
  private def nextOption(): Option[T] =
    if (iterator.hasNext) {
      Some(iterator.next())
    } else {
      None
    }

  var current: Option[T] = nextOption()

  def getNext() { current = nextOption() }
}

case class ValueCache[T]() {
  private var value: Option[T] = None

  def get(getter: () => T) = {
    if (!value.isDefined) {
      value = Some(getter())
    }

    value.get
  }
}
