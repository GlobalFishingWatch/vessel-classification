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
import org.joda.time.{DateTime, DateTimeZone}
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

object Implicits {
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

    // TODO(alexwilson): this is not a true median atm, because for an even
    // number of elements it does not average the two central values but picks
    // the lower arbitrarily. This is just to avoid having to have addition
    // and division also defined for T.
    def medianBy[V <% Ordered[V]](fn: T => V): T = {
      val asIndexedSeq = iterable.toIndexedSeq.sortBy(fn)
      asIndexedSeq.apply(asIndexedSeq.size / 2)
    }
  }

  implicit class RicherDoubleUIterable[T <: MUnit](val iterable: Iterable[DoubleU[T]]) {
    def mean: DoubleU[T] = {
      var acc = 0.0
      var count = 0
      iterable.foreach { v =>
        acc += v.value
        count += 1
      }
      (acc / count.toDouble).of[T]
    }

    def weightedMean(weights: Iterable[Double]): DoubleU[T] = {
      var acc = 0.0
      var weight = 0.0
      iterable.zip(weights).foreach {
        case (v, w) =>
          acc += v.value * w
          weight += w
      }
      (acc / weight).of[T]
    }
  }
}

// TODO(alexwilson): This config is too hard-coded to our current setup. Move
// out to config files for greater flexibility. Note there is an equivalent to
// this in gcp_config.py which should remain in-sync.
object GcpConfig extends LazyLogging {
  import Implicits._

  private def projectId = "world-fishing-827"

  // TODO(alexwilson): No locally-generated date for prod. Needs to be sourced
  // from outside so all prod stages share the same path.
  def makeConfig(environment: String, jobId: String) = {
    val now = new DateTime(DateTimeZone.UTC)
    val rootPath = environment match {
      case "prod" => {
        s"gs://world-fishing-827/data-production/classification/$jobId"
      }
      case "dev" => {
        sys.env.get("USER") match {
          case Some(user) =>
            s"gs://world-fishing-827-dev-ttl30d/data-production/classification/$user/$jobId"
          case _ => logger.fatal("USER environment variable cannot be empty for dev runs.")
        }
      }
      case _ => logger.fatal(s"Invalid environment: $environment.")
    }

    GcpConfig(now, projectId, rootPath)
  }
}

case class GcpConfig(startTime: DateTime, projectId: String, private val rootPath: String) {
  def dataflowStagingPath = s"$rootPath/pipeline/staging"
  def pipelineOutputPath = s"$rootPath/pipeline/output"
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

object ScioContextResource {
  implicit def scioContextResource[A <: ScioContext] = new Resource[A] {
    override def close(r: A) = r.close()
  }
}
