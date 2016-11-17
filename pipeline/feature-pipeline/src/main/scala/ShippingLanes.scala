package org.skytruth.feature_pipeline

import io.github.karols.units._
import io.github.karols.units.SI._
import io.github.karols.units.defining._

import com.google.common.geometry.S2CellId
import com.typesafe.scalalogging.LazyLogging
import com.spotify.scio.values.SCollection
import org.joda.time.{Duration, Instant}
import org.skytruth.common.AdditionalUnits._
import org.skytruth.common.Implicits._
import org.skytruth.common.LatLon

import scala.collection.{mutable, immutable}
import scala.math._

abstract class AbstractStdDev {
    def get = math.sqrt(getSqr)
    def +(other: AbstractStdDev) = StdDevSum(getSqr + other.getSqr)
    def getSqr : Double
}

case class StdDevSum(sqr : Double) extends AbstractStdDev {
    def getSqr = sqr
}
        
case class StdDev(
  count : Double = 0.0,
  sum : Double = 0.0,
  sqrsum : Double = 0.0
) extends AbstractStdDev {
  def +(value : Double) = StdDev(count + 1.0, sum + value, sqrsum + math.pow(value, 2.0))
  def -(value : Double) = {
    assert(count > 0.0, "Negative population size")
    StdDev(count - 1.0, sum - value, sqrsum - math.pow(value, 2.0))
  }
  def ::(other : StdDev) = StdDev(count + other.count, sum + other.sum, sqrsum + other.sqrsum)
  def getSqr = {
    if (count == 0.0) {
      0.0
    } else {
      val a = sqrsum/count
      val b = math.pow(sum/count, 2.0)
      if (a < b) {
        assert(b - a < 1e10-10, "sqrsum/count < math.pow(sum/count, 2.0) and not a rounding error")
        0.0
      } else {
        a - b
      }
    }
  }
}

case class ShippingInfo(coursex : StdDev = StdDev(), coursey : StdDev = StdDev()) {
  def +(other : ShippingInfo) : ShippingInfo = {
    ShippingInfo(coursex :: other.coursex, coursey :: other.coursey)
  }

  def score : Double = {
    (coursex + coursey).get
  }
}

object ShippingInfo {
  def fromLocation(location : VesselLocationRecord) : ShippingInfo = {
    val course = location.course.convert[radians].value
    ShippingInfo(StdDev() + math.cos(course), StdDev() + math.sin(course))
  }
}

object ShippingLanes extends LazyLogging {

  def calculateShippingLanes(input: SCollection[(VesselMetadata, Seq[VesselLocationRecord])]): SCollection[(S2CellId, Double)] = {
    input
      .flatMap{
        case (meta, locations) => {
          locations.foldLeft(immutable.HashMap[S2CellId, ShippingInfo]())((raster, location) => {
            val cell = location.location.getS2CellId(13)
            val value = ShippingInfo.fromLocation(location)
            if (raster.contains(cell)) {
              raster + (cell -> value)
            } else {
              raster + (cell -> (raster.get(cell).get + value))
            }
          }).toIterable
        }
      }
      .foldByKey(ShippingInfo())((info1, info2) => info1 + info2)
      .map{ case (cell, info) => (cell, info.score) }
  }
}
