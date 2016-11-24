package org.skytruth.common

import io.github.karols.units._
import io.github.karols.units.SI._
import org.joda.time.{Duration, Instant}
import org.json4s._
import org.json4s.JsonDSL.WithDouble._
import org.json4s.native.JsonMethods._
import AdditionalUnits._

object TestHelper {
  def buildMessage(mmsi: Int,
                   timestamp: String,
                   lat: Double = 0.0,
                   lon: Double = 0.0,
                   distanceFromShore: Double = 500.0,
                   distanceFromPort: Double = 500.0,
                   speed: Double = 0.0,
                   course: Double = 0.0,
                   heading: Double = 0.0) =
    compact(
      render(
        ("mmsi" -> mmsi) ~
          ("timestamp" -> timestamp) ~
          ("lon" -> lon) ~
          ("lat" -> lat) ~
          ("distance_from_shore" -> (distanceFromShore * 1000.0)) ~
          ("distance_from_port" -> (distanceFromPort * 1000.0)) ~
          ("speed" -> speed) ~
          ("course" -> course) ~
          ("heading" -> heading)))

  def ts(timestamp: String) = Instant.parse(timestamp)

  def vlr(timestamp: String = "1970-01-01T00:00:00Z",
          lat: Double = 0.0,
          lon: Double = 0.0,
          distanceToShore: Double = 500.0,
          speed: Double = 0.0,
          course: Double = 0.0,
          heading: Double = 0.0) =
    VesselLocationRecord(ts(timestamp),
                         LatLon(lat.of[degrees], lon.of[degrees]),
                         distanceToShore.of[kilometer],
                         speed.of[knots],
                         course.of[degrees],
                         heading.of[degrees])
}
