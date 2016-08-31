package org.skytruth.feature_pipeline

import io.github.karols.units._
import io.github.karols.units.SI._
import io.github.karols.units.defining._
import com.spotify.scio._
import com.spotify.scio.bigquery._
import com.typesafe.scalalogging.LazyLogging
import org.joda.time.Instant

object AdditionalUnits {
  type knots = DefineUnit[_k ~: _n ~: _o ~: _t]
  type degrees = DefineUnit[_d ~: _e ~: _g]
}

import AdditionalUnits._

case class LatLon(val lat: Double, val lon: Double)

case class VesselLocationRecord(
    val mmsi: Int,
    val timestamp: Instant,
    val location: LatLon,
    val distanceToShore: DoubleU[kilometer],
    val distanceToPort: DoubleU[kilometer],
    val speed: DoubleU[knots],
    val course: DoubleU[degrees],
    val heading: DoubleU[degrees]
)

object Pipeline extends LazyLogging {
  def main(argArray: Array[String]) {
    val (sc, args) = ContextAndArgs(argArray)

    val inputData = args("input_data")

    val locationRecords = sc.tableRowJsonFile(inputData)
      // Keep only records with a location.
      .filter((json) => json.containsKey("lat") && json.containsKey("lon"))
      // Build a typed location record with units of measure.
      .map((json) => {
        val record =
          VesselLocationRecord(json.getLong("mmsi").toInt,
                               Instant.parse(json.getString("timestamp")),
                               LatLon(json.getDouble("lat"), json.getDouble("lon")),
                               json.getDouble("distance_to_shore").of[kilometer],
                               json.getDouble("distance_to_port").of[kilometer],
                               json.getDouble("speed").of[knots],
                               json.getDouble("course").of[degrees],
                               json.getDouble("heading").of[degrees])
      })

    sc.close()
  }
}
