package org.skytruth.feature_pipeline

import io.github.karols.units._
import io.github.karols.units.SI._
import io.github.karols.units.defining._
import com.spotify.scio._
import com.spotify.scio.values.SCollection
import com.spotify.scio.bigquery._
import com.typesafe.scalalogging.LazyLogging
import org.joda.time.Instant

object AdditionalUnits {
  type knots = DefineUnit[_k ~: _n ~: _o ~: _t]
  type degrees = DefineUnit[_d ~: _e ~: _g]
}

import AdditionalUnits._

case class LatLon(val lat: DoubleU[degrees], val lon: DoubleU[degrees])

case class VesselLocationRecord(
    val timestamp: Instant,
    val location: LatLon,
    val distanceToShore: DoubleU[kilometer],
    val distanceToPort: DoubleU[kilometer],
    val speed: DoubleU[knots],
    val course: DoubleU[degrees],
    val heading: DoubleU[degrees]
)

object Pipeline extends LazyLogging {
  lazy val blacklistedMmsis = Set(0, 12345)

  // Reads JSON vessel records, filters to only location records, groups by MMSI and sorts
  // by ascending timestamp.
  def readJsonRecords(
      input: SCollection[TableRow]): SCollection[(Int, Seq[VesselLocationRecord])] =
    // Keep only records with a location.
    input
      .filter((json) => json.containsKey("lat") && json.containsKey("lon"))
      // Build a typed location record with units of measure.
      .map((json) => {
        val mmsi = json.getLong("mmsi").toInt
        val record =
          VesselLocationRecord(Instant.parse(json.getString("timestamp")),
                               LatLon(json.getDouble("lat").of[degrees],
                                      json.getDouble("lon").of[degrees]),
                               json.getDouble("distance_to_shore").of[kilometer],
                               json.getDouble("distance_to_port").of[kilometer],
                               json.getDouble("speed").of[knots],
                               json.getDouble("course").of[degrees],
                               json.getDouble("heading").of[degrees])
        (mmsi, record)
      })
      .filter { case (mmsi, records) => !blacklistedMmsis.contains(mmsi) }
      .groupByKey
      .map { case (mmsi, records) => (mmsi, records.toSeq.sortBy(_.timestamp.getMillis)) }

  def main(argArray: Array[String]) {
    val (sc, args) = ContextAndArgs(argArray)

    val inputData = args("input_data")

    // Read, filter and build location records.
    val locationRecords = readJsonRecords(sc.tableRowJsonFile(inputData))

    sc.close()
  }
}
