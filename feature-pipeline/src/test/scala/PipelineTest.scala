package org.skytruth.feature_pipeline

import com.spotify.scio.bigquery.TableRow
import com.spotify.scio.testing.PipelineSpec
import io.github.karols.units._
import io.github.karols.units.SI._
import java.io.File
import org.joda.time.{Duration, Instant}
import org.scalatest._

object TestHelper {
  import AdditionalUnits._

  def buildMessage(mmsi: Int,
                   timestamp: String,
                   lat: Double = 0.0,
                   lon: Double = 0.0,
                   distanceFromShore: Double = 500.0,
                   distanceFromPort: Double = 500.0,
                   speed: Double = 0.0,
                   course: Double = 0.0,
                   heading: Double = 0.0) =
    TableRow("mmsi" -> mmsi.toString,
             "timestamp" -> timestamp,
             "lon" -> lon.toString,
             "lat" -> lat.toString,
             "distance_from_shore" -> (distanceFromShore * 1000.0).toString,
             "distance_from_port" -> (distanceFromPort * 1000.0).toString,
             "speed" -> speed.toString,
             "course" -> course.toString,
             "heading" -> heading.toString)

  def ts(timestamp: String) = Instant.parse(timestamp)

  def buildLocationRecord(timestamp: String,
                          lat: Double,
                          lon: Double,
                          distanceToShore: Double = 500.0,
                          distanceToPort: Double = 500.0,
                          speed: Double = 0.0,
                          course: Double = 0.0,
                          heading: Double = 0.0) =
    VesselLocationRecord(ts(timestamp),
                         LatLon(lat.of[degrees], lon.of[degrees]),
                         distanceToShore.of[kilometer],
                         distanceToPort.of[kilometer],
                         speed.of[knots],
                         course.of[degrees],
                         heading.of[degrees])

}

class PipelineTests extends PipelineSpec with Matchers {
  import AdditionalUnits._
  import TestHelper._

  "The pipeline" should "filter out messages without location" in {
    runWithContext { sc =>
      val input = sc.parallelize(
        Seq(
          TableRow("mmsi" -> "45", "foo" -> "bar")
        ))
      val locationRecords =
        Pipeline.readJsonRecords(Seq(input))

      locationRecords should beEmpty
    }
  }

  "The pipeline" should "filter out valid messages from blacklisted MMSIs" in {
    runWithContext { sc =>
      val input = sc.parallelize(
        Seq(
          buildMessage(mmsi = 0, timestamp = "2016-01-01T00:00:00Z", lon = 45.3, lat = 0.5),
          buildMessage(mmsi = 12345, timestamp = "2016-01-01T00:00:00Z", lon = 21.3, lat = 4.6)
        ))
      val locationRecords =
        Pipeline.readJsonRecords(Seq(input))

      locationRecords should beEmpty
    }
  }

  "The pipeline" should "accept valid messages, and group them by mmsi, sorted by timestamp" in {
    runWithContext { sc =>
      val input = sc.parallelize(
        Seq(buildMessage(mmsi = 45, timestamp = "2016-01-01T00:02:00Z", lat = 45.3, lon = 0.5),
            buildMessage(mmsi = 127, timestamp = "2016-01-01T00:01:30Z", lat = 68.4, lon = 32.0),
            buildMessage(mmsi = 45, timestamp = "2016-01-01T00:00:00Z", lat = 45.3, lon = 0.5),
            buildMessage(mmsi = 127, timestamp = "2016-01-01T00:00:00Z", lat = 68.4, lon = 32.0),
            buildMessage(mmsi = 127, timestamp = "2016-01-01T00:03:00Z", lat = 68.4, lon = 32.0)))

      val correctRecords =
        Map(VesselMetadata(45) -> Seq(
              buildLocationRecord("2016-01-01T00:00:00Z", lat = 45.3, lon = 0.5),
              buildLocationRecord("2016-01-01T00:02:00Z", lat = 45.3, lon = 0.5)),
            VesselMetadata(127) -> Seq(
              buildLocationRecord("2016-01-01T00:00:00Z", lat = 68.4, lon = 32.0),
              buildLocationRecord("2016-01-01T00:01:30Z", lat = 68.4, lon = 32.0),
              buildLocationRecord("2016-01-01T00:03:00Z", lat = 68.4, lon = 32.0)))

      val locationRecords =
        Pipeline.readJsonRecords(Seq(input))

      locationRecords should haveSize(2)
      locationRecords should equalMapOf(correctRecords)
    }
  }
}

class VesselSeriesTests extends PipelineSpec with Matchers {
  import TestHelper._
  import AdditionalUnits._

  "The pipeline" should "successfully thin points down" in {
    val inputRecords = Seq(buildLocationRecord("2011-07-01T00:00:00Z", lat = 10.0, lon = 10.0),
                           buildLocationRecord("2011-07-01T00:02:00Z", lat = 10.0, lon = 10.0),
                           buildLocationRecord("2011-07-01T00:03:00Z", lat = 10.0, lon = 10.0),
                           buildLocationRecord("2011-07-01T00:05:00Z", lat = 10.0, lon = 10.0),
                           buildLocationRecord("2011-07-01T00:07:00Z", lat = 10.0, lon = 10.0),
                           buildLocationRecord("2011-07-01T00:15:00Z", lat = 10.0, lon = 10.0),
                           buildLocationRecord("2011-07-01T00:19:00Z", lat = 10.0, lon = 10.0),
                           buildLocationRecord("2011-07-01T00:30:00Z", lat = 10.0, lon = 10.0),
                           buildLocationRecord("2011-07-01T00:40:00Z", lat = 10.0, lon = 10.0))

    val expected = Seq(buildLocationRecord("2011-07-01T00:00:00Z", lat = 10.0, lon = 10.0),
                       buildLocationRecord("2011-07-01T00:05:00Z", lat = 10.0, lon = 10.0),
                       buildLocationRecord("2011-07-01T00:15:00Z", lat = 10.0, lon = 10.0),
                       buildLocationRecord("2011-07-01T00:30:00Z", lat = 10.0, lon = 10.0),
                       buildLocationRecord("2011-07-01T00:40:00Z", lat = 10.0, lon = 10.0))

    val result = Pipeline.thinPoints(inputRecords)

    result should contain theSameElementsAs expected
  }

  "The pipeline" should "remove long stationary periods" in {
    val inputRecords = Seq(buildLocationRecord("2011-07-01T00:00:00Z", lat = 10, lon = 10),
                           buildLocationRecord("2011-07-02T00:00:00Z", lat = 10, lon = 10),
                           buildLocationRecord("2011-07-03T00:00:00Z", lat = 10, lon = 10),
                           buildLocationRecord("2011-07-04T00:00:00Z", lat = 10, lon = 10),
                           buildLocationRecord("2011-07-05T00:00:00Z", lat = 11, lon = 10),
                           buildLocationRecord("2011-07-06T00:00:00Z", lat = 12, lon = 10),
                           buildLocationRecord("2011-07-07T00:00:00Z", lat = 12, lon = 10),
                           buildLocationRecord("2011-07-08T00:00:00Z", lat = 13, lon = 10),
                           buildLocationRecord("2011-07-09T00:00:00Z", lat = 14, lon = 10),
                           buildLocationRecord("2011-07-10T00:00:00Z", lat = 14, lon = 10),
                           buildLocationRecord("2011-07-12T00:00:00Z", lat = 14, lon = 10),
                           buildLocationRecord("2011-07-13T00:00:00Z", lat = 15, lon = 10),
                           buildLocationRecord("2011-07-14T00:00:00Z", lat = 16, lon = 10))

    val expectedLocations = Seq(buildLocationRecord("2011-07-01T00:00:00Z", lat = 10, lon = 10),
                                buildLocationRecord("2011-07-04T00:00:00Z", lat = 10, lon = 10),
                                buildLocationRecord("2011-07-05T00:00:00Z", lat = 11, lon = 10),
                                buildLocationRecord("2011-07-06T00:00:00Z", lat = 12, lon = 10),
                                buildLocationRecord("2011-07-07T00:00:00Z", lat = 12, lon = 10),
                                buildLocationRecord("2011-07-08T00:00:00Z", lat = 13, lon = 10),
                                buildLocationRecord("2011-07-09T00:00:00Z", lat = 14, lon = 10),
                                buildLocationRecord("2011-07-12T00:00:00Z", lat = 14, lon = 10),
                                buildLocationRecord("2011-07-13T00:00:00Z", lat = 15, lon = 10),
                                buildLocationRecord("2011-07-14T00:00:00Z", lat = 16, lon = 10))

    val expectedStationaryPeriods = Seq(
      StationaryPeriod(LatLon(10.0.of[degrees], 10.0.of[degrees]), Duration.standardHours(24 * 3)),
      StationaryPeriod(LatLon(14.0.of[degrees], 10.0.of[degrees]), Duration.standardHours(24 * 3)))

    val result = Pipeline.removeStationaryPeriods(inputRecords)

    result.locations should contain theSameElementsAs expectedLocations
    result.stationaryPeriods should contain theSameElementsAs expectedStationaryPeriods
  }
}

class LocationResamplerTests extends PipelineSpec with Matchers {
  import TestHelper._
  import AdditionalUnits._

  def rvl(timestamp: String, lat: Double, lon: Double) =
    ResampledVesselLocation(ts(timestamp),
                            LatLon(lat.of[degrees], lon.of[degrees]),
                            500.0.of[kilometer])

  "The resampler" should "resample points, but not if they are too far apart" in {
    val inputRecords = Seq(buildLocationRecord("2011-06-30T23:58:00Z", lat = 10.0, lon = 10.0),
                           // Pick up the exact value at 00:00:00
                           buildLocationRecord("2011-07-01T00:00:00Z", lat = 10.3, lon = 10.0),
                           buildLocationRecord("2011-07-01T00:02:00Z", lat = 10.0, lon = 10.0),
                           buildLocationRecord("2011-07-01T00:04:00Z", lat = 10.0, lon = 10.0),
                           buildLocationRecord("2011-07-01T00:06:00Z", lat = 10.0, lon = 10.0),
                           buildLocationRecord("2011-07-01T00:08:00Z", lat = 10.0, lon = 10.0),
                           // Interpolate time into 00:10:00, but no movement.
                           buildLocationRecord("2011-07-01T00:12:00Z", lat = 10.0, lon = 10.0),
                           buildLocationRecord("2011-07-01T00:18:00Z", lat = 10.0, lon = 10.0),
                           // Interpolate into 00:20:00, closer to the 00:18:00 point.
                           buildLocationRecord("2011-07-01T00:26:00Z", lat = 10.0, lon = 11.0),
                           // Do not generate samples where the surrounding points are more than
                           // an hour apart.
                           buildLocationRecord("2011-07-01T01:38:00Z", lat = 10.0, lon = 11.0),
                           // Interpolate into 01:40:00.
                           buildLocationRecord("2011-07-01T01:42:00Z", lat = 11.0, lon = 11.0))

    val expected =
      Seq(rvl("2011-07-01T00:00:00Z", 10.3, 10.0),
          rvl("2011-07-01T00:10:00Z", 10.0, 10.0),
          rvl("2011-07-01T00:20:00Z", 10.0, 10.25),
          rvl("2011-07-01T01:40:00Z", 10.5, 11.0))

    val result = Utility.resampleVesselSeries(Duration.standardMinutes(10), inputRecords)

    result should contain theSameElementsAs expected
  }
}
