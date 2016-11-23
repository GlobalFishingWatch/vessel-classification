package org.skytruth.common

import io.github.karols.units._
import io.github.karols.units.SI._
import com.spotify.scio.testing.PipelineSpec
import com.spotify.scio.bigquery.TableRow
import com.spotify.scio.io._
import org.joda.time.{Duration, Instant}
import org.scalatest._

import AdditionalUnits._
import TestHelper._

class AISDataProcessingTests extends PipelineSpec with Matchers {
  "The pipeline" should "filter out messages without location" in {
    runWithContext { sc =>
      val input = sc.parallelize(
        Seq(
          TableRow("mmsi" -> "45", "foo" -> "bar")
        ))
      val locationRecords =
        AISDataProcessing.readJsonRecords(Seq(input), Set(), 0)

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
        AISDataProcessing.readJsonRecords(Seq(input), Set(), 0)

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
        Map(VesselMetadata(45) -> Seq(vlr("2016-01-01T00:00:00Z", lat = 45.3, lon = 0.5),
                                      vlr("2016-01-01T00:02:00Z", lat = 45.3, lon = 0.5)),
            VesselMetadata(127) -> Seq(vlr("2016-01-01T00:00:00Z", lat = 68.4, lon = 32.0),
                                       vlr("2016-01-01T00:01:30Z", lat = 68.4, lon = 32.0),
                                       vlr("2016-01-01T00:03:00Z", lat = 68.4, lon = 32.0)))

      val locationRecords =
        AISDataProcessing.readJsonRecords(Seq(input), Set(), 0)

      locationRecords should haveSize(2)
      locationRecords should equalMapOf(correctRecords)
    }
  }
}

class VesselSeriesTests extends PipelineSpec with Matchers {
  "The pipeline" should "successfully thin points down" in {
    val inputRecords = Seq(vlr("2011-07-01T00:00:00Z", lat = 10.0, lon = 10.0),
                           vlr("2011-07-01T00:02:00Z", lat = 10.0, lon = 10.0),
                           vlr("2011-07-01T00:03:00Z", lat = 10.0, lon = 10.0),
                           vlr("2011-07-01T00:05:00Z", lat = 10.0, lon = 10.0),
                           vlr("2011-07-01T00:07:00Z", lat = 10.0, lon = 10.0),
                           vlr("2011-07-01T00:15:00Z", lat = 10.0, lon = 10.0),
                           vlr("2011-07-01T00:19:00Z", lat = 10.0, lon = 10.0),
                           vlr("2011-07-01T00:30:00Z", lat = 10.0, lon = 10.0),
                           vlr("2011-07-01T00:40:00Z", lat = 10.0, lon = 10.0))

    val expected = Seq(vlr("2011-07-01T00:00:00Z", lat = 10.0, lon = 10.0),
                       vlr("2011-07-01T00:05:00Z", lat = 10.0, lon = 10.0),
                       vlr("2011-07-01T00:15:00Z", lat = 10.0, lon = 10.0),
                       vlr("2011-07-01T00:30:00Z", lat = 10.0, lon = 10.0),
                       vlr("2011-07-01T00:40:00Z", lat = 10.0, lon = 10.0))

    val result = AISDataProcessing.thinPoints(inputRecords)

    result should contain theSameElementsAs expected
  }

  "The pipeline" should "remove long stationary periods" in {
    val inputRecords =
      Seq(vlr("2011-07-01T00:00:00Z", lat = 10, lon = 10, distanceToShore = 400.0),
          vlr("2011-07-02T00:00:00Z", lat = 10, lon = 10, distanceToShore = 400.0),
          vlr("2011-07-03T00:00:00Z", lat = 10, lon = 10, distanceToShore = 400.0),
          vlr("2011-07-04T00:00:00Z", lat = 10, lon = 10, distanceToShore = 400.0),
          vlr("2011-07-05T00:00:00Z", lat = 11, lon = 10),
          vlr("2011-07-06T00:00:00Z", lat = 12, lon = 10),
          vlr("2011-07-07T00:00:00Z", lat = 12, lon = 10),
          vlr("2011-07-08T00:00:00Z", lat = 13, lon = 10),
          vlr("2011-07-09T00:00:00Z", lat = 14, lon = 10),
          vlr("2011-07-10T00:00:00Z", lat = 14, lon = 10.001),
          vlr("2011-07-12T00:00:00Z", lat = 14, lon = 10.002),
          vlr("2011-07-13T00:00:00Z", lat = 15, lon = 10),
          vlr("2011-07-14T00:00:00Z", lat = 16, lon = 10))

    val expectedLocations =
      Seq(vlr("2011-07-01T00:00:00Z", lat = 10, lon = 10, distanceToShore = 400.0),
          vlr("2011-07-04T00:00:00Z", lat = 10, lon = 10, distanceToShore = 400.0),
          vlr("2011-07-05T00:00:00Z", lat = 11, lon = 10),
          vlr("2011-07-06T00:00:00Z", lat = 12, lon = 10),
          vlr("2011-07-07T00:00:00Z", lat = 12, lon = 10),
          vlr("2011-07-08T00:00:00Z", lat = 13, lon = 10),
          vlr("2011-07-09T00:00:00Z", lat = 14, lon = 10),
          vlr("2011-07-12T00:00:00Z", lat = 14, lon = 10.002),
          vlr("2011-07-13T00:00:00Z", lat = 15, lon = 10),
          vlr("2011-07-14T00:00:00Z", lat = 16, lon = 10))

    val expectedStationaryPeriods =
      Seq(StationaryPeriod(LatLon(10.0.of[degrees], 10.0.of[degrees]),
                           Duration.standardHours(24 * 3),
                           400.0.of[kilometer],
                           0.0.of[kilometer]),
          StationaryPeriod(LatLon(14.0.of[degrees], 10.001.of[degrees]),
                           Duration.standardHours(24 * 3),
                           500.0.of[kilometer],
                           0.07188281512413591.of[kilometer]))

    val result = AISDataProcessing.removeStationaryPeriods(inputRecords)

    result.locations should contain theSameElementsAs expectedLocations
    result.stationaryPeriods should contain theSameElementsAs expectedStationaryPeriods
  }
}
