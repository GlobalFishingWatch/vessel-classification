package org.skytruth.feature_pipeline

import com.google.common.geometry.{S2CellId}
import com.spotify.scio.testing.PipelineSpec

import io.github.karols.units._
import io.github.karols.units.SI._

import org.joda.time.{Duration, Instant}
import org.scalatest._

import scala.collection.{mutable, immutable}

class AnchorageVisitsTests extends PipelineSpec with Matchers {
  import TestHelper._
  import AdditionalUnits._

  val anchoragePoint1 =
    AnchoragePoint(LatLon(0.0.of[degrees], 0.0.of[degrees]),
                   Seq(VesselMetadata(1)),
                   0,
                   0.0.of[kilometer])
  val anchoragePoint2 =
    AnchoragePoint(LatLon(0.0.of[degrees], 1.0.of[degrees]),
                   Seq(VesselMetadata(1)),
                   0,
                   0.0.of[kilometer])

  val anchorages = Seq(
    Anchorage.fromAnchoragePoints(Seq(anchoragePoint1)),
    Anchorage.fromAnchoragePoints(Seq(anchoragePoint2))
  )

  val vesselPath = Seq(
    vlr("2016-01-01T00:00:00Z", 0.008, 0.0, speed = 1.0),
    vlr("2016-01-01T00:01:00Z", 0.006, 0.0, speed = 1.0),
    vlr("2016-01-01T00:02:00Z", 0.004, 0.0, speed = 1.0),
    vlr("2016-01-01T00:03:00Z", 0.002, 0.0, speed = 1.0),
    vlr("2016-01-01T00:04:00Z", 0.000, 0.0, speed = 1.0),
    vlr("2016-01-01T00:05:00Z", -0.002, 0.0, speed = 1.0),
    vlr("2016-01-01T00:06:00Z", -0.004, 0.0, speed = 1.0),
    vlr("2016-01-01T00:07:00Z", -0.006, 0.0, speed = 1.0),
    vlr("2016-01-01T00:08:00Z", -0.008, 0.0, speed = 1.0),
    vlr("2016-01-01T01:00:00Z", 0.008, 0.5, speed = 1.0),
    vlr("2016-01-01T01:01:00Z", 0.006, 0.5, speed = 1.0),
    vlr("2016-01-01T01:02:00Z", 0.004, 0.5, speed = 1.0),
    vlr("2016-01-01T01:03:00Z", 0.002, 0.5, speed = 1.0),
    vlr("2016-01-01T01:04:00Z", 0.000, 0.5, speed = 1.0),
    vlr("2016-01-01T01:05:00Z", -0.002, 0.5, speed = 1.0),
    vlr("2016-01-01T01:06:00Z", -0.004, 0.5, speed = 1.0),
    vlr("2016-01-01T01:07:00Z", -0.006, 0.5, speed = 1.0),
    vlr("2016-01-01T01:08:00Z", -0.008, 0.5, speed = 1.0),
    vlr("2016-01-01T02:00:00Z", 0.008, 1.0, speed = 1.0),
    vlr("2016-01-01T02:01:00Z", 0.006, 1.0, speed = 1.0),
    vlr("2016-01-01T02:02:00Z", 0.004, 1.0, speed = 1.0),
    vlr("2016-01-01T02:03:00Z", 0.002, 1.0, speed = 1.0),
    vlr("2016-01-01T02:04:00Z", 0.000, 1.0, speed = 1.0),
    vlr("2016-01-01T02:05:00Z", -0.002, 1.0, speed = 1.0),
    vlr("2016-01-01T02:06:00Z", -0.004, 1.0, speed = 1.0),
    vlr("2016-01-01T02:07:00Z", -0.006, 1.0, speed = 1.0),
    vlr("2016-01-01T02:08:00Z", -0.008, 1.0, speed = 1.0)
  )

  val expected =
    (VesselMetadata(45),
     immutable.Seq(AnchorageVisit(Anchorage(LatLon(0.0.of[degrees], 0.0.of[degrees]),
                                            Set(anchoragePoint1),
                                            0.0.of[kilometer]),
                                  Instant.parse("2016-01-01T00:00:00.000Z"),
                                  Instant.parse("2016-01-01T00:08:00.000Z")),
                   AnchorageVisit(Anchorage(LatLon(0.0.of[degrees], 1.0.of[degrees]),
                                            Set(anchoragePoint2),
                                            0.0.of[kilometer]),
                                  Instant.parse("2016-01-01T02:00:00.000Z"),
                                  Instant.parse("2016-01-01T02:08:00.000Z"))))

  "Vessel" should "visit the correct anchorages" in {
    runWithContext { sc =>
      val vesselRecords = sc.parallelize(Seq((VesselMetadata(45), vesselPath)))
      val res = Anchorages.findAnchorageVisits(
        vesselRecords,
        sc.parallelize(anchorages),
        Duration.standardMinutes(5)
      )

      res should containSingleValue(expected)
    }
  }
}

class AnchoragesGroupingTests extends PipelineSpec with Matchers {
  import TestHelper._
  import AdditionalUnits._

  def anchoragePointFromS2CellToken(token: String, vessels: Seq[VesselMetadata]) =
    AnchoragePoint(LatLon.fromS2CellId(S2CellId.fromToken(token)), vessels, 0, 0.0.of[kilometer])

  "Anchorage merging" should "work!" in {
    val anchorages = IndexedSeq(
      anchoragePointFromS2CellToken("89c19c9c",
                                    Seq(VesselMetadata(1), VesselMetadata(2), VesselMetadata(3))),
      anchoragePointFromS2CellToken("89c19b64", Seq(VesselMetadata(1), VesselMetadata(2))),
      anchoragePointFromS2CellToken("89c1852c", Seq(VesselMetadata(1))),
      anchoragePointFromS2CellToken("89c19b04", Seq(VesselMetadata(1), VesselMetadata(2))),
      anchoragePointFromS2CellToken("89c19bac", Seq(VesselMetadata(1), VesselMetadata(2))),
      anchoragePointFromS2CellToken("89c19bb4", Seq(VesselMetadata(1), VesselMetadata(2))))

    val groupedAnchorages = Anchorages.mergeAdjacentAnchoragePoints(anchorages)

    groupedAnchorages should have size 3

    val expected =
      Seq(Anchorage(LatLon(40.016824742437635.of[degrees], -74.07113588841028.of[degrees]),
                    Set(anchorages(2)),
                    0.0.of[kilometer]),
          Anchorage(LatLon(39.994377589412146.of[degrees], -74.12517039688245.of[degrees]),
                    Set(anchorages(0), anchorages(1)),
                    0.0.of[kilometer]),
          Anchorage(LatLon(39.96842156104703.of[degrees], -74.0828838592642.of[degrees]),
                    Set(anchorages(3), anchorages(4), anchorages(5)),
                    0.0.of[kilometer]))

    groupedAnchorages should contain theSameElementsAs expected
  }
}
