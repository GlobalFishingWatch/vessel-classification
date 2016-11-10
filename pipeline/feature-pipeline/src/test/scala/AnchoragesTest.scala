package org.skytruth.feature_pipeline

import com.google.common.geometry.{S2CellId}
import com.spotify.scio.testing.PipelineSpec

import io.github.karols.units._
import io.github.karols.units.SI._

import org.joda.time.{Instant}
import org.scalatest._

import scala.collection.{mutable, immutable}

class AnchorageVisitsTests extends PipelineSpec with Matchers {
  import TestHelper._
  import AdditionalUnits._

  val anchorages1 = Anchorage(LatLon(0.0.of[degrees], 0.0.of[degrees]), Seq(VesselMetadata(1)), 0)
  val anchorages2 = Anchorage(LatLon(0.0.of[degrees], 1.0.of[degrees]), Seq(VesselMetadata(1)), 0)

  val anchorageGroups = Seq(
    AnchorageGroup.fromAnchorages(Seq(anchorages1)),
    AnchorageGroup.fromAnchorages(Seq(anchorages2))
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

  val expected = (VesselMetadata(45),
                  immutable.Seq(
                    AnchorageGroupVisit(
                      AnchorageGroup(LatLon(0.0.of[degrees], 0.0.of[degrees]), Set(anchorages1)),
                      Instant.parse("2016-01-01T00:00:00.000Z"),
                      Instant.parse("2016-01-01T00:08:00.000Z")),
                    AnchorageGroupVisit(
                      AnchorageGroup(LatLon(0.0.of[degrees], 1.0.of[degrees]), Set(anchorages2)),
                      Instant.parse("2016-01-01T02:00:00.000Z"),
                      Instant.parse("2016-01-01T02:08:00.000Z"))))

  "Vessel" should "visit the correct anchorages" in {
    runWithContext { sc =>
      val vesselRecords = sc.parallelize(Seq((VesselMetadata(45), vesselPath)))
      val res = Anchorages.findAnchorageGroupVisits(
        vesselRecords,
        sc.parallelize(anchorageGroups)
      )

      res should containSingleValue(expected)
    }
  }
}

class AnchoragesGroupingTests extends PipelineSpec with Matchers {
  import TestHelper._
  import AdditionalUnits._

  def anchorageFromS2CellToken(token: String, vessels: Seq[VesselMetadata]) =
    Anchorage(LatLon.fromS2CellId(S2CellId.fromToken(token)), vessels, 0)

  "Anchorage merging" should "work!" in {
    val anchorages = IndexedSeq(
      anchorageFromS2CellToken("89c19c9c",
                               Seq(VesselMetadata(1), VesselMetadata(2), VesselMetadata(3))),
      anchorageFromS2CellToken("89c19b64", Seq(VesselMetadata(1), VesselMetadata(2))),
      anchorageFromS2CellToken("89c1852c", Seq(VesselMetadata(1))),
      anchorageFromS2CellToken("89c19b04", Seq(VesselMetadata(1), VesselMetadata(2))),
      anchorageFromS2CellToken("89c19bac", Seq(VesselMetadata(1), VesselMetadata(2))),
      anchorageFromS2CellToken("89c19bb4", Seq(VesselMetadata(1), VesselMetadata(2))))

    val groupedAnchorages = Anchorages.mergeAdjacentAnchorages(anchorages)

    groupedAnchorages should have size 3

    val expected =
      Seq(AnchorageGroup(LatLon(40.016824742437635.of[degrees], -74.07113588841028.of[degrees]),
                         Set(anchorages(2))),
          AnchorageGroup(LatLon(39.994377589412146.of[degrees], -74.12517039688245.of[degrees]),
                         Set(anchorages(0), anchorages(1))),
          AnchorageGroup(LatLon(39.96842156104703.of[degrees], -74.0828838592642.of[degrees]),
                         Set(anchorages(3), anchorages(4), anchorages(5))))

    groupedAnchorages should contain theSameElementsAs expected
  }
}
