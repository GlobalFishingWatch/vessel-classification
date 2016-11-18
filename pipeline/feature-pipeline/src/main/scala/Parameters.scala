package org.skytruth.feature_pipeline

import io.github.karols.units._
import io.github.karols.units.SI._

import org.joda.time.{Duration, Instant}

object Parameters {
  // TODO(alexwilson): This should be per year.
  val minRequiredPositions = 1000
  val minTimeBetweenPoints = Duration.standardMinutes(5)

  val stationaryPeriodMaxDistance = 0.8.of[kilometer]
  val stationaryPeriodMinDuration = Duration.standardHours(2 * 24)

  // TODO(alexwilson): remove years list when cloud dataflow text source can
  // handle our volume of files.
  val allDataYears = List("2012", "2013", "2014", "2015", "2016")
  //val allDataYears = List("2015")
  val inputMeasuresPath =
    "gs://benthos-pipeline/data-production-740/measures-pipeline/st-segment"
  def measuresPathPattern(year: String) =
    s"${Parameters.inputMeasuresPath}/$year-*-*/*-of-*"

  val knownFishingMMSIs = "feature-pipeline/src/main/data/treniformis_known_fishing_mmsis_2016.txt"

  val minValidTime = Instant.parse("2012-01-01T00:00:00Z")
  lazy val maxValidTime = Instant.now()

  val trainingSplit = "Training"
  val testSplit = "Test"
  val unclassifiedSplit = "Unclassified"
  val splits = Seq(trainingSplit, testSplit, unclassifiedSplit)

  // Around 1km^2
  val anchoragesS2Scale = 13
  val minUniqueVesselsForAnchorage = 20
  val anchorageVisitDistanceThreshold = 0.5.of[kilometer]
  val minAnchorageVisitDuration = Duration.standardMinutes(60)

  val adjacencyResamplePeriod = Duration.standardMinutes(10)
  val maxInterpolateGap = Duration.standardMinutes(60)

  val maxClosestNeighbours = 10
  val maxEncounterRadius = 1.0.of[kilometer]

  val maxDistanceForEncounter = 0.5.of[kilometer]
  val minDurationForEncounter = Duration.standardHours(3)
  val minDistanceToShoreForEncounter = 20.0.of[kilometer]
}
