package org.skytruth.common

import io.github.karols.units._
import io.github.karols.units.SI._

import org.joda.time.{Duration, Instant}

object InputDataParameters {
  // TODO(alexwilson): This should be per year.
  val minRequiredPositions = 1000
  val minTimeBetweenPoints = Duration.standardMinutes(5)

  val stationaryPeriodMaxDistance = 0.8.of[kilometer]
  val stationaryPeriodMinDuration = Duration.standardHours(2 * 24)

  val inputMeasuresPath =
    "gs://benthos-pipeline/data-production-740/measures-pipeline/st-segment"
  def measuresPathPattern(year: String) =
    s"${inputMeasuresPath}/$year-*-*/*-of-*"

  val knownFishingMMSIs = "feature-pipeline/src/main/data/treniformis_known_fishing_mmsis_2016.txt"

  val minValidTime = Instant.parse("2012-01-01T00:00:00Z")
  lazy val maxValidTime = Instant.now()
}
