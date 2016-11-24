package org.skytruth.feature_pipeline

import io.github.karols.units._
import io.github.karols.units.SI._

import org.joda.time.{Duration, Instant}

object Parameters {
  val adjacencyResamplePeriod = Duration.standardMinutes(10)
  val maxInterpolateGap = Duration.standardMinutes(60)

  val maxClosestNeighbours = 10
  val maxEncounterRadius = 1.0.of[kilometer]

  val maxDistanceForEncounter = 0.5.of[kilometer]
  val minDurationForEncounter = Duration.standardHours(3)
  val minDistanceToShoreForEncounter = 20.0.of[kilometer]
}
