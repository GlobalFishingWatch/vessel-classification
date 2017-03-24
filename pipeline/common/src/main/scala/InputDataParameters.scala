// Copyright 2017 Google Inc. and Skytruth Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

  val knownFishingMMSIs = "feature-pipeline/src/main/data/treniformis_known_fishing_mmsis_2016.txt"

  val minValidTime = Instant.parse("2012-01-01T00:00:00Z")
  lazy val maxValidTime = Instant.now()

  val defaultYearsToRun = Seq("2012", "2013", "2014", "2015", "2016")
  val defaultDataFileGlob = "-*-*/*-of-*"

  def dataFileGlobPerYear(dataYearsArg: Seq[String], dataFileGlob: String, 
          measuresPath: String = inputMeasuresPath): Seq[String] = {
    val dataYears = if (dataYearsArg.isEmpty) {
      InputDataParameters.defaultYearsToRun
    } else {
      dataYearsArg
    }

    dataYears.map { year =>
      s"${measuresPath}/$year$dataFileGlob"
    }
  }
}
