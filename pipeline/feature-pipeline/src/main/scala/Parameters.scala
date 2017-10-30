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

package org.skytruth.feature_pipeline

import io.github.karols.units._
import io.github.karols.units.SI._

import org.joda.time.{Duration, Instant}

object Parameters {
  val adjacencyResamplePeriod = Duration.standardMinutes(10)
  val maxInterpolateGap = Duration.standardMinutes(70) // max gap for VMS should be 60 minutes, but leave 10 minutes of leeway

  val maxClosestNeighbours = 10

  val minDistanceToShoreForEncounter = 0.0.of[kilometer]
}
