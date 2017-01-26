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

import org.scalatest._

class CommonTest extends FlatSpec with Matchers {
  import Implicits._

  "Richer iterable" should "correctly support countBy" in {
    val input = Seq(1, 4, 3, 6, 4, 1, 1, 1, 3)

    input.countBy(x => x * 2) should contain allOf (2 -> 4, 6 -> 2, 8 -> 2, 12 -> 1)
  }

  "Richer iterable" should "correctly support medianBy" in {
    val input1 = Seq(1)
    input1.medianBy(Predef.identity) should equal(1)

    val input2 = Seq(1, 2, 3)
    input2.medianBy(Predef.identity) should equal(2)

    val input3 = Seq(1, 4, 3, 6, 4, 1, 4, 1, 4, 3)
    input3.medianBy(Predef.identity) should equal(4)
  }
}
