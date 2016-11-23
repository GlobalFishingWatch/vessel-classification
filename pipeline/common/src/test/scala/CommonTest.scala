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
