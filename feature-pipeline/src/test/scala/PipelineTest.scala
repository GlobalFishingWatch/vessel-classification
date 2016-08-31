package org.skytruth.feature_pipeline

import org.scalatest._

class PipelineSpec extends FlatSpec with Matchers {
  "Any language" should "implement addition correctly" in {
    (3 + 1) should be(4)
  }
}
