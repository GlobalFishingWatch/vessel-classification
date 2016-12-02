package org.skytruth.ais_annotator

import com.fasterxml.jackson.databind.{ObjectMapper, SerializationFeature}
import com.spotify.scio.testing.PipelineSpec
import com.spotify.scio.values.SCollection
import org.joda.time.{Instant}
import org.json4s._
import org.json4s.JsonAST.JValue
import org.json4s.JsonDSL.WithDouble._
import org.json4s.native.JsonMethods._
import org.scalatest._
import scala.collection.JavaConverters._

class AnnotatorTests extends PipelineSpec with Matchers {

  private def tr(mmsi: Int,
                 timestamp: String,
                 lat: Double,
                 lon: Double,
                 fields: Seq[(String, Double)]): JValue = {
    var baseJson: JValue = ("mmsi" -> mmsi) ~
        ("timestamp" -> timestamp) ~
        ("lat" -> lat) ~
        ("lon" -> lon)
    fields.foreach {
      case (key, value) =>
        val asJson: JValue = (key -> value)
        baseJson = baseJson merge asJson
    }
    baseJson
  }

  private def jsonFromString(lines: SCollection[String]) = lines.map(l => parse(l))

  "The annotator" should "annotate" in {
    val msgs = Seq(
      tr(123, "20150101T01:00:00Z", 0.0, 0.0, Seq()),
      tr(123, "20150101T02:00:00Z", 0.0, 0.0, Seq()),
      tr(123, "20150101T03:00:00Z", 0.0, 0.0, Seq()),
      tr(123, "20150101T04:00:00Z", 0.0, 0.0, Seq()),
      tr(123, "20150101T05:00:00Z", 0.0, 0.0, Seq()),
      tr(123, "20150101T06:00:00Z", 0.0, 0.0, Seq()),
      tr(123, "20150101T07:00:00Z", 0.0, 0.0, Seq()),
      tr(123, "20150101T08:00:00Z", 0.0, 0.0, Seq()),
      tr(123, "20150101T09:00:00Z", 0.0, 0.0, Seq())
    )

    runWithContext { sc =>
      val annot1 = Seq(MessageAnnotation(123,
                                         "height",
                                         Instant.parse("20150101T00:00:00Z"),
                                         Instant.parse("20150101T04:00:00Z"),
                                         2.5),
                       MessageAnnotation(123,
                                         "height",
                                         Instant.parse("20150101T06:00:00Z"),
                                         Instant.parse("20150101T08:00:00Z"),
                                         4.5))

      val annot2 = Seq(MessageAnnotation(123,
                                         "weight",
                                         Instant.parse("20150101T06:00:00Z"),
                                         Instant.parse("20150101T10:00:00Z"),
                                         100.0),
                       MessageAnnotation(456,
                                         "weight",
                                         Instant.parse("20150101T07:00:00Z"),
                                         Instant.parse("20150101T10:00:00Z"),
                                         32.0))

      val annotations = Seq(sc.parallelize(annot1), sc.parallelize(annot2))
      val res = AISAnnotator.annotateAllMessages(Seq(sc.parallelize(msgs)), annotations)

      res should haveSize(9)

      val expected = Seq(
        tr(123, "20150101T01:00:00Z", 0.0, 0.0, Seq("height" -> 2.5)),
        tr(123, "20150101T02:00:00Z", 0.0, 0.0, Seq("height" -> 2.5)),
        tr(123, "20150101T03:00:00Z", 0.0, 0.0, Seq("height" -> 2.5)),
        tr(123, "20150101T04:00:00Z", 0.0, 0.0, Seq("height" -> 2.5)),
        tr(123, "20150101T05:00:00Z", 0.0, 0.0, Seq()),
        tr(123, "20150101T06:00:00Z", 0.0, 0.0, Seq("height" -> 4.5, "weight" -> 100.0)),
        tr(123, "20150101T07:00:00Z", 0.0, 0.0, Seq("height" -> 4.5, "weight" -> 100.0)),
        tr(123, "20150101T08:00:00Z", 0.0, 0.0, Seq("height" -> 4.5, "weight" -> 100.0)),
        tr(123, "20150101T09:00:00Z", 0.0, 0.0, Seq("weight" -> 100.0))
      )

      res should containInAnyOrder(expected)

    }
  }

  "The annotator" should "parse out JSON timeranges" in {
    val inputLines = Seq(
      """{"mmsi": 123, "heights": [{"start_time": "20150101T01:00:00Z", "end_time": "20150101T07:00:00Z", "value": 4.5}]}"""
    )
    runWithContext { sc =>
      val tableRows = jsonFromString(sc.parallelize(inputLines))
      val res = AISAnnotator.jsonAnnotationReader(tableRows, "heights_out", "heights", 2.0)

      res should containInAnyOrder(
        Seq(
          MessageAnnotation(123,
                            "heights_out",
                            Instant.parse("20150101T01:00:00Z"),
                            Instant.parse("20150101T07:00:00Z"),
                            4.5)))
    }
  }

  "The annotator" should "parse config from YAML correctly" in {
    val input = """
    |inputFilePatterns:
    |  - foo
    |  - bar
    |outputFilePath: baz
    |jsonAnnotations:
    |  - inputFilePattern: one
    |    outputFieldName: foo
    |    timeRangeFieldName: two
    |    defaultValue: 6.8
    """.stripMargin('|')

    val res = AISAnnotator.readYamlConfig(input)

    res.inputFilePatterns should contain theSameElementsAs Seq("foo", "bar")
    res.outputFilePath should equal("baz")
    res.jsonAnnotations should have size (1)
    res.jsonAnnotations.head.inputFilePattern should equal("one")
    res.jsonAnnotations.head.outputFieldName should equal("foo")
    res.jsonAnnotations.head.timeRangeFieldName should equal("two")
    res.jsonAnnotations.head.defaultValue should equal(6.8)
  }
}
