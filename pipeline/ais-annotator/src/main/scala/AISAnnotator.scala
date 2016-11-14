package org.skytruth.ais_annotator

import com.google.cloud.dataflow.sdk.runners.{DataflowPipelineRunner}
import com.google.cloud.dataflow.sdk.options.DataflowPipelineOptions
import com.spotify.scio._
import com.spotify.scio.values.SCollection
import com.spotify.scio.bigquery._
import com.typesafe.scalalogging.{LazyLogging, Logger}
import java.util.LinkedHashMap
import org.joda.time.{Instant}
import org.skytruth.common.{GcpConfig, IteratorWithCurrent}
import org.skytruth.common.ScioContextResource._
import scala.collection.{mutable, immutable}
import resource._

case class JSONFileAnnotatorConfig(inputFilePattern: String, timerangeFieldName: String)

case class AnnotatorConfig(
    inputFilePatterns: Seq[String],
    outputFilePath: String,
    jsonAnnotations: Seq[JSONFileAnnotatorConfig]
)

case class MessageAnnotation(mmsi: Int,
                             name: String,
                             startTime: Instant,
                             endTime: Instant,
                             value: Double)

object AISAnnotator extends LazyLogging {

  // TODO(alexwilson):
  //
  // * Config for message annotations (JSON + CSV).
  // * Readers for different types of annotations.
  // * YAML config file for job?

  def jsonAnnotationReader(annotations: SCollection[TableRow],
                           valueFieldName: String): SCollection[MessageAnnotation] = {
    annotations.flatMap { json =>
      val mmsi = json.getLong("mmsi").toInt

      val valueFieldList =
        json.getRepeated(valueFieldName).map(_.asInstanceOf[LinkedHashMap[String, Any]])

      valueFieldList.map { valueField =>
        val startTime = Instant.parse(valueField.get("start_time").asInstanceOf[String])
        val endTime = Instant.parse(valueField.get("end_time").asInstanceOf[String])
        val value = valueField.get("value").asInstanceOf[Double]

        MessageAnnotation(mmsi, valueFieldName, startTime, endTime, value)
      }
    }
  }

  def annotateVesselMessages(messages: Iterable[TableRow],
                             annotations: Iterable[MessageAnnotation]): Seq[TableRow] = {
    // Sort messages and annotations by (start) time.
    val sortedMessages = messages.map { msg =>
      (Instant.parse(msg.getString("timestamp")), msg)
    }
    // Ensure only a single message per timestamp.
      .groupBy(_._1)
      .map(_._2.head)
      .toSeq
      .sortBy(_._1.getMillis)

    val sortedAnnotations = annotations.groupBy(_.startTime).toSeq.sortBy(_._1.getMillis)

    var activeAnnotations = List[MessageAnnotation]()
    var annotationIterator = IteratorWithCurrent(sortedAnnotations.iterator)

    // Annotate each message with any active timerange values.
    sortedMessages.map {
      case (ts, msg) =>
        // Remove annotations from the past.
        activeAnnotations = activeAnnotations.filter { !_.endTime.isBefore(ts) }

        // Add any newly-active annotations.
        while (annotationIterator.current.isDefined && !ts.isBefore(
                 annotationIterator.current.get._1)) {
          val annotations = annotationIterator.current.get._2

          activeAnnotations ++= annotations.filter { _.endTime.isAfter(ts) }

          annotationIterator.getNext()
        }

        val clonedMessage = msg.clone()
        activeAnnotations.map { annotation =>
          clonedMessage.set(annotation.name, annotation.value)
        }

        clonedMessage
    }
  }

  def annotateAllMessages(
      aisMessageInputs: Seq[SCollection[TableRow]],
      annotationInputs: Seq[SCollection[MessageAnnotation]]): SCollection[TableRow] = {

    val aisMessages = SCollection.unionAll(aisMessageInputs)

    // Remove all but location messages and key by mmsi.
    val filteredByMmsi = aisMessages
    // Keep only records with a location.
      .filter(json => json.containsKey("lat") && json.containsKey("lon"))
      // Build a typed location record with units of measure.
      .groupBy(_.getLong("mmsi").toInt)

    val annotationsByMmsi = SCollection.unionAll(annotationInputs).groupBy(_.mmsi)

    filteredByMmsi.fullOuterJoin(annotationsByMmsi).flatMap {
      case (mmsi, (messages, annotations)) =>
        annotateVesselMessages(messages.toSeq.flatten, annotations.toSeq.flatten)
    }
  }

  def main(argArray: Array[String]) {
    val (options, remaining_args) = ScioContext.parseArguments[DataflowPipelineOptions](argArray)
    val environment = remaining_args.required("env")
    val jobName = remaining_args.required("job-name")

    val config = GcpConfig.makeConfig(environment, jobName)

    options.setRunner(classOf[DataflowPipelineRunner])
    options.setProject(config.projectId)
    options.setStagingLocation(config.dataflowStagingPath)

    managed(ScioContext(options)).acquireAndGet((sc) => {
      // TODO(alexwilson): Read from YAML. See http://stackoverflow.com/questions/19441400/working-with-yaml-for-scala
      val exampleAnnotatorConfig = AnnotatorConfig(
        Seq(
          "gs://new-benthos-pipeline/data-production/measures-pipeline/st-segment/2015-*-*/*.json"),
        config.pipelineOutputPath + "/annotated",
        Seq(JSONFileAnnotatorConfig("gs://somewhere-or-other", "fishing"))
      )

      val inputData = exampleAnnotatorConfig.inputFilePatterns.map { path =>
        sc.tableRowJsonFile(path)
      }

      val annotations = exampleAnnotatorConfig.jsonAnnotations.map {
        case annotation =>
          val inputAnnotationFile = sc.tableRowJsonFile(annotation.inputFilePattern)
          jsonAnnotationReader(inputAnnotationFile, annotation.timerangeFieldName)
      }

      val annotated = annotateAllMessages(inputData, annotations)

      annotated.saveAsTextFile(exampleAnnotatorConfig.outputFilePath)
    })
  }
}
