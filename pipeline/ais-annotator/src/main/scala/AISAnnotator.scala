package org.skytruth.ais_annotator

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import com.google.cloud.dataflow.sdk.runners.{DataflowPipelineRunner}
import com.google.cloud.dataflow.sdk.options.DataflowPipelineOptions
import com.spotify.scio._
import com.spotify.scio.values.SCollection
import com.spotify.scio.bigquery._
import com.typesafe.scalalogging.{LazyLogging, Logger}
import java.util.LinkedHashMap
import org.joda.time.{Instant}
import org.skytruth.common.{GcpConfig, IteratorWithCurrent, ValueCache}
import org.skytruth.common.ScioContextResource._
import scala.collection.{mutable, immutable}

import resource._

case class JSONFileAnnotatorConfig(inputFilePattern: String,
                                   outputFieldName: String,
                                   timeRangeFieldName: String,
                                   defaultValue: Double)

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

  def readYamlConfig(config: String): AnnotatorConfig = {
    val mapper: ObjectMapper = new ObjectMapper(new YAMLFactory())
    mapper.registerModule(DefaultScalaModule)

    mapper.readValue(config, classOf[AnnotatorConfig])
  }

  def jsonAnnotationReader(annotations: SCollection[TableRow],
                           outputFieldName: String,
                           timeRangeFieldName: String,
                           defaultValue: Double): SCollection[MessageAnnotation] = {
    annotations.flatMap { json =>
      val mmsi = json.getLong("mmsi").toInt

      val timeRangeFieldList =
        json.getRepeated(timeRangeFieldName).map(_.asInstanceOf[LinkedHashMap[String, Any]])

      timeRangeFieldList.map { timeRangeField =>
        val startTime = Instant.parse(timeRangeField.get("start_time").asInstanceOf[String])
        val endTime = Instant.parse(timeRangeField.get("end_time").asInstanceOf[String])
        val value =
          Option(timeRangeField.get("value")).map(_.asInstanceOf[Double]).getOrElse(defaultValue)

        MessageAnnotation(mmsi, outputFieldName, startTime, endTime, value)
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
    var annotatedRows = mutable.ListBuffer[TableRow]()
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

        annotatedRows.append(clonedMessage)
    }

    annotatedRows.toSeq
  }

  def annotateAllMessages(
      aisMessageInputs: Seq[SCollection[TableRow]],
      annotationInputs: Seq[SCollection[MessageAnnotation]]): SCollection[TableRow] = {

    val aisMessages = SCollection.unionAll(aisMessageInputs)

    val annotationsByMmsi = SCollection.unionAll(annotationInputs).groupBy(_.mmsi)
    val mmsisWithAnnotation =
      annotationsByMmsi.map(_._1).map(x => (0, x)).groupByKey.map(_._2.toSet).asSingletonSideInput

    // Do not process messages for MMSIs for which we have no annotations.
    val allowedMMSIs = ValueCache[Set[Int]]()
    val filteredAISMessages = aisMessages.map { json =>
      (json.getLong("mmsi").toInt, json)
    }.withSideInputs(mmsisWithAnnotation)
      .filter {
        case ((mmsi, json), ctx) =>
          val mmsiSet = allowedMMSIs.get(() => ctx(mmsisWithAnnotation))

          mmsiSet.contains(mmsi)
      }
      .toSCollection

    // Remove all but location messages and key by mmsi.
    val filteredGroupedByMmsi = filteredAISMessages
    // Keep only records with a location.
    .filter { case (_, json) => json.containsKey("lat") && json.containsKey("lon") }.groupByKey

    filteredGroupedByMmsi.join(annotationsByMmsi).flatMap {
      case (mmsi, (messagesIt, annotationsIt)) =>
        val messages = messagesIt.toSeq
        val annotations = annotationsIt.toSeq
        annotateVesselMessages(messages, annotations)
    }
  }

  def main(argArray: Array[String]) {
    val (options, remaining_args) = ScioContext.parseArguments[DataflowPipelineOptions](argArray)
    val environment = remaining_args.required("env")
    val jobName = remaining_args.required("job-name")
    val jobConfigurationFile = remaining_args.required("job-config")

    val config = GcpConfig.makeConfig(environment, jobName)

    options.setRunner(classOf[DataflowPipelineRunner])
    options.setProject(config.projectId)
    options.setStagingLocation(config.dataflowStagingPath)

    val annotatorConfig = managed(scala.io.Source.fromFile(jobConfigurationFile)).acquireAndGet {
      s =>
        readYamlConfig(s.mkString)
    }

    managed(ScioContext(options)).acquireAndGet { sc =>
      val inputData = annotatorConfig.inputFilePatterns.map { path =>
        sc.tableRowJsonFile(path)
      }

      val annotations = annotatorConfig.jsonAnnotations.map {
        case annotation =>
          val inputAnnotationFile = sc.tableRowJsonFile(annotation.inputFilePattern)
          jsonAnnotationReader(inputAnnotationFile,
                               annotation.outputFieldName,
                               annotation.timeRangeFieldName,
                               annotation.defaultValue)
      }

      val annotated = annotateAllMessages(inputData, annotations)

      annotated.saveAsTextFile(annotatorConfig.outputFilePath)
    }
  }
}
