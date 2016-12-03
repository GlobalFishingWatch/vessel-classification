package org.skytruth.ais_annotator

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import com.google.cloud.dataflow.sdk.runners.{DataflowPipelineRunner}
import com.google.cloud.dataflow.sdk.options.DataflowPipelineOptions
import com.spotify.scio._
import com.spotify.scio.values.SCollection
import com.typesafe.scalalogging.{LazyLogging, Logger}
import java.util.LinkedHashMap
import org.joda.time.{Instant}
import org.json4s._
import org.json4s.JsonAST.JValue
import org.json4s.JsonDSL.WithDouble._
import org.json4s.native.JsonMethods._
import org.skytruth.common._
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
  implicit val formats = DefaultFormats
  import AISDataProcessing.JValueExtended

  def readYamlConfig(config: String): AnnotatorConfig = {
    val mapper: ObjectMapper = new ObjectMapper(new YAMLFactory())
    mapper.registerModule(DefaultScalaModule)

    mapper.readValue(config, classOf[AnnotatorConfig])
  }

  def jsonAnnotationReader(annotations: SCollection[JValue],
                           outputFieldName: String,
                           timeRangeFieldName: String,
                           defaultValue: Double): SCollection[MessageAnnotation] = {
    annotations.flatMap { json =>
      val mmsi = (json \ "mmsi").extract[Int]

      val timeRangeFieldList = (json \ timeRangeFieldName).extract[List[JValue]]

      timeRangeFieldList.map { timeRangeField =>
        val startTime = Instant.parse(timeRangeField.getString("start_time"))
        val endTime = Instant.parse(timeRangeField.getString("end_time"))
        val value = if (timeRangeField.has("value")) {
          timeRangeField.getDouble("value")
        } else {
          defaultValue
        }

        MessageAnnotation(mmsi, outputFieldName, startTime, endTime, value)
      }
    }
  }

  def annotateVesselMessages(messages: Iterable[JValue],
                             annotations: Iterable[MessageAnnotation]): Seq[JValue] = {
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
    var annotatedRows = mutable.ListBuffer[JValue]()
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

        var annotatedMsg = msg
        activeAnnotations.map { annotation =>
          val json: JValue = (annotation.name -> annotation.value)
          annotatedMsg = annotatedMsg merge json
        }

        annotatedRows.append(annotatedMsg)
    }

    annotatedRows.toSeq
  }

  def annotateAllMessages(
      aisMessageInputs: Seq[SCollection[JValue]],
      annotationInputs: Seq[SCollection[MessageAnnotation]]): SCollection[JValue] = {

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
    .filter { case (_, json) => json.has("lat") && json.has("lon") }.groupByKey

    filteredGroupedByMmsi.join(annotationsByMmsi).flatMap {
      case (mmsi, (messagesIt, annotationsIt)) =>
        val messages = messagesIt.toSeq
        val annotations = annotationsIt.toSeq
        annotateVesselMessages(messages, annotations)
    }
  }

  def readJsonFile(sc: ScioContext, filePath: String) = sc.textFile(filePath).map(l => parse(l))

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
        readJsonFile(sc, path)
      }

      val annotations = annotatorConfig.jsonAnnotations.map {
        case annotation =>
          val inputAnnotationFile = readJsonFile(sc, annotation.inputFilePattern)
          jsonAnnotationReader(inputAnnotationFile,
                               annotation.outputFieldName,
                               annotation.timeRangeFieldName,
                               annotation.defaultValue)
      }

      val annotated = annotateAllMessages(inputData, annotations)
      val annotatedToString = annotated.map(json => compact(render(json)))

      annotatedToString.saveAsTextFile(annotatorConfig.outputFilePath)
    }
  }
}
