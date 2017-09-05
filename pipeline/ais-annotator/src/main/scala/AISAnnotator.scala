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
import org.joda.time.{Instant, Duration}
import org.joda.time.DateTimeZone.UTC
import org.joda.time.format.ISODateTimeFormat
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
    knownFishingMMSIs: String,
    jsonAnnotations: Seq[JSONFileAnnotatorConfig]
)

case class MessageAnnotation(mmsi: Int,
                             name: String,
                             startTime: Instant,
                             endTime: Instant,
                             value: Double,
                             weight: Double)

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
                           defaultValue: Double,
                           min_time: Option[Instant],
                           max_time: Option[Instant]): SCollection[MessageAnnotation] = {
    annotations.flatMap { json =>
      val mmsi = (json \ "mmsi").extract[Int]

      val timeRangeFieldList = (json \ timeRangeFieldName).extract[List[JValue]]

      timeRangeFieldList.flatMap { timeRangeField =>
        val startTime = Instant.parse(timeRangeField.getString("start_time"))
        val endTime = Instant.parse(timeRangeField.getString("end_time"))
        val value = if (timeRangeField.has("value")) {
          timeRangeField.getDouble("value")
        } else {
          defaultValue
        }
        val weight = if (timeRangeField.has("weight")) {
          timeRangeField.getDouble("weight")
        } else {
          1
        }     


        if ((startTime.getMillis > max_time.getOrElse(startTime).getMillis) || 
                (endTime.getMillis < min_time.getOrElse(startTime).getMillis)) None
          else Some(MessageAnnotation(mmsi, outputFieldName, startTime, endTime, value, weight))
      }
    }
  }

  def annotateVesselMessages(messages: Iterable[JValue],
                             annotations: Iterable[MessageAnnotation]): Seq[JValue] = {
    // Sort messages and annotations by (start) time.
    val sortedMessages = messages.map { msg =>
      (Instant.parse(msg.getString("timestamp").replace(" UTC", "Z").replace(" ", "T")), msg)
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
        activeAnnotations.groupBy(_.name).map {
          case (name, annotation_list) => {
            val json: JValue = (name -> 
              // Note that we round to nearest hundreth here
              (math rint annotation_list.map(x => x.value * x.weight).sum / annotation_list.map(_.weight).sum * 100) / 100)
            annotatedMsg = annotatedMsg merge json
          }
        }

        annotatedRows.append(annotatedMsg)
    }

    annotatedRows.toSeq
  }

  def annotateAllMessages(
      allowedMMSIs: Set[Int],
      aisMessages: SCollection[JValue],
      annotationInputs: Seq[SCollection[MessageAnnotation]]): SCollection[JValue] = {

    val allAnnotations = SCollection.unionAll(annotationInputs)
    val annotationsByMmsi = allAnnotations.groupBy(x => 
      (x.mmsi, x.startTime.toDateTime(UTC).getYear(), x.startTime.toDateTime(UTC).getDayOfYear()))

    // Remove all but location messages and key by mmsi and day.
    val filteredAISMessages = aisMessages
    .filter {
      json =>  json.has("lat") && json.has("lon") && json.has("timestamp")
    }
    .map { json =>
      val dateTime = Instant.parse(json.getString("timestamp").replace(" UTC", "Z").replace(" ", "T")).toDateTime(UTC)
      ((json.getLong("mmsi").toInt, dateTime.getYear(), dateTime.getDayOfYear()), json)
    }
    .filter { case ((mmsi, _, _), _) =>
      !AISDataProcessing.blacklistedMmsis.contains(mmsi) && (
        allowedMMSIs.isEmpty || allowedMMSIs.contains(mmsi))
    }

    val filteredGroupedByMmsi = filteredAISMessages.groupByKey

    // TODO: figure out how to add back in unfiltered messages
    filteredGroupedByMmsi.leftOuterJoin(annotationsByMmsi).flatMap {
      case (_, (messagesIt, Some(annotationsIt))) =>
        val messages = messagesIt.toSeq
        val annotations = annotationsIt.toSeq
        annotateVesselMessages(messages, annotations)
      case (_, (messagesIt, None)) =>
        val messages = messagesIt.toSeq
        annotateVesselMessages(messages, Nil)
    }
  }

  def readJsonFile(sc: ScioContext, filePath: String) = sc.textFile(filePath).map(l => parse(l))

  def main(argArray: Array[String]) {
    val (options, remaining_args) = ScioContext.parseArguments[DataflowPipelineOptions](argArray)
    val environment = remaining_args.required("env")
    val jobName = remaining_args.required("job-name")
    val jobConfigurationFile = remaining_args.required("job-config")
    val outputFilePath = remaining_args.required("output-path")
    val startDate = remaining_args.optional("annotation-start")
    val endDate = remaining_args.optional("annotation-end")

    val config = GcpConfig.makeConfig(environment, jobName)

    options.setRunner(classOf[DataflowPipelineRunner])
    options.setProject(config.projectId)
    options.setStagingLocation(config.dataflowStagingPath)



    val annotatorConfig = managed(scala.io.Source.fromFile(jobConfigurationFile)).acquireAndGet {
      s =>
        readYamlConfig(s.mkString)
    }

    val includedMMSIs = if (annotatorConfig.knownFishingMMSIs != "") {
      AISDataProcessing.loadFishingMMSIs(annotatorConfig.knownFishingMMSIs)
    } else {
      Set[Int]()
    }

    managed(ScioContext(options)).acquireAndGet { sc =>
      val inputData = annotatorConfig.inputFilePatterns.map { path =>
        readJsonFile(sc, path)
      }

      val aisMessages = SCollection.unionAll(inputData)

      // TODO verify that min_time is start of day, till then pull back 24 hours
      val min_time = startDate.map(Instant.parse(_, ISODateTimeFormat.date()).withDurationAdded(Duration.standardHours(24), -1))
      // Push Max time forward 24 hours to include whole day.
      val max_time = endDate.map(Instant.parse(_, ISODateTimeFormat.date()).withDurationAdded(Duration.standardHours(24), 1))

      val annotations = annotatorConfig.jsonAnnotations.map {
        case annotation =>
          val inputAnnotationFile = readJsonFile(sc, annotation.inputFilePattern)
          jsonAnnotationReader(inputAnnotationFile,
                               annotation.outputFieldName,
                               annotation.timeRangeFieldName,
                               annotation.defaultValue,
                               min_time,
                               max_time)
      }



      val annotated = annotateAllMessages(includedMMSIs, aisMessages, annotations)
      val annotatedToString = annotated.map(json => compact(render(json)))

      annotatedToString.saveAsTextFile(outputFilePath)
    }
  }
}
