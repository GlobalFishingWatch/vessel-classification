package org.skytruth.ais_annotator

import com.spotify.scio._
import com.spotify.scio.values.SCollection
import com.spotify.scio.bigquery._
import com.typesafe.scalalogging.{LazyLogging, Logger}
import org.joda.time.{Instant}
import org.skytruth.common.{IteratorWithCurrent}
import scala.collection.{mutable, immutable}

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
        activeAnnotations = activeAnnotations.filter { _.endTime.isAfter(ts) }

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
    logger.info("Hello world!")
  }
}
