package org.skytruth.common

import com.typesafe.scalalogging.{LazyLogging, Logger}
import org.joda.time.{DateTime, DateTimeZone}
import org.joda.time.format.ISODateTimeFormat
import scala.collection.{mutable, immutable}

object Implicits {
  implicit class RichLogger(val logger: Logger) {
    def fatal(message: String) = {
      logger.error(message)
      throw new RuntimeException(s"Fatal error: $message")
    }
  }

  implicit class RicherIterable[T](val iterable: Iterable[T]) {
    def countBy[K](fn: T => K): Map[K, Int] = {
      val counts = mutable.Map[K, Int]()
      iterable.foreach { el =>
        val k = fn(el)
        counts(k) = counts.getOrElse(k, 0) + 1
      }
      // Converts to immutable map.
      counts.toMap
    }

    def medianBy[V <% Ordered[V]](fn: T => V): T =
      iterable.toIndexedSeq.sortBy(fn).apply(iterable.size / 2)
  }
}

object GcpConfig extends LazyLogging {
  import Implicits._

  private def projectId = "world-fishing-827"

  // TODO(alexwilson): No locally-generated date for prod. Needs to be sourced
  // from outside so all prod stages share the same path.
  def makeConfig(environment: String, jobId: String) = {
    val now = new DateTime(DateTimeZone.UTC)
    val rootPath = environment match {
      case "prod" => {
        "gs://world-fishing-827/data-production/classification/$jobId"
      }
      case "dev" => {
        val user = sys.env("USER")
        if (user.isEmpty) {
          logger.fatal("USER environment variable cannot be empty for dev runs.")
        }
        s"gs://world-fishing-827-dev-ttl30d/data-production/classification/$user/$jobId"
      }
      case _ => logger.fatal(s"Invalid environment: $environment.")
    }

    GcpConfig(now, projectId, rootPath)
  }
}

case class GcpConfig(startTime: DateTime, projectId: String, private val rootPath: String) {
  def dataflowStagingPath = s"$rootPath/pipeline/staging"
  def pipelineOutputPath = s"$rootPath/pipeline/output"
}
