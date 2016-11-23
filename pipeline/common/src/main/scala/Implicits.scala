package org.skytruth.common

import io.github.karols.units._
import io.github.karols.units.SI._
import io.github.karols.units.defining._

import com.typesafe.scalalogging.{LazyLogging, Logger}
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

    // TODO(alexwilson): this is not a true median atm, because for an even
    // number of elements it does not average the two central values but picks
    // the lower arbitrarily. This is just to avoid having to have addition
    // and division also defined for T.
    def medianBy[V <% Ordered[V]](fn: T => V): T = {
      val asIndexedSeq = iterable.toIndexedSeq.sortBy(fn)
      asIndexedSeq.apply(asIndexedSeq.size / 2)
    }
  }

  implicit class RicherDoubleUIterable[T <: MUnit](val iterable: Iterable[DoubleU[T]]) {
    def mean: DoubleU[T] = {
      var acc = 0.0
      var count = 0
      iterable.foreach { v =>
        acc += v.value
        count += 1
      }
      (acc / count.toDouble).of[T]
    }

    def weightedMean(weights: Iterable[Double]): DoubleU[T] = {
      var acc = 0.0
      var weight = 0.0
      iterable.zip(weights).foreach {
        case (v, w) =>
          acc += v.value * w
          weight += w
      }
      (acc / weight).of[T]
    }
  }
}