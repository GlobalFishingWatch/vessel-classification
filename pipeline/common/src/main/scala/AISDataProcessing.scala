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

package org.skytruth.common

import io.github.karols.units._
import io.github.karols.units.SI._
import io.github.karols.units.defining._

import com.opencsv.CSVReader
import com.spotify.scio._
import com.spotify.scio.values.SCollection
import com.spotify.scio.bigquery._
import com.typesafe.scalalogging.{LazyLogging, Logger}
import java.io.{FileReader}
import org.apache.commons.math3.util.MathUtils
import org.joda.time.{Duration, Instant}
import org.json4s._
import org.json4s.JsonAST.JValue
import org.json4s.JsonDSL.WithDouble._
import org.json4s.native.JsonMethods._
import scala.collection.{mutable, immutable}
import scala.collection.JavaConversions._

import AdditionalUnits._
import Implicits._

case class RangeValidator(valid: Boolean) extends AnyVal {
  def inRange[T <: Ordered[T]](value: T, min: T, max: T) =
    RangeValidator(valid && (value >= min && value < max))
  // TODO(alexwilson): Both of these could be removed if Instant and MUnit can be
  // made to be Ordered, with appropriate use of implicits.
  def inRange(value: Instant, min: Instant, max: Instant) =
    RangeValidator(valid && (value.getMillis >= min.getMillis && value.getMillis < max.getMillis))
  def inRange[T <: MUnit](value: DoubleU[T], min: DoubleU[T], max: DoubleU[T]) =
    RangeValidator(valid && (value.value >= min.value && value.value < max.value))
}

object RangeValidator {
  def apply() = new RangeValidator(true)
}

object AISDataProcessing extends LazyLogging {
  lazy val blacklistedMmsis = Set(0, 12345)

  // Normalize from -180 to + 180
  def angleNormalize(angle: DoubleU[degrees]) =
    MathUtils.normalizeAngle(angle.convert[radians].value, 0.0).of[radians].convert[degrees]

  implicit val formats = DefaultFormats
  implicit class JValueExtended(value: JValue) {
    def has(field: String) = ((value \ field) != JNothing)
    def getString(field: String) = (value \ field).extractOpt[String].getOrElse("")
    def getLong(field: String) = (value \ field).extractOpt[Long].getOrElse(0L)
    def getDouble(field: String) = (value \ field).extractOpt[Double].getOrElse(0.0)
  }

  // Reads JSON vessel records, filters to only location records, groups by MMSI and sorts
  // by ascending timestamp.
  def readJsonRecords(
      inputs: Seq[SCollection[String]],
      knownFishingMMSIs: Set[Int],
      minRequiredPositions: Long): SCollection[(VesselMetadata, Seq[VesselLocationRecord])] = {

    val input = SCollection.unionAll(inputs)
    // Keep only records with a location.
    val validRecords = input
      .map(line => parse(line))
      .filter(json => json.has("lat") && json.has("lon"))
      // Build a typed location record with units of measure.
      .map(json => {
        val mmsi = (json \ "mmsi").extract[Int]
        val metadata = VesselMetadata(mmsi, knownFishingMMSIs.contains(mmsi))
        val record =
          // TODO(alexwilson): Double-check all these units are correct.
          VesselLocationRecord(Instant.parse(json.getString("timestamp").replace(" UTC", "Z").replace(" ", "T")),
                               LatLon(angleNormalize(json.getDouble("lat").of[degrees]),
                                      angleNormalize(json.getDouble("lon").of[degrees])),
                               (json.getDouble("distance_from_shore") / 1000.0).of[kilometer],
                               json.getDouble("speed").of[knots],
                               angleNormalize(json.getDouble("course").of[degrees]),
                               angleNormalize(json.getDouble("heading").of[degrees]))
        (metadata, record)
      })
      .filter { case (metadata, _) => !blacklistedMmsis.contains(metadata.mmsi) }
      .filter {
        case (_, record) =>
          RangeValidator()
            .inRange(record.timestamp,
                     InputDataParameters.minValidTime,
                     InputDataParameters.maxValidTime)
            .inRange(record.location.lat, -90.0.of[degrees], 90.0.of[degrees])
            .inRange(record.location.lon, -180.0.of[degrees], 180.of[degrees])
            .inRange(record.distanceToShore, 0.0.of[kilometer], 20000.0.of[kilometer])
            .inRange(record.speed, 0.0.of[knots], 100.0.of[knots])
            .inRange(record.course, -180.0.of[degrees], 180.of[degrees])
            .inRange(record.heading, -180.0.of[degrees], 180.of[degrees])
            .valid
      }

    validRecords.groupByKey.flatMap {
      case (metadata, records) =>
        if (records.size >= minRequiredPositions) {
          val dedupedSorted = records.toIndexedSeq
          // On occasion the same message seems to appear twice in the record. Remove.
          .distinct.sortBy(_.timestamp.getMillis)
          Some((metadata, dedupedSorted))
        } else {
          None
        }
    }
  }

  // TODO: consider interpolating onto hour grid instead of thinning
  def thinPoints(records: Iterable[VesselLocationRecord]): Iterable[VesselLocationRecord] = {
    val thinnedPoints = mutable.ListBuffer.empty[VesselLocationRecord]

    // Thin locations down to a minimum time between each.
    records.foreach { vr =>
      if (thinnedPoints.isEmpty || !vr.timestamp.isBefore(
            thinnedPoints.last.timestamp.plus(InputDataParameters.minTimeBetweenPoints))) {
        thinnedPoints.append(vr)
      }
    }

    thinnedPoints
  }

  def removeStationaryPeriods(records: Iterable[VesselLocationRecord],
                              stationaryPeriodMinDuration: Duration): ProcessedLocations = {
    // Remove long stationary periods from the record: anything over the threshold
    // time will be reduced to just the start and end points of the period.
    // TODO(alexwilson): Tim points out that leaves vessels sitting around for t - delta looking
    // significantly different from those sitting around for t + delta. Consider his scheme of just
    // cropping all excess time over the threshold instead.
    val withoutLongStationaryPeriods = mutable.ListBuffer.empty[VesselLocationRecord]
    val stationaryPeriods = mutable.ListBuffer.empty[StationaryPeriod]
    val currentPeriod = mutable.Queue.empty[VesselLocationRecord]
    records.foreach { vr =>
      if (!currentPeriod.isEmpty) {
        val periodFirst = currentPeriod.front
        val speed = vr.speed
        val distanceDelta = vr.location.getDistance(periodFirst.location)
        if (distanceDelta > InputDataParameters.stationaryPeriodMaxDistance) {
          if (vr.timestamp.isAfter(periodFirst.timestamp.plus(stationaryPeriodMinDuration))) {
            withoutLongStationaryPeriods.append(periodFirst)
            if (currentPeriod.last != periodFirst) {
              withoutLongStationaryPeriods.append(currentPeriod.last)
            }
            val numPoints = currentPeriod.length.toDouble
            val duration = new Duration(periodFirst.timestamp, currentPeriod.last.timestamp)
            val aveLatLon = LatLon.mean(currentPeriod.map { _.location })
            val meanDistanceToShore = currentPeriod.map { _.distanceToShore }.mean
            val meanDriftRadius = currentPeriod.map { _.location.getDistance(aveLatLon) }.mean

            stationaryPeriods.append(
              StationaryPeriod(aveLatLon, duration, meanDistanceToShore, meanDriftRadius))
          } else {
            withoutLongStationaryPeriods ++= currentPeriod
          }

          currentPeriod.clear()
        }
      }

      currentPeriod.enqueue(vr)
    }
    withoutLongStationaryPeriods ++= currentPeriod

    ProcessedLocations(withoutLongStationaryPeriods.toIndexedSeq, stationaryPeriods.toIndexedSeq)
  }

  def filterAndProcessVesselRecords(
      input: SCollection[(VesselMetadata, Seq[VesselLocationRecord])],
      stationaryPeriodMinDuration: Duration): SCollection[(VesselMetadata, ProcessedLocations)] = {
    input.map {
      case (metadata, records) =>
        val thinnedPoints = thinPoints(records)
        val processedLocations =
          removeStationaryPeriods(thinnedPoints, stationaryPeriodMinDuration)

        (metadata, processedLocations)
    }
  }

  def loadFishingMMSIs(knownFishingMMSIs:String): Set[Int] = {
    val fishingMMSIreader = new CSVReader(new FileReader(knownFishingMMSIs))
    val mmsis = fishingMMSIreader
      .readAll()
      .map { l =>
        l(0).toInt
      }
      .toSet
    logger.info(s"mmsi count: ${mmsis.size}")
    mmsis
  }
}
