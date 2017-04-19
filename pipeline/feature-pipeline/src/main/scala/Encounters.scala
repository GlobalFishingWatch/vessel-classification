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

package org.skytruth.feature_pipeline

import io.github.karols.units._
import io.github.karols.units.SI._
import io.github.karols.units.defining._

import com.google.common.geometry.S2CellId
import com.typesafe.scalalogging.LazyLogging
import com.spotify.scio.values.SCollection
import org.joda.time.{Duration, Instant}
import org.skytruth.common._
import org.skytruth.common.AdditionalUnits._
import org.skytruth.common.Implicits._

import scala.collection.{mutable, immutable}
import scala.math._

object Encounters extends LazyLogging {
  def resampleVesselSeries(increment: Duration,
                           input: Seq[VesselLocationRecord]): Seq[ResampledVesselLocation] = {
    val incrementSeconds = increment.getStandardSeconds()
    val maxInterpolateGapSeconds = Parameters.maxInterpolateGap.getStandardSeconds()
    def tsToUnixSeconds(timestamp: Instant): Long = (timestamp.getMillis / 1000L)
    def roundToIncrement(timestamp: Instant): Long =
      (tsToUnixSeconds(timestamp) / incrementSeconds) * incrementSeconds

    var iterTime = roundToIncrement(input.head.timestamp)
    val endTime = roundToIncrement(input.last.timestamp)

    var iterLocation = input.iterator

    val interpolatedSeries = mutable.ListBuffer.empty[ResampledVesselLocation]
    var lastLocationRecord: Option[VesselLocationRecord] = None
    var currentLocationRecord = iterLocation.next()
    while (iterTime <= endTime) {
      while (tsToUnixSeconds(currentLocationRecord.timestamp) < iterTime && iterLocation.hasNext) {
        lastLocationRecord = Some(currentLocationRecord)
        currentLocationRecord = iterLocation.next()
      }

      lastLocationRecord.foreach { llr =>
        val firstTimeSeconds = tsToUnixSeconds(llr.timestamp)
        val secondTimeSeconds = tsToUnixSeconds(currentLocationRecord.timestamp)
        val timeDeltaSeconds = secondTimeSeconds - firstTimeSeconds

        val pointDensity = math.min(1.0, incrementSeconds.toDouble / timeDeltaSeconds.toDouble)

        if (firstTimeSeconds <= iterTime && secondTimeSeconds >= iterTime &&
            timeDeltaSeconds < maxInterpolateGapSeconds) {
          val mix = (iterTime - firstTimeSeconds).toDouble / (secondTimeSeconds - firstTimeSeconds).toDouble

          val interpLat = currentLocationRecord.location.lat.value * mix +
              llr.location.lat.value * (1.0 - mix)
          val interpLon = currentLocationRecord.location.lon.value * mix +
              llr.location.lon.value * (1.0 - mix)

          val interpDistFromShore = currentLocationRecord.distanceToShore.value * mix +
              llr.distanceToShore.value * (1.0 - mix)

          interpolatedSeries.append(
            ResampledVesselLocation(new Instant(iterTime * 1000),
                                    LatLon(interpLat.of[degrees], interpLon.of[degrees]),
                                    interpDistFromShore.of[kilometer],
                                    pointDensity))
        }
      }

      iterTime += incrementSeconds
    }

    interpolatedSeries.toIndexedSeq
  }

  def calculateEncounters(
      minDurationForEncounter: Duration,
      input: SCollection[(VesselMetadata, Seq[ResampledVesselLocationWithAdjacency])],
      maxDistanceForEncounter : DoubleU[kilometer])
    : SCollection[VesselEncounters] = {

    input.flatMap {
      case (md, locationSeries) =>
        val encounters =
          mutable.Map.empty[(VesselMetadata, VesselMetadata), mutable.ArrayBuffer[SingleEncounter]]

        var currentEncounterVessel: Option[VesselMetadata] = None
        val currentRun = mutable.ArrayBuffer.empty[ResampledVesselLocationWithAdjacency]

        def tryAddEncounter(newEncounterVessel: Option[VesselMetadata]) = {
          if (currentEncounterVessel.isDefined && currentRun.size >= 2) {
            val startTime = currentRun.head.locationRecord.timestamp
            val endTime = currentRun.last.locationRecord.timestamp
            val encounterDuration = new Duration(startTime, endTime)

            if (encounterDuration.isLongerThan(minDurationForEncounter)) {
              val impliedSpeeds = currentRun
                .sliding(2)
                .map {
                  case Seq(first, second) =>
                    val dist =
                      first.locationRecord.location.getDistance(second.locationRecord.location)
                    val timeDelta =
                      new Duration(first.locationRecord.timestamp, second.locationRecord.timestamp)

                    dist.convert[meter].value / timeDelta.getStandardSeconds
                }
                .toSeq

              val medianDistance = currentRun.map { _.adjacency.closestNeighbour.get._2 }
                .medianBy(_.value)
              val meanLocation = LatLon.mean(currentRun.map(_.locationRecord.location))

              val medianSpeed =
                impliedSpeeds.medianBy(Predef.identity).of[meters_per_second].convert[knots]

              val vessel1Points = currentRun.map(_.locationRecord.pointDensity).sum.toInt
              val vessel2Points =
                currentRun.map(_.adjacency.closestNeighbour.get._3.pointDensity).sum.toInt

              val key = (md, currentEncounterVessel.get)
              if (!encounters.contains(key)) {
                encounters(key) = mutable.ArrayBuffer.empty[SingleEncounter]
              }
              encounters(key).append(
                SingleEncounter(startTime,
                                endTime,
                                meanLocation,
                                medianDistance,
                                medianSpeed,
                                vessel1Points,
                                vessel2Points))
            }
          }
          currentEncounterVessel = newEncounterVessel
          currentRun.clear()
        }

        locationSeries.foreach { l =>
          val possibleEncounterPoint =
            l.locationRecord.distanceToShore > Parameters.minDistanceToShoreForEncounter &&
              l.adjacency.closestNeighbour.isDefined &&
              l.adjacency.closestNeighbour.get._2 < maxDistanceForEncounter

          if (possibleEncounterPoint) {
            val closestNeighbour = l.adjacency.closestNeighbour.get._1
            if (currentEncounterVessel.isDefined && currentEncounterVessel.get.mmsi != closestNeighbour.mmsi) {
              tryAddEncounter(Some(closestNeighbour))
            }
            currentEncounterVessel = Some(closestNeighbour)
            currentRun.append(l)
          } else {
            tryAddEncounter(None)
          }

        }

        tryAddEncounter(None)

        encounters.map {
          case (key, encounters) =>
            VesselEncounters(key._1, key._2, encounters.toSeq)
        }.toIndexedSeq
    }
  }

  def annotateAdjacency(
      locations: SCollection[(VesselMetadata, ProcessedLocations)],
      adjacencies: SCollection[(VesselMetadata, Seq[ResampledVesselLocationWithAdjacency])])
    : SCollection[(VesselMetadata, Seq[VesselLocationRecordWithAdjacency])] = {
    locations.join(adjacencies).map {
      case (vessel, (processedLocations, resampled)) => {
        val resampledIter = resampled.iterator.buffered
        var current = resampledIter.next()

        val locationsWithAdjacency = processedLocations.locations.map { location =>
          while (resampledIter.hasNext
                 && abs(
                   new Duration(current.locationRecord.timestamp, location.timestamp).getMillis())
                   > abs(new Duration(resampledIter.head.locationRecord.timestamp,
                                      location.timestamp).getMillis())) {
            current = resampledIter.next()
          }
          VesselLocationRecordWithAdjacency(
            location,
            Adjacency(
              current.adjacency.numNeighbours,
              current.adjacency.closestNeighbour
            )
          )
        }
        (vessel, locationsWithAdjacency)
      }
    }
  }

  def calculateAdjacency(interpolateIncrementSeconds: Duration,
                         vesselSeries: SCollection[(VesselMetadata, Seq[VesselLocationRecord])],
                         maxEncounterRadius : DoubleU[kilometer])
    : SCollection[(VesselMetadata, Seq[ResampledVesselLocationWithAdjacency])] = {
    val s2Level = 12

    val resampled: SCollection[(VesselMetadata, Seq[ResampledVesselLocation])] = vesselSeries.map {
      case (md, locations) =>
        (md, resampleVesselSeries(interpolateIncrementSeconds, locations))
    }

    val groupedByTime = resampled.flatMap {
      case (md, locations) =>
        locations.map { l =>
          (l.timestamp, (md, l))
        }
    }.groupByKey

    val adjacencyAnnotated = groupedByTime.flatMap {
      case (time, vesselLocations) =>
        // First, segment vessels by S2 cell to decrease the computational overhead of the
        // distance calculations.
        val cellMap = mutable.Map
          .empty[S2CellId, mutable.ListBuffer[(VesselMetadata, ResampledVesselLocation)]]

        val vesselLocationMap = mutable.Map.empty[VesselMetadata, ResampledVesselLocation]
        vesselLocations.foreach {
          case (md, vl) =>
            val cells = vl.location.getCapCoveringCells(1.0.of[kilometer], s2Level)
            cells.foreach { cell =>
              if (!cellMap.contains(cell)) {
                cellMap(cell) = mutable.ListBuffer.empty[(VesselMetadata, ResampledVesselLocation)]
              }
              cellMap(cell).append((md, vl))
            }
            vesselLocationMap(md) = vl
        }

        // Second, calculate all pair-wise distances per vessel per cell. We include self-distances
        // because, although they're zero, it's simpler to keep them here to prevent filtering of
        // points w/out adjacent vessels.
        val vesselDistances = mutable.HashMap
          .empty[VesselMetadata, mutable.ListBuffer[(VesselMetadata, DoubleU[kilometer])]]
        cellMap.foreach {
          case (_, vessels) =>
            vessels.foreach {
              case (md1, vl1) =>
                vessels.foreach {
                  case (md2, vl2) =>
                    val distance = vl1.location.getDistance(vl2.location)

                    if (distance < maxEncounterRadius) {
                      if (!vesselDistances.contains(md1)) {
                        vesselDistances(md1) =
                          mutable.ListBuffer.empty[(VesselMetadata, DoubleU[kilometer])]
                      }
                      vesselDistances(md1).append((md2, distance))
                    }
                }
            }
        }

        vesselDistances.map {
          case (md, adjacencies) =>
            val vl = vesselLocationMap(md)
            val (identity, withoutIdentity) = adjacencies.partition(_._1 == md)
            val closestN =
              withoutIdentity.toSeq.distinct.sortBy(_._2).take(Parameters.maxClosestNeighbours)

            val closestNeighbour = closestN.headOption.map {
              case (md2, dist) => (md2, dist, vesselLocationMap(md2))
            }

            val number = closestN.size

            val res =
              (md, ResampledVesselLocationWithAdjacency(vl, Adjacency(number, closestNeighbour)))
            res
        }.toSeq
    }

    // Join by vessel and sort by time asc.
    adjacencyAnnotated.groupByKey.map {
      case (md, locations) =>
        (md, locations.toIndexedSeq.sortBy(_.locationRecord.timestamp.getMillis))
    }
  }
}
