package org.skytruth.feature_pipeline

import io.github.karols.units._
import io.github.karols.units.SI._
import io.github.karols.units.defining._

import com.google.common.geometry.S2CellId
import com.typesafe.scalalogging.LazyLogging
import com.spotify.scio.values.SCollection
import org.joda.time.{Duration, Instant}

import scala.collection.{mutable, immutable}

import AdditionalUnits._

object Encounters extends LazyLogging {
  def calculateEncounters(
      minDurationForEncounter: Duration,
      input: SCollection[(VesselMetadata, Seq[ResampledVesselLocationWithAdjacency])])
    : SCollection[VesselEncounter] = {

    input.flatMap {
      case (md, locationSeries) =>
        val encounters = mutable.ArrayBuffer.empty[VesselEncounter]

        var currentEncounterVessel: Option[VesselMetadata] = None
        val currentRun = mutable.ArrayBuffer.empty[ResampledVesselLocationWithAdjacency]

        def tryAddEncounter(newEncounterVessel: Option[VesselMetadata]) = {
          if (currentEncounterVessel.isDefined) {
            val startTime = currentRun.head.timestamp
            val endTime = currentRun.last.timestamp
            val encounterDuration = new Duration(startTime, endTime)
            val medianDistance = currentRun.map { _.closestNeighbour.get._2 }
              .sortBy(Predef.identity)
              .toIndexedSeq
              .apply(currentRun.size / 2)
            if (encounterDuration.isLongerThan(minDurationForEncounter)) {
              val meanLocation = LatLon.mean(currentRun.map(_.location))
              encounters.append(
                VesselEncounter(md,
                                currentEncounterVessel.get,
                                startTime,
                                endTime,
                                meanLocation,
                                medianDistance))
            }
          }
          currentEncounterVessel = newEncounterVessel
          currentRun.clear()
        }

        locationSeries.foreach { l =>
          val possibleEncounterPoint =
            l.distanceToShore > Parameters.minDistanceToShoreForEncounter &&
              l.closestNeighbour.isDefined &&
              l.closestNeighbour.get._2 < Parameters.maxDistanceForEncounter

          if (possibleEncounterPoint) {
            val closestNeighbour = l.closestNeighbour.get._1
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

        encounters.toIndexedSeq
    }
  }

  def annotateAdjacency(interpolateIncrementSeconds: Duration,
                        vesselSeries: SCollection[(VesselMetadata, Seq[VesselLocationRecord])])
    : SCollection[(VesselMetadata, Seq[ResampledVesselLocationWithAdjacency])] = {
    val s2Level = 12

    val resampled: SCollection[(VesselMetadata, Seq[ResampledVesselLocation])] = vesselSeries.map {
      case (md, locations) =>
        (md, Utility.resampleVesselSeries(interpolateIncrementSeconds, locations))
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
            val cells = Utility.getCapCoveringCells(vl.location, 1.0.of[kilometer], s2Level)
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

                    if (distance < Parameters.maxEncounterRadius) {
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

            val closestNeighbour = if (closestN.isEmpty) {
              None
            } else {
              Some(closestN.head)
            }
            val number = closestN.size

            val res = (md,
                       ResampledVesselLocationWithAdjacency(vl.timestamp,
                                                            vl.location,
                                                            vl.distanceToShore,
                                                            number,
                                                            closestNeighbour))
            res
        }.toSeq
    }

    // Join by vessel and sort by time asc.
    adjacencyAnnotated.groupByKey.map {
      case (md, locations) =>
        (md, locations.toIndexedSeq.sortBy(_.timestamp.getMillis))
    }
  }
}
