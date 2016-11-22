package org.skytruth.feature_pipeline

import io.github.karols.units._
import io.github.karols.units.SI._
import io.github.karols.units.defining._

import com.google.common.geometry.S2CellId
import com.typesafe.scalalogging.LazyLogging
import com.spotify.scio.values.SCollection
import org.joda.time.{Duration, Instant}
import org.skytruth.common.AdditionalUnits._
import org.skytruth.common.Implicits._
import org.skytruth.common.LatLon

import scala.collection.{mutable, immutable}
import scala.math._

import shapeless._
import ops.hlist._

object Encounters extends LazyLogging {

  def calculateEncounters[Annotations <: HList](
      minDurationForEncounter: Duration,
      input: SCollection[(VesselMetadata, Seq[VesselLocationRecord[Annotations]])])(
      implicit resamplingSelector: Selector[Annotations, Resampling],
      adjacencySelector: Selector[Annotations, Adjacency]): SCollection[VesselEncounters] = {

    input.flatMap {
      case (md, locationSeries) =>
        val encounters =
          mutable.Map.empty[(VesselMetadata, VesselMetadata), mutable.ArrayBuffer[SingleEncounter]]

        var currentEncounterVessel: Option[VesselMetadata] = None
        val currentRun = mutable.ArrayBuffer.empty[VesselLocationRecord[Annotations]]

        def tryAddEncounter(newEncounterVessel: Option[VesselMetadata])(
            implicit resamplingSelector: Selector[Annotations, Resampling],
            adjacencySelector: Selector[Annotations, Adjacency]) = {
          if (currentEncounterVessel.isDefined && currentRun.size >= 2) {
            val startTime = currentRun.head.timestamp
            val endTime = currentRun.last.timestamp
            val encounterDuration = new Duration(startTime, endTime)

            if (encounterDuration.isLongerThan(minDurationForEncounter)) {
              val impliedSpeeds = currentRun
                .sliding(2)
                .map {
                  case Seq(first, second) =>
                    val dist =
                      first.location.getDistance(second.location)
                    val timeDelta =
                      new Duration(first.timestamp, second.timestamp)

                    dist.convert[meter].value / timeDelta.getStandardSeconds
                }
                .toSeq

              val medianDistance = currentRun.map {
                _.annotation[Adjacency].closestNeighbour.get._2
              }.medianBy(_.value)
              val meanLocation = LatLon.mean(currentRun.map(_.location))

              val medianSpeed =
                impliedSpeeds.medianBy(Predef.identity).of[meters_per_second].convert[knots]

              val vessel1Points = currentRun.map(_.annotation[Resampling].pointDensity).sum.toInt
              val vessel2Points =
                currentRun
                  .map(
                    _.annotation[Adjacency].closestNeighbour.get._3
                      .annotation[Resampling]
                      .pointDensity)
                  .sum
                  .toInt

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
            l.distanceToShore > Parameters.minDistanceToShoreForEncounter &&
              l.annotation[Adjacency].closestNeighbour.isDefined &&
              l.annotation[Adjacency].closestNeighbour.get._2 < Parameters.maxDistanceForEncounter

          if (possibleEncounterPoint) {
            val closestNeighbour = l.annotation[Adjacency].closestNeighbour.get._1
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

  def annotateAdjacency[LocationAnnotations <: HList, AdjacencyAnnotations <: HList](
      locations: SCollection[(VesselMetadata, ProcessedLocations[LocationAnnotations])],
      adjacencies: SCollection[(VesselMetadata, Seq[VesselLocationRecord[AdjacencyAnnotations]])])(
      implicit adjacencySelector: Selector[AdjacencyAnnotations, Adjacency]) = {
    locations.join(adjacencies).map {
      case (vessel, (locations, resampled)) => {
        val resampledIter = resampled.iterator.buffered
        var current = resampledIter.next()
        (
          vessel,
          locations.copy(
            locations = locations.locations.map { location =>
              while (resampledIter.hasNext
                     && abs(new Duration(current.timestamp, location.timestamp).getMillis())
                       > abs(new Duration(resampledIter.head.timestamp, location.timestamp)
                         .getMillis())) {
                current = resampledIter.next()
              }

              location.addAnnotation(current.annotation[Adjacency])
            }
          )
        )
      }
    }
  }

  def calculateAdjacency[Annotations <: HList](
      interpolateIncrementSeconds: Duration,
      vesselSeries: SCollection[(VesselMetadata, Seq[VesselLocationRecord[Annotations]])])
    : SCollection[(VesselMetadata, Seq[VesselLocationRecord[Adjacency :: Resampling :: HNil]])] = {
    val s2Level = 12

    type ResampledLocation = VesselLocationRecord[Resampling :: HNil]

    val resampled = vesselSeries.map {
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
        val cellMap =
          mutable.Map.empty[S2CellId, mutable.ListBuffer[(VesselMetadata, ResampledLocation)]]

        val vesselLocationMap = mutable.Map.empty[VesselMetadata, ResampledLocation]
        vesselLocations.foreach {
          case (md, vl) =>
            val cells = vl.location.getCapCoveringCells(1.0.of[kilometer], s2Level)
            cells.foreach { cell =>
              if (!cellMap.contains(cell)) {
                cellMap(cell) = mutable.ListBuffer.empty[(VesselMetadata, ResampledLocation)]
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

            val closestNeighbour = closestN.headOption.map {
              case (md2, dist) => (md2, dist, vesselLocationMap(md2))
            }

            val number = closestN.size

            val res =
              (md, vl.addAnnotation(Adjacency(number, closestNeighbour)))

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
