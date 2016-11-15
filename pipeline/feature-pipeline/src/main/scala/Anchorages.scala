package org.skytruth.feature_pipeline

import io.github.karols.units._
import io.github.karols.units.SI._
import io.github.karols.units.defining._

import com.google.common.geometry.{S2CellId}
import com.spotify.scio._
import com.spotify.scio.values.SCollection

import org.jgrapht.alg.util.UnionFind
import org.joda.time.{Duration}

import org.skytruth.common.AdditionalUnits._
import org.skytruth.common.Implicits._
import org.skytruth.common.LatLon

import scala.collection.{mutable, immutable}
import scala.collection.JavaConverters._

object Anchorages {
  def findAnchoragePointCells(input: SCollection[(VesselMetadata, ProcessedLocations)],
                              knownFishingMMSIs: Set[Int]): SCollection[AnchoragePoint] = {

    input.flatMap {
      case (md, processedLocations) =>
        processedLocations.stationaryPeriods.map { pl =>
          val cell = pl.location.getS2CellId(Parameters.anchoragesS2Scale)
          (cell, (md, pl))
        }
    }.groupByKey.map {
      case (cell, visits) =>
        val centralPoint = LatLon.mean(visits.map(_._2.location))
        val uniqueVessels = visits.map(_._1).toIndexedSeq.distinct
        val fishingVesselCount = uniqueVessels.filter { md =>
          knownFishingMMSIs.contains(md.mmsi)
        }.size
        val meanDistanceToShore = visits.map { _._2.meanDistanceToShore }.mean

        AnchoragePoint(centralPoint, uniqueVessels, fishingVesselCount, meanDistanceToShore)
    }.filter { _.vessels.size >= Parameters.minUniqueVesselsForAnchorage }
  }

  def mergeAdjacentAnchoragePoints(anchoragePoints: Iterable[AnchoragePoint]): Seq[Anchorage] = {
    val anchoragesById =
      anchoragePoints.map(anchoragePoint => (anchoragePoint.id, anchoragePoint)).toMap

    // Merge adjacent anchorages.
    val unionFind = new UnionFind[AnchoragePoint](anchoragePoints.toSet.asJava)
    anchoragePoints.foreach { ancorage =>
      val neighbourCells = Array.fill[S2CellId](4) { new S2CellId() }
      ancorage.meanLocation
        .getS2CellId(Parameters.anchoragesS2Scale)
        .getEdgeNeighbors(neighbourCells)

      neighbourCells.flatMap { nc =>
        anchoragesById.get(nc.toToken)
      }.foreach { neighbour =>
        unionFind.union(ancorage, neighbour)
      }
    }

    // Build anchorage groups.
    anchoragePoints.groupBy { anchoragePoint =>
      unionFind.find(anchoragePoint).id
    }.map {
      case (_, anchoragePoints) =>
        Anchorage.fromAnchoragePoints(anchoragePoints)
    }.toSeq
  }

  def buildAnchoragesFromAnchoragePoints(
      anchorages: SCollection[AnchoragePoint]): SCollection[Anchorage] =
    anchorages
    // TODO(alexwilson): These three lines hackily group all anchorages on one mapper
    .map { a =>
      (0, a)
    }.groupByKey.map { case (_, anchorages) => anchorages }
    // Build anchorage group list.
    .flatMap { anchorages =>
      mergeAdjacentAnchoragePoints(anchorages)
    }

  def findAnchorageVisits(
      locationEvents: SCollection[(VesselMetadata, Seq[VesselLocationRecord])],
      anchorages: SCollection[Anchorage],
      minVisitDuration: Duration
  ): SCollection[(VesselMetadata, immutable.Seq[AnchorageVisit])] = {
    val si = anchorages.asListSideInput

    locationEvents
      .withSideInputs(si)
      .map {
        case ((metadata, locations), ctx) => {
          val anchoragePointToAnchorage = ctx(si).flatMap { ag =>
            ag.anchoragePoints.map { a =>
              (a.id, ag)
            }
          }.toMap
          val allPorts = ctx(si).flatMap { ag =>
            ag.anchoragePoints
          }
          val lookup =
            AdjacencyLookup(allPorts,
                            (anchorage: AnchoragePoint) => anchorage.meanLocation,
                            Parameters.anchorageVisitDistanceThreshold,
                            Parameters.anchoragesS2Scale)
          (metadata,
           locations
             .map((location) => {
               val anchoragePoints = lookup.nearby(location.location)
               if (anchoragePoints.length > 0) {
                 Some(
                   AnchorageVisit(anchoragePointToAnchorage(anchoragePoints.head._2.id),
                                  location.timestamp,
                                  location.timestamp))
               } else {
                 None
               }
             })
             .foldLeft(Vector[Option[AnchorageVisit]]())((res, visit) => {
               if (res.length == 0) {
                 res :+ visit
               } else {
                 (visit, res.last) match {
                   case (None, None) => res
                   case (None, Some(last)) => res :+ None
                   case (Some(visit), None) => res.init :+ Some(visit)
                   case (Some(visit), Some(last)) =>
                     res.init ++ last.extend(visit).map(visit => Some(visit))
                 }
               }
             })
             .filter(_.nonEmpty)
             .map(_.head)
             .filter(_.duration.isLongerThan(minVisitDuration))
             .toSeq)
        }
      }
      .toSCollection
  }
}
