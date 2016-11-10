package org.skytruth.feature_pipeline

import io.github.karols.units._
import io.github.karols.units.SI._
import io.github.karols.units.defining._

import com.google.common.geometry.{S2CellId}
import com.spotify.scio._
import com.spotify.scio.values.SCollection

import org.jgrapht.alg.util.UnionFind

import scala.collection.{mutable, immutable}
import scala.collection.JavaConverters._

import AdditionalUnits._

object Anchorages {
  def findAnchorageCells(input: SCollection[(VesselMetadata, ProcessedLocations)],
                         knownFishingMMSIs: Set[Int]): SCollection[Anchorage] = {

    input.flatMap {
      case (md, processedLocations) =>
        processedLocations.stationaryPeriods.map { pl =>
          val cell = pl.location.getS2CellId(Parameters.portsS2Scale)
          (cell, (md, pl))
        }
    }.groupByKey.map {
      case (cell, visits) =>
        val centralPoint = LatLon.mean(visits.map(_._2.location))
        val uniqueVessels = visits.map(_._1).toIndexedSeq.distinct
        val fishingVesselCount = uniqueVessels.filter { md =>
          knownFishingMMSIs.contains(md.mmsi)
        }.size

        Anchorage(centralPoint, uniqueVessels, fishingVesselCount)
    }.filter { _.vessels.size >= Parameters.minUniqueVesselsForPort }
  }

  def mergeAdjacentAnchorages(anchorages: Iterable[Anchorage]): Seq[AnchorageGroup] = {
    val anchoragesById = anchorages.map(anchorage => (anchorage.id, anchorage)).toMap

    // Merge adjacent anchorages.
    val unionFind = new UnionFind[Anchorage](anchorages.toSet.asJava)
    anchorages.foreach { ancorage =>
      val neighbourCells = Array.fill[S2CellId](4) { new S2CellId() }
      ancorage.meanLocation.getS2CellId(Parameters.portsS2Scale).getEdgeNeighbors(neighbourCells)

      neighbourCells.flatMap { nc =>
        anchoragesById.get(nc.toToken)
      }.foreach { neighbour =>
        unionFind.union(ancorage, neighbour)
      }
    }

    // Build anchorage groups.
    anchorages.groupBy { anchorage =>
      unionFind.find(anchorage).id
    }.map {
      case (_, anchorages) =>
        AnchorageGroup.fromAnchorages(anchorages)
    }.toSeq
  }

  def buildAnchorageGroups(anchorages: SCollection[Anchorage]): SCollection[AnchorageGroup] =
    anchorages
    // TODO(alexwilson): These three lines hackily group all anchorages on one mapper
    .map { a =>
      (0, a)
    }.groupByKey.map { case (_, anchorages) => anchorages }
    // Build anchorage group list.
    .flatMap { anchorages =>
      mergeAdjacentAnchorages(anchorages)
    }

  def findAnchorageGroupVisits(
      locationEvents: SCollection[(VesselMetadata, Seq[VesselLocationRecord])],
      anchorageGroups: SCollection[AnchorageGroup]
  ): SCollection[(VesselMetadata, immutable.Seq[AnchorageGroupVisit])] = {
    val si = anchorageGroups.asListSideInput

    locationEvents
      .withSideInputs(si)
      .map {
        case ((metadata, locations), ctx) => {
          val anchorageToAnchorageGroup = ctx(si).flatMap { ag =>
            ag.anchorages.map { a =>
              (a.id, ag)
            }
          }.toMap
          val allPorts = ctx(si).flatMap { ag =>
            ag.anchorages
          }
          val lookup =
            AdjacencyLookup(allPorts,
                            (anchorage: Anchorage) => anchorage.meanLocation,
                            Parameters.anchorageVisitDistanceThreshold,
                            Parameters.portsS2Scale)
          (metadata,
           locations
             .map((location) => {
               val anchorages = lookup.nearby(location.location)
               if (anchorages.length > 0) {
                 Some(
                   AnchorageGroupVisit(anchorageToAnchorageGroup(anchorages.head._2.id),
                                       location.timestamp,
                                       location.timestamp))
               } else {
                 None
               }
             })
             .foldLeft(Vector[Option[AnchorageGroupVisit]]())((res, visit) => {
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
             .toSeq)
        }
      }
      .toSCollection
  }
}
