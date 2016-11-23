package org.skytruth.anchorages

import io.github.karols.units._
import io.github.karols.units.SI._
import io.github.karols.units.defining._

import org.json4s._
import org.json4s.JsonDSL.WithDouble._
import org.json4s.native.JsonMethods._

import com.google.common.geometry.{S2CellId}
import com.spotify.scio._
import com.spotify.scio.values.SCollection

import org.jgrapht.alg.util.UnionFind
import org.joda.time.{Duration, Instant}

import org.skytruth.common._
import org.skytruth.common.Implicits._

import scala.collection.{mutable, immutable}
import scala.collection.JavaConverters._

object AnchorageParameters {
  // Around 1km^2
  val anchoragesS2Scale = 13
  val minUniqueVesselsForAnchorage = 20
  val anchorageVisitDistanceThreshold = 0.5.of[kilometer]
  val minAnchorageVisitDuration = Duration.standardMinutes(60)

}

case class AnchoragePoint(meanLocation: LatLon,
                          vessels: Set[VesselMetadata],
                          meanDistanceToShore: DoubleU[kilometer],
                          meanDriftRadius: DoubleU[kilometer]) {
  def toJson = {
    val flagStateDistribution = vessels.countBy(_.flagState).toSeq.sortBy(c => -c._2).map {
      case (name, count) =>
        ("name" -> name) ~
          ("count" -> count)
    }
    val knownFishingVesselCount = vessels.filter(_.isFishingVessel).size
    ("id" -> id) ~
      ("latitude" -> meanLocation.lat.value) ~
      ("longitude" -> meanLocation.lon.value) ~
      ("unique_vessel_count" -> vessels.size) ~
      ("known_fishing_vessel_count" -> knownFishingVesselCount) ~
      ("flag_state_distribution" -> flagStateDistribution) ~
      ("mmsis" -> vessels.map(_.mmsi)) ~
      ("mean_distance_to_shore_km" -> meanDistanceToShore.value) ~
      ("mean_drift_radius_km" -> meanDriftRadius.value)
  }

  def id: String =
    meanLocation.getS2CellId(AnchorageParameters.anchoragesS2Scale).toToken
}

case class Anchorage(meanLocation: LatLon,
                     anchoragePoints: Set[AnchoragePoint],
                     meanDistanceToShore: DoubleU[kilometer],
                     meanDriftRadius: DoubleU[kilometer]) {
  def id: String =
    meanLocation.getS2CellId(AnchorageParameters.anchoragesS2Scale).toToken

  def toJson = {
    val vessels = anchoragePoints.flatMap(_.vessels).toSet
    val flagStateDistribution = vessels.countBy(_.flagState).toSeq.sortBy(c => -c._2).map {
      case (name, count) =>
        ("name" -> name) ~
          ("count" -> count)
    }
    val knownFishingVesselCount = vessels.filter(_.isFishingVessel).size

    ("id" -> id) ~
      ("latitude" -> meanLocation.lat.value) ~
      ("longitude" -> meanLocation.lon.value) ~
      ("unique_vessel_count" -> vessels.size) ~
      ("known_fishing_vessel_count" -> knownFishingVesselCount) ~
      ("flag_state_distribution" -> flagStateDistribution) ~
      ("anchorage_points" -> anchoragePoints.toSeq.sortBy(_.id).map(_.id)) ~
      ("mean_distance_to_shore_km" -> meanDistanceToShore.value) ~
      ("mean_drift_radius_km" -> meanDriftRadius.value)
  }
}

object Anchorage {
  def fromAnchoragePoints(anchoragePoints: Iterable[AnchoragePoint]) = {
    val weights = anchoragePoints.map(_.vessels.size.toDouble)
    Anchorage(LatLon.weightedMean(anchoragePoints.map(_.meanLocation), weights),
              anchoragePoints.toSet,
              anchoragePoints.map(_.meanDistanceToShore).weightedMean(weights),
              // Averaging across multiple anchorage points for drift radius
              // is perhaps a little statistically dubious, but may still prove
              // useful to distinguish fixed vs drifting anchorage groups.
              anchoragePoints.map(_.meanDriftRadius).weightedMean(weights))
  }
}

case class AnchorageVisit(anchorage: Anchorage, arrival: Instant, departure: Instant) {
  def extend(other: AnchorageVisit): immutable.Seq[AnchorageVisit] = {
    if (anchorage eq other.anchorage) {
      Vector(AnchorageVisit(anchorage, arrival, other.departure))
    } else {
      Vector(this, other)
    }
  }

  def duration = new Duration(arrival, departure)

  def toJson =
    ("anchorage" -> anchorage.id) ~
      ("start_time" -> arrival.toString()) ~
      ("end_time" -> departure.toString())
}

object Anchorages {
  def findAnchoragePointCells(
      input: SCollection[(VesselMetadata, ProcessedLocations)]): SCollection[AnchoragePoint] = {

    input.flatMap {
      case (md, processedLocations) =>
        processedLocations.stationaryPeriods.map { pl =>
          val cell = pl.location.getS2CellId(AnchorageParameters.anchoragesS2Scale)
          (cell, (md, pl))
        }
    }.groupByKey.map {
      case (cell, visits) =>
        val centralPoint = LatLon.mean(visits.map(_._2.location))
        val uniqueVessels = visits.map(_._1).toIndexedSeq.distinct
        val meanDistanceToShore = visits.map { _._2.meanDistanceToShore }.mean
        val meanDriftRadius = visits.map { _._2.meanDriftRadius }.mean

        AnchoragePoint(centralPoint, uniqueVessels.toSet, meanDistanceToShore, meanDriftRadius)
    }.filter { _.vessels.size >= AnchorageParameters.minUniqueVesselsForAnchorage }
  }

  def mergeAdjacentAnchoragePoints(anchoragePoints: Iterable[AnchoragePoint]): Seq[Anchorage] = {
    val anchoragesById =
      anchoragePoints.map(anchoragePoint => (anchoragePoint.id, anchoragePoint)).toMap

    // Merge adjacent anchorages.
    val unionFind = new UnionFind[AnchoragePoint](anchoragePoints.toSet.asJava)
    anchoragePoints.foreach { ancorage =>
      val neighbourCells = Array.fill[S2CellId](4) { new S2CellId() }
      ancorage.meanLocation
        .getS2CellId(AnchorageParameters.anchoragesS2Scale)
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
    val anchoragePointIdToAnchorageCache = ValueCache[Map[String, Anchorage]]()
    val anchorageLookupCache = ValueCache[AdjacencyLookup[AnchoragePoint]]()

    locationEvents
      .withSideInputs(si)
      .map {
        case ((metadata, locations), ctx) => {
          val anchoragePointIdToAnchorage = anchoragePointIdToAnchorageCache.get { () =>
            ctx(si).flatMap { ag =>
              ag.anchoragePoints.map { a =>
                (a.id, ag)
              }
            }.toMap
          }

          val lookup = anchorageLookupCache.get { () =>
            AdjacencyLookup(ctx(si).flatMap(_.anchoragePoints),
                            (anchorage: AnchoragePoint) => anchorage.meanLocation,
                            AnchorageParameters.anchorageVisitDistanceThreshold,
                            AnchorageParameters.anchoragesS2Scale)
          }

          (metadata,
           locations
             .map((location) => {
               val anchoragePoints = lookup.nearby(location.location)
               if (anchoragePoints.length > 0) {
                 Some(
                   AnchorageVisit(anchoragePointIdToAnchorage(anchoragePoints.head._2.id),
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
