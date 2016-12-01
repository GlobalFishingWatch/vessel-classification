package org.skytruth.anchorages

import io.github.karols.units._
import io.github.karols.units.SI._
import io.github.karols.units.defining._

import com.google.cloud.dataflow.sdk.options.{
  DataflowPipelineOptions,
  PipelineOptions,
  PipelineOptionsFactory
}
import com.google.cloud.dataflow.sdk.runners.{DataflowPipelineRunner}
import com.typesafe.scalalogging.{LazyLogging, Logger}

import org.json4s._
import org.json4s.JsonAST.JValue
import org.json4s.JsonDSL.WithDouble._
import org.json4s.native.JsonMethods._

import com.google.common.geometry.{S2CellId}
import com.spotify.scio._
import com.spotify.scio.values.SCollection

import java.io.File

import org.jgrapht.alg.util.UnionFind
import org.joda.time.{Duration, Instant}

import org.skytruth.common._
import org.skytruth.common.AdditionalUnits._
import org.skytruth.common.Implicits._
import org.skytruth.common.ScioContextResource._

import scala.collection.{mutable, immutable}
import scala.collection.JavaConverters._

import org.apache.commons.math3.ml.clustering.DBSCANClusterer
import org.apache.commons.math3.ml.clustering.DoublePoint
import org.apache.commons.math3.ml.distance.DistanceMeasure
import org.apache.commons.math3.ml.clustering.Clusterable

import resource._

object AnchorageParameters {
  // Around 1km^2
  val anchoragesS2Scale = 13
  val minUniqueVesselsForAnchorage = 20
  val anchorageVisitDistanceThreshold = 0.5.of[kilometer]
  val minAnchorageVisitDuration = Duration.standardMinutes(60)
  val stationaryPeriodMinDuration = Duration.standardHours(24)

}

case class VesselStationaryPeriod(metadata : VesselMetadata, visit : StationaryPeriod) extends Clusterable {
  def getPoint() : Array[Double] = {
    return Array(visit.location.lat.value, visit.location.lon.value)
  }

  def toJson() =
    ("metadata" -> metadata.toJson()) ~
      ("visit" -> visit.toJson())
}

object VesselStationaryPeriod {
  implicit val formats = DefaultFormats

  def fromJson(json: JValue) =
    new VesselStationaryPeriod(
      VesselMetadata.fromJson(json \ "metadata"),
      StationaryPeriod.fromJson(json \ "visit"))
}










case class AnchorageGridPoint(meanLocation: LatLon, vessels: Set[VesselMetadata], visits: Seq[VesselStationaryPeriod]) {
  def id: String = meanLocation.getS2CellId(AnchorageParameters.anchoragesS2Scale).toToken
}

object AnchorageGridPoint {
  implicit val formats = DefaultFormats

  def fromVisits(visits : Seq[VesselStationaryPeriod]) = {
    AnchorageGridPoint(
      LatLon.mean(visits.map(_.visit.location)),
      visits.map(_.metadata).toIndexedSeq.distinct.toSet,
      visits)
  }
}

case class AnchorageGridCluster(meanLocation: LatLon, anchoragePoints: Set[AnchorageGridPoint]) {
  def id: String = meanLocation.getS2CellId(AnchorageParameters.anchoragesS2Scale).toToken
}

object AnchorageGridCluster extends LazyLogging {
  implicit val formats = DefaultFormats

  def fromAnchorageGridPoints(points: Iterable[AnchorageGridPoint]) = {
    AnchorageGridCluster(
      LatLon.weightedMean(
        points.map(_.meanLocation),
        points.map(_.vessels.size.toDouble)),
      points.toSet)
  }

  def fromVisits(visits : Seq[VesselStationaryPeriod]) : AnchorageGridCluster = {
    val centralPoint = LatLon.mean(visits.map(_.visit.location))
    val uniqueVessels = visits.map(_.metadata).toIndexedSeq.distinct
    AnchorageGridCluster(centralPoint, Set[AnchorageGridPoint]())
  }
}
















case class AnchoragePoint(meanLocation: LatLon,
                          vessels: Set[VesselMetadata],
                          meanDistanceToShore: DoubleU[kilometer],
                          meanDriftRadius: DoubleU[kilometer],
                          visits: Seq[VesselStationaryPeriod]) {
  def toJson() = {
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
      ("vessels" -> vessels.map(_.toJson)) ~
      ("mean_distance_to_shore_km" -> meanDistanceToShore.value) ~
      ("mean_drift_radius_km" -> meanDriftRadius.value) ~
      ("visits" -> visits.map((visit) => visit.toJson()))
  }

  def id: String =
    meanLocation.getS2CellId(AnchorageParameters.anchoragesS2Scale).toToken
}

object AnchoragePoint {
  implicit val formats = DefaultFormats

  def fromVisits(visits : Seq[VesselStationaryPeriod]) : AnchoragePoint = {
    val centralPoint = LatLon.mean(visits.map(_.visit.location))
    val uniqueVessels = visits.map(_.metadata).toIndexedSeq.distinct
    val meanDistanceToShore = visits.map { _.visit.meanDistanceToShore }.mean
    val meanDriftRadius = visits.map { _.visit.meanDriftRadius }.mean

    AnchoragePoint(centralPoint, uniqueVessels.toSet, meanDistanceToShore, meanDriftRadius, visits)
  }

  def fromJson(json: JValue) = {
    val meanLocation = LatLon((json \ "latitude").extract[Double].of[degrees],
                              (json \ "longitude").extract[Double].of[degrees])
    val meanDistanceToShore = (json \ "mean_distance_to_shore_km").extract[Double].of[kilometer]
    val meanDriftRadius = (json \ "mean_drift_radius_km").extract[Double].of[kilometer]
    val vessels =
      (json \ "vessels").extract[List[JValue]].map(jv => VesselMetadata.fromJson(jv)).toSet
    val visits =
      (json \ "visits").extract[List[JValue]].map(jv => VesselStationaryPeriod.fromJson(jv)).toSeq

    AnchoragePoint(meanLocation, vessels, meanDistanceToShore, meanDriftRadius, visits)
  }
}

case class Anchorage(meanLocation: LatLon,
                     anchoragePoints: Set[AnchoragePoint],
                     meanDistanceToShore: DoubleU[kilometer],
                     meanDriftRadius: DoubleU[kilometer]) {
  def id: String =
    meanLocation.getS2CellId(AnchorageParameters.anchoragesS2Scale).toToken

  def toJson() = {
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
      ("anchorage_point_ids" -> anchoragePoints.map(_.id)) ~
      ("mean_distance_to_shore_km" -> meanDistanceToShore.value) ~
      ("mean_drift_radius_km" -> meanDriftRadius.value)
  }
}

object Anchorage extends LazyLogging {
  implicit val formats = DefaultFormats

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

  def fromVisits(visits : Seq[VesselStationaryPeriod]) : Anchorage = {
    val centralPoint = LatLon.mean(visits.map(_.visit.location))
    val uniqueVessels = visits.map(_.metadata).toIndexedSeq.distinct
    val meanDistanceToShore = visits.map { _.visit.meanDistanceToShore }.mean
    val meanDriftRadius = visits.map { _.visit.meanDriftRadius }.mean

    Anchorage(centralPoint, Set[AnchoragePoint](), meanDistanceToShore, meanDriftRadius)
  }

  def fromJson(json: JValue, anchoragePointMap: Map[String, AnchoragePoint]) = {
    val meanLocation = LatLon((json \ "latitude").extract[Double].of[degrees],
                              (json \ "longitude").extract[Double].of[degrees])
    val meanDistanceToShore = (json \ "mean_distance_to_shore_km").extract[Double].of[kilometer]
    val meanDriftRadius = (json \ "mean_drift_radius_km").extract[Double].of[kilometer]
    val anchoragePoints =
      (json \ "anchorage_point_ids").extract[List[String]].map(id => anchoragePointMap(id)).toSet
    Anchorage(meanLocation, anchoragePoints, meanDistanceToShore, meanDriftRadius)
  }

  def readAnchorages(gcs: GoogleCloudStorage, rootPath: String): Seq[Anchorage] = {
    val anchoragePointsPath = rootPath + "/anchorage_points"
    val anchoragesPath = rootPath + "/anchorages"

    def readAllToJson(rootPath: String) =
      gcs.list(rootPath).flatMap { p =>
        gcs.get(p).map(l => parse(l))
      }

    logger.info("Reading anchorage points.")
    val anchoragePointMap = readAllToJson(anchoragePointsPath)
      .map(AnchoragePoint.fromJson _)
      .map(ap => (ap.id, ap))
      .toMap

    logger.info(s"Read ${anchoragePointMap.size} anchorage points.")

    val anchorages =
      readAllToJson(anchoragesPath).map(json => Anchorage.fromJson(json, anchoragePointMap))
    logger.info(s"Read ${anchorages.size} anchorages.")
    anchorages.toSeq
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

  def toJson() =
    ("anchorage" -> anchorage.id) ~
      ("start_time" -> arrival.toString()) ~
      ("end_time" -> departure.toString())
}


object Anchorages extends LazyLogging {
  def getAnchoragesLookup(anchoragesRootPath: String) = {
    val gcs = GoogleCloudStorage()
    val anchorages = if (!anchoragesRootPath.isEmpty) {
      Anchorage.readAnchorages(gcs, anchoragesRootPath)
    } else {
      Seq.empty[Anchorage]
    }
    AdjacencyLookup(anchorages,
                    (anchorage: Anchorage) => anchorage.meanLocation,
                    AnchorageParameters.anchorageVisitDistanceThreshold,
                    AnchorageParameters.anchoragesS2Scale)
  }

  def findAnchoragePointCells(
      input: SCollection[(VesselMetadata, ProcessedLocations)]): SCollection[AnchorageGridPoint] = {

    input.flatMap {
      case (md, processedLocations) =>
        processedLocations.stationaryPeriods.map { pl =>
          val cell = pl.location.getS2CellId(AnchorageParameters.anchoragesS2Scale)
          (cell, VesselStationaryPeriod(md, pl))
        }
    }.groupByKey.map {
      case (cell, visits) =>
        AnchorageGridPoint.fromVisits(visits.toSeq)
    }.filter { _.vessels.size >= AnchorageParameters.minUniqueVesselsForAnchorage }
  }

  def mergeAdjacentAnchoragePoints(anchoragePoints: Iterable[AnchorageGridPoint]): Seq[AnchorageGridCluster] = {
    val anchoragesById =
      anchoragePoints.map(anchoragePoint => (anchoragePoint.id, anchoragePoint)).toMap

    // Merge adjacent anchorages.
    val unionFind = new UnionFind[AnchorageGridPoint](anchoragePoints.toSet.asJava)
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
        AnchorageGridCluster.fromAnchorageGridPoints(anchoragePoints)
    }.toSeq
  }

  def buildAnchoragesFromAnchoragePoints(
      anchorages: SCollection[AnchorageGridPoint]): SCollection[AnchorageGridCluster] =
    anchorages.groupAll
    // Build anchorage group list.
    .flatMap { anchorages =>
      mergeAdjacentAnchoragePoints(anchorages)
    }

  def findNearestAnchorage(point : AnchoragePoint, anchorages : Seq[Anchorage]) : (Anchorage, DoubleU[kilometer]) = {
    anchorages.map(
      (anchorage) => (anchorage, anchorage.meanLocation.getDistance(point.meanLocation))
    ).reduceLeft((a, b) => (a, b) match {
      case ((anchorage1, distance1), (anchorage2, distance2)) => {
        if (distance1 < distance2) {
          (anchorage1, distance1)
        } else {
          (anchorage2, distance2)
        }
      }
    })
  }

  def attachPointsToNearestAnchorage(points : Seq[AnchoragePoint], anchorages : Seq[Anchorage]) : Seq[Anchorage] = {    
    points.map(point => {
      val (anchorage, distance) = findNearestAnchorage(point, anchorages)
      (anchorage, point)
    }).groupBy({ case (anchorage, point) => anchorage })
    .toIterable
    .toSeq
    .map({ case (anchorage, points) =>
      anchorage.copy(anchoragePoints = points.map({ case (anchorage, point) => point}).toSet)
    })
  }

  def clusterAnchorage(tentativeAnchorage: AnchorageGridCluster) : Seq[Anchorage] = {
    val vesselPoints = tentativeAnchorage.anchoragePoints
      .flatMap(
        point => point.visits)

    val points = new DBSCANClusterer[VesselStationaryPeriod](
      AnchorageParameters.anchorageVisitDistanceThreshold.convert[nauticalMile].value,
      AnchorageParameters.minUniqueVesselsForAnchorage)
    .cluster(vesselPoints.toIterable.asJavaCollection)
    .asScala
    .map(cluster => {
      AnchoragePoint.fromVisits(cluster.getPoints().asScala)
    })

    val anchorages = new DBSCANClusterer[VesselStationaryPeriod](
      AnchorageParameters.anchorageVisitDistanceThreshold.convert[nauticalMile].value * 10,
      AnchorageParameters.minUniqueVesselsForAnchorage)
    .cluster(vesselPoints.toIterable.asJavaCollection)
    .asScala
    .map(cluster => {
      Anchorage.fromVisits(cluster.getPoints().asScala)
    })

    attachPointsToNearestAnchorage(points, anchorages)
  }

  def clusterAnchorages(tentativeAnchorages: SCollection[AnchorageGridCluster]) : SCollection[Anchorage] = {
    tentativeAnchorages.flatMap(clusterAnchorage)
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

  def main(argArray: Array[String]) {
    val (options, remaining_args) = ScioContext.parseArguments[DataflowPipelineOptions](argArray)

    val environment = remaining_args.required("env")
    val jobName = remaining_args.required("job-name")
    val dataYears = remaining_args.getOrElse("data-years", InputDataParameters.defaultYearsToRun)
    val dataFileGlob =
      remaining_args.getOrElse("data-file-glob", InputDataParameters.defaultDataFileGlob)

    val config = GcpConfig.makeConfig(environment, jobName)

    logger.info(s"Pipeline output path: ${config.pipelineOutputPath}")

    options.setRunner(classOf[DataflowPipelineRunner])
    options.setProject(config.projectId)
    options.setStagingLocation(config.dataflowStagingPath)

    managed(ScioContext(options)).acquireAndGet { sc =>
      // Read, filter and build location records. We build a set of matches for all
      // relevant years, as a single Cloud Dataflow text reader currently can't yet
      // handle the sheer volume of matching files.
      val matches = (dataYears.split(",")).map { year =>
        val path = InputDataParameters.measuresPathPattern(year, dataFileGlob)
        sc.textFile(path)
      }

      val knownFishingMMSIs = AISDataProcessing.loadFishingMMSIs()

      val minValidLocations = 200
      val locationRecords: SCollection[(VesselMetadata, Seq[VesselLocationRecord])] =
        AISDataProcessing
          .readJsonRecords(matches, knownFishingMMSIs, InputDataParameters.minRequiredPositions)

      val processed =
        AISDataProcessing.filterAndProcessVesselRecords(
          locationRecords,
          AnchorageParameters.stationaryPeriodMinDuration)

      val anchorages = clusterAnchorages(
        Anchorages.buildAnchoragesFromAnchoragePoints(
          Anchorages.findAnchoragePointCells(processed)))

      // Output anchorages points.
      val anchoragePointsPath = config.pipelineOutputPath + "/anchorage_points"
      anchorages
        .flatMap(anchorage => anchorage.anchoragePoints)
        .map { anchoragePoint =>
          compact(render(anchoragePoint.toJson))
        }
        .saveAsTextFile(anchoragePointsPath)

      // And anchorages.
      val anchoragesPath = config.pipelineOutputPath + "/anchorages"
      anchorages.map { anchorage =>
        compact(render(anchorage.toJson))
      }.saveAsTextFile(anchoragesPath)

      // And find anchorage visits.
      val anchorageVisitsPath = config.pipelineOutputPath + "/anchorage_visits"
      val anchorageVisits =
        Anchorages.findAnchorageVisits(locationRecords,
                                       anchorages,
                                       AnchorageParameters.minAnchorageVisitDuration)

      anchorageVisits.map {
        case (metadata, visits) =>
          compact(
            render(("mmsi" -> metadata.mmsi) ~
              ("visits" -> visits.map(_.toJson))))
      }.saveAsTextFile(anchorageVisitsPath)
    }
  }
}
