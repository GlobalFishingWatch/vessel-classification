package org.skytruth.feature_pipeline

import io.github.karols.units._
import io.github.karols.units.SI._
import io.github.karols.units.defining._

import com.spotify.scio.values.SCollection
import org.joda.time.{Instant}

import AdditionalUnits._

object Encounters {

  def annotateAdjacency(vesselSeries: SCollection[(VesselMetadata, Seq[VesselLocationRecord])])
    : SCollection[(VesselMetadata, Seq[ResampledVesselLocationWithAdjacency])] = {
    val resampled: SCollection[(VesselMetadata, Seq[ResampledVesselLocation])] = vesselSeries.map {
      case (md, locations) => (md, Utility.resampleVesselSeries(locations))
    }

    // Shard each vessel location by (timestamp, s2cell id) for relevant covering.
    val keyForShardedJoin = resampled.flatMap {
      case (md, locations) =>
        locations.flatMap { l =>
          val cellIds = Utility.getCapCoveringCells(l.location,
                                                    1.0.of[kilometer],
                                                    Parameters.levelForAdjacencySharding)

          cellIds.map { cid =>
            val key = (l.timestamp, cid)

            (key, (md, l))
          }
        }
    }

    val maxClosestNeighbours = 10

    // Join by cell and timestamp to find the top N adjacent vessels per vessel per timestamp per cell.
    val vesselAdjacency: SCollection[((Instant, VesselMetadata, ResampledVesselLocation),
                                      (VesselMetadata, DoubleU[kilometer]))] =
      keyForShardedJoin.groupByKey.flatMap {
        case ((timestamp, _), vesselsAndLocations) =>
          // Now we have all vessels and locations within the cell, do an N^2 comparison,
          // (where N is the number of vessels in this grid cell at this time point, so should
          // be at max a few thousand).

          // For each vessel, find the closest neighbours.
          val encounters = vesselsAndLocations.flatMap {
            case (md1, vl1) =>
              val closestEncounters = vesselsAndLocations.map {
                case (md2, vl2) if md1 != md2 =>
                  ((timestamp, md1, vl1), (md2, vl1.location.getDistance(vl2.location)))
              }.filter(_._2._2 < Parameters.maxEncounterRadius)
                .toSeq
                .sortBy(_._2._2.value)
                .take(maxClosestNeighbours)
                .toSeq

              closestEncounters
          }

          encounters
      }

    // Join by timestamp and first vessel to get the top N adjacent vessels per vessel per timestamp
    val topNPerVesselPerTimestamp = vesselAdjacency.groupByKey.map {
      case ((timestamp, md, vl), adjacencies) =>
        val closestN = adjacencies.toSeq.distinct.sortBy(_._2).take(maxClosestNeighbours)

        val closest = closestN.head._2
        val number = closestN.size

        (md, ResampledVesselLocationWithAdjacency(vl.timestamp, vl.location, closest, number))
    }

    // Join by vessel and sort by time asc.
    topNPerVesselPerTimestamp.groupByKey.map {
      case (md, locations) =>
        (md, locations.toIndexedSeq.sortBy(_.timestamp.getMillis))
    }
  }

}
