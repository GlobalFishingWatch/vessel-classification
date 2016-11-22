package org.skytruth.feature_pipeline

import io.github.karols.units._
import io.github.karols.units.SI._

import com.google.common.geometry.{S2, S2LatLng}
import com.spotify.scio.values.SCollection
import com.typesafe.scalalogging.{LazyLogging, Logger}
import org.joda.time.{DateTimeZone, Duration, Instant, LocalDateTime}
import org.skytruth.common.AdditionalUnits._
import org.skytruth.common.Implicits._
import org.skytruth.common.{AdjacencyLookup, LatLon, ValueCache}
import org.tensorflow.example.{
  Example,
  Feature,
  Features,
  FeatureList,
  FeatureLists,
  SequenceExample,
  Int64List,
  FloatList,
  BytesList
}

object ModelFeatures extends LazyLogging {

  private case class BoundingAnchorage(startTime: Instant,
                                       endTime: Instant,
                                       startAnchorage: Anchorage,
                                       endAnchorage: Anchorage) {
    def minDistance(location: LatLon): DoubleU[kilometer] = {
      val d1 = startAnchorage.meanLocation.getDistance(location)
      val d2 = endAnchorage.meanLocation.getDistance(location)

      if (d1 < d2) d1 else d2
    }

    private def absDuration(time1: Instant, time2: Instant) =
      if (time2.isAfter(time1)) {
        new Duration(time1, time2)
      } else {
        new Duration(time2, time1)
      }

    def minDuration(timestamp: Instant): Duration = {
      val d1 = absDuration(startTime, timestamp)
      val d2 = absDuration(timestamp, endTime)
      if (d1.isShorterThan(d2)) d1 else d2
    }
  }

  def buildSingleVesselFeatures(
      input: Seq[VesselLocationRecordWithAdjacency],
      anchorageLookup: AdjacencyLookup[Anchorage]): Seq[Array[Double]] = {
    val boundingAnchoragesIterator = input.flatMap {
      case vlra => {
        val localAnchorages = anchorageLookup.nearby(vlra.location.location)
        localAnchorages.headOption.map { la =>
          (vlra.location.timestamp, la)
        }
      }
    }.sliding(2).filter(_.size == 2).map {
      case Seq((startTime, la1), (endTime, la2)) =>
        BoundingAnchorage(startTime, endTime, la1._2, la2._2)
    }

    var currentBoundingAnchorage: Option[BoundingAnchorage] = None
    input
      .sliding(3)
      .map {
        case Seq(VesselLocationRecordWithAdjacency(p0, a0),
                 VesselLocationRecordWithAdjacency(p1, a1),
                 VesselLocationRecordWithAdjacency(p2, a2)) =>
          if (p0 == p1) {
            logger.fatal(s"p0 and p1 are the same: $p0, $p1")
          }

          while (boundingAnchoragesIterator.hasNext && (currentBoundingAnchorage.isEmpty ||
                 !currentBoundingAnchorage.isEmpty && currentBoundingAnchorage.get.endTime
                   .isBefore(p1.timestamp))) {
            currentBoundingAnchorage = Some(boundingAnchoragesIterator.next)
          }

          val ll0 = S2LatLng.fromDegrees(p0.location.lat.value, p0.location.lon.value)
          val ll1 = S2LatLng.fromDegrees(p1.location.lat.value, p1.location.lon.value)
          val ll2 = S2LatLng.fromDegrees(p2.location.lat.value, p2.location.lon.value)
          val timestampSeconds = p1.timestamp.getMillis / 1000
          val timestampDeltaSeconds = p1.timestamp.getMillis / 1000 - p0.timestamp.getMillis / 1000
          val distanceDeltaMeters = p1.location.getDistance(p0.location).value
          val speedMps = p1.speed.convert[meters_per_second].value
          val integratedSpeedMps = distanceDeltaMeters / timestampDeltaSeconds
          val cogDeltaDegrees = Utility.angleNormalize((p1.course - p0.course)).value
          val integratedCogDeltaDegrees = S2
            .turnAngle(ll0.normalized().toPoint(),
                       ll1.normalized().toPoint(),
                       ll2.normalized().toPoint())
            .of[radians]
            .convert[degrees]
            .value
          val distanceToShoreKm = p1.distanceToShore.convert[kilometer].value

          // Calculate time features. TODO(alexwilson): Add tests.
          val longitudeTzOffsetSeconds = 60 * 60 * 12 * (p1.location.lon
              .convert[degrees]
              .value / 180.0)
          val offsetTimezone = DateTimeZone.forOffsetMillis(longitudeTzOffsetSeconds.toInt * 1000)
          val localTime = new LocalDateTime(timestampSeconds, offsetTimezone)
          val localTodFeature = ((localTime
              .getHourOfDay() + (localTime.getMinuteOfHour() / 60.0)) - 12.0) / 12.0
          val localMonthOfYearFeature = (localTime.getMonthOfYear() - 6.0) / 6.0

          val (distanceToBoundingAnchorageKm, timeToBoundingAnchorageS) =
            if (!currentBoundingAnchorage.isEmpty) {
              (currentBoundingAnchorage.get
                 .minDistance(p1.location)
                 .convert[kilometer]
                 .value
                 .toDouble,
               currentBoundingAnchorage.get.minDuration(p1.timestamp).getMillis.toDouble / 1000.0)
            } else {
              // TODO(alexwilson): These are probably not good values for when we don't have a bounding
              // anchorage. Tim: any suggestions?
              (0.0, 0.0)
            }

          // TODO(alexwilson): #neighbours, distance to closest neighbour, is_dark.

          // We include the absolute time not as a feature, but to make it easy
          // to binary-search for the start and end of time ranges when running
          // under TensorFlow.
          val feature = Array[Double](timestampSeconds,
                                      math.log(1.0 + timestampDeltaSeconds),
                                      math.log(1.0 + distanceDeltaMeters),
                                      math.log(1.0 + speedMps),
                                      math.log(1.0 + integratedSpeedMps),
                                      cogDeltaDegrees / 180.0,
                                      localTodFeature,
                                      localMonthOfYearFeature,
                                      integratedCogDeltaDegrees / 180.0,
                                      math.log(1.0 + distanceToShoreKm),
                                      math.log(1.0 + distanceToBoundingAnchorageKm),
                                      math.log(1.0 + timeToBoundingAnchorageS),
                                      /* We should probably add
                                         distance to closest neighbour
                                         here too - but what value
                                         should we use if one does not
                                         exist? */
                                      a0.numNeighbours)

          feature.foreach { v =>
            if (v.isNaN || v.isInfinite) {
              logger.fatal(s"Malformed feature: ${feature.toList}, $p0, $p1, $p2")
            }
          }

          feature
      }
      .toIndexedSeq
  }

  def buildTFExampleProto(metadata: VesselMetadata, data: Seq[Array[Double]]): SequenceExample = {
    val example = SequenceExample.newBuilder()
    val contextBuilder = example.getContextBuilder()

    // Add the mmsi, weight and vessel type to the builder
    contextBuilder.putFeature(
      "mmsi",
      Feature.newBuilder().setInt64List(Int64List.newBuilder().addValue(metadata.mmsi)).build())

    // Add all the sequence data as a feature.
    val sequenceData = FeatureList.newBuilder()
    data.foreach { datum =>
      val floatData = FloatList.newBuilder()
      datum.foreach { v =>
        floatData.addValue(v.toFloat)
      }
      sequenceData.addFeature(Feature.newBuilder().setFloatList(floatData.build()))
    }
    val featureLists = FeatureLists.newBuilder()
    featureLists.putFeatureList("movement_features", sequenceData.build())
    example.setFeatureLists(featureLists)

    example.build()
  }

  def buildVesselFeatures(
      input: SCollection[(VesselMetadata, ProcessedAdjacencyLocations)],
      anchorages: SCollection[Anchorage]): SCollection[(VesselMetadata, SequenceExample)] = {
    val siAnchorages = anchorages.asListSideInput

    val anchorageLookupCache = ValueCache[AdjacencyLookup[Anchorage]]()
    input
      .withSideInputs(siAnchorages)
      .filter {
        case ((metadata, processedLocations), _) => processedLocations.locations.size >= 3
      }
      .map {
        case ((metadata, processedLocations), s) =>
          val anchorageLookup = anchorageLookupCache.get { () =>
            AdjacencyLookup(s(siAnchorages),
                            (v: Anchorage) => v.meanLocation,
                            Parameters.anchorageVisitDistanceThreshold,
                            Parameters.anchoragesS2Scale)
          }
          val features = buildSingleVesselFeatures(processedLocations.locations, anchorageLookup)
          val featuresAsTFExample = buildTFExampleProto(metadata, features)
          (metadata, featuresAsTFExample)
      }
      .toSCollection
  }
}
