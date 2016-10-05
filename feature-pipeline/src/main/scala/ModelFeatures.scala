package org.skytruth.feature_pipeline

import io.github.karols.units._
import io.github.karols.units.SI._

import com.google.common.geometry.{S2, S2LatLng}
import com.spotify.scio.values.SCollection
import com.typesafe.scalalogging.{LazyLogging, Logger}
import org.joda.time.{DateTimeZone, LocalDateTime}
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
  import AdditionalUnits._
  import Utility._

   def buildSingleVesselFeatures(input: Seq[VesselLocationRecord]): Seq[Array[Double]] =
    input
      .sliding(3)
      .map {
        case Seq(p0, p1, p2) =>
          if (p0 == p1) {
            logger.fatal(s"p0 and p1 are the same: $p0, $p1")
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
              .getHourOfDay() * (localTime.getMinuteOfHour() / 60.0)) - 12.0) / 12.0
          val localMonthOfYearFeature = (localTime.getMonthOfYear() - 6.0) / 6.0

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
                                      math.log(1.0 + distanceToShoreKm))

          feature.foreach { v =>
            if (v.isNaN || v.isInfinite) {
              logger.fatal(s"Malformed feature: ${feature.toList}, $p0, $p1, $p2")
            }
          }

          feature
      }
      .toIndexedSeq

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

  def buildVesselFeatures(input: SCollection[(VesselMetadata, ProcessedLocations)])
    : SCollection[(VesselMetadata, SequenceExample)] =
    input.filter { case (metadata, pl) => pl.locations.size >= 3 }.map {
      case (metadata, processedLocations) =>
        val features = buildSingleVesselFeatures(processedLocations.locations)
        val featuresAsTFExample = buildTFExampleProto(metadata, features)
        (metadata, featuresAsTFExample)
    }
}