package org.skytruth.feature_pipeline

import com.spotify.scio.bigquery.TableRow
import com.spotify.scio.testing.PipelineSpec
import com.spotify.scio.io._
import io.github.karols.units._
import io.github.karols.units.SI._
import java.io.File
import org.joda.time.{Duration, Instant}
import org.scalatest._
import org.skytruth.anchorages._
import org.skytruth.common._
import org.skytruth.common.AdditionalUnits._
import scala.concurrent._
import ExecutionContext.Implicits.global
import org.apache.commons.lang3.builder.ToStringBuilder._
import scala.collection.{mutable, immutable}

import TestHelper._
import EncountersTestHelper._



class FeatureBuilderTests extends PipelineSpec with Matchers {
  val anchorageLocations = IndexedSeq(
    Anchorage.fromAnchoragePoints(Seq(AnchoragePoint(LatLon(-1.4068508.of[degrees], 55.2363158.of[degrees]),
        Set(VesselMetadata(1)), 0.0.of[kilometer], 0.0.of[kilometer]))),
    Anchorage.fromAnchoragePoints(Seq(AnchoragePoint(LatLon(-1.4686489.of[degrees], 55.2206029.of[degrees]),
        Set(VesselMetadata(1)), 0.0.of[kilometer], 0.0.of[kilometer]))),
    Anchorage.fromAnchoragePoints(Seq(AnchoragePoint(LatLon(-1.3983536.of[degrees], 55.2026308.of[degrees]), 
        Set(VesselMetadata(1)), 0.0.of[kilometer], 0.0.of[kilometer]))))

  val vesselPath = Seq(vlra("2011-07-01T00:00:00Z", -1.4065933, 55.2350923, speed = 1.0),
                       vlra("2011-07-01T00:05:00Z", -1.4218712, 55.2342113, speed = 1.0),
                       vlra("2011-07-01T00:10:00Z", -1.4467621, 55.2334282, speed = 1.0),
                       vlra("2011-07-01T00:15:00Z", -1.4623833, 55.2310789, speed = 1.0),
                       vlra("2011-07-01T00:20:00Z", -1.469593, 55.2287294, speed = 1.0),
                       vlra("2011-07-01T00:25:00Z", -1.471138, 55.2267713, speed = 1.0),
                       vlra("2011-07-01T00:30:00Z", -1.470623, 55.2236383, speed = 1.0),
                       vlra("2011-07-01T00:35:00Z", -1.4704514, 55.2206029, speed = 1.0),
                       vlra("2011-07-01T00:40:00Z", -1.4704514, 55.218057, speed = 1.0),
                       vlra("2011-07-01T00:45:00Z", -1.4704514, 55.215217, speed = 1.0),
                       vlra("2011-07-01T00:50:00Z", -1.4728546, 55.2116913, speed = 1.0),
                       vlra("2011-07-01T01:00:00Z", -1.4718246, 55.2088509, speed = 1.0),
                       vlra("2011-07-01T01:10:00Z", -1.4474487, 55.2057165, speed = 1.0),
                       vlra("2011-07-01T01:20:00Z", -1.4278793, 55.2040512, speed = 1.0),
                       vlra("2011-07-01T01:30:00Z", -1.4084816, 55.2036594, speed = 1.0),
                       vlra("2011-07-01T01:40:00Z", -1.3998985, 55.2037573, speed = 1.0))

  "Adjacency lookup" should "correctly return the nearest locations" in {
    val anchorageLookup =
        AdjacencyLookup(anchorageLocations, (v: Anchorage) => v.meanLocation, 0.5.of[kilometer], 12)
    val firstLoc = LatLon(-1.4065933.of[degrees], 55.2350923.of[degrees])
    val secondLoc = LatLon(-1.4704514.of[degrees], 55.2206029.of[degrees])
    val thirdLoc = LatLon(-1.3998985.of[degrees], 55.2037573.of[degrees])

    anchorageLookup.nearby(firstLoc).head._2 should equal(anchorageLocations(0))
    anchorageLookup.nearby(secondLoc).head._2 should equal(anchorageLocations(1))
    anchorageLookup.nearby(thirdLoc).head._2 should equal(anchorageLocations(2))
  }

  "The pipeline" should "correctly build single vessel features" in {
    val locations = Seq[VesselLocationRecord]()
    val anchorages = Seq[Anchorage]()

    val anchorageLookup =
        AdjacencyLookup(anchorageLocations, (v: Anchorage) => v.meanLocation, 0.1.of[kilometer], 13)

    val features = ModelFeatures.buildSingleVesselFeatures(vesselPath, anchorageLookup)

    val expectedFeatures = Vector(
      Array(1.3094787E9, 5.707110264748875, 0.9934654718911284, 0.4150483749018927, 0.005652584164039989, 0.0, -0.6875, 0.16666666666666666, 0.008321569619043047, 6.2166061010848646, 0.9878435975166979, 5.707110264748875, 0.0, 0.0, 2.0),
      Array(1.309479E9, 5.707110264748875, 1.3263776243488667, 0.4150483749018927, 0.009182286163263529, 0.0, -0.6805555555555555, 0.16666666666666666, -0.03749178728367031, 6.2166061010848646, 1.339924626974254, 6.398594934535208, 0.0, 0.0, 2.0),
      Array(1.3094793E9, 5.707110264748875, 1.0135685947460997, 0.4150483749018927, 0.005834335303245511, 0.0, -0.6736111111111112, 0.16666666666666666, -0.052745486205012515, 6.2166061010848646, 0.857027975842463, 6.398594934535208, 0.0, 0.0, 2.0),
      Array(1.3094796E9, 5.707110264748875, 0.6111889414513352, 0.4150483749018927, 0.0028047990885309423, 0.0, -0.6666666666666666, 0.16666666666666666, -0.18706674146856153, 6.2166061010848646, 0.6464942180906326, 5.707110264748875, 0.0, 0.0, 2.0),
      Array(1.3094799E9, 5.707110264748875, 0.24460385399507656, 0.4150483749018927, 9.2329126495125E-4, 0.0, -0.6597222222222222, 0.16666666666666666, -0.264563297490627, 6.2166061010848646, 0.5532866293775093, 0.0, 0.0, 0.0, 2.0),
      Array(1.3094802E9, 5.707110264748875, 0.3021130151441151, 0.4150483749018927, 0.001175023040790746, 0.0, -0.6527777777777778, 0.16666666666666666, 0.03389521907623849, 6.2166061010848646, 0.3380973113032052, 0.0, 0.0, 0.0, 2.0),
      Array(1.3094805E9, 5.707110264748875, 0.29097926079281866, 0.4150483749018927, 0.0011251562411156483, 0.0, -0.6458333333333334, 0.16666666666666666, 0.017982158365888096, 6.2166061010848646, 0.18257403911572168, 0.0, 0.0, 0.0, 2.0),
      Array(1.3094808E9, 5.707110264748875, 0.24906098287037828, 0.4150483749018927, 9.422901076859697E-4, 0.0, -0.638888888888889, 0.16666666666666666, 3.839159005547539E-7, 6.2166061010848646, 0.2975583666680227, 0.0, 0.0, 0.0, 2.0),
      Array(1.3094811E9, 5.707110264748875, 0.2742102796368779, 0.4150483749018927, 0.0010510853968555757, 0.0, -0.6319444444444444, 0.16666666666666666, 0.19048967267865144, 6.2166061010848646, 0.4891629205486719, 0.0, 0.0, 0.0, 2.0),
      Array(1.3094814E9, 5.707110264748875, 0.38801126768384675, 0.4150483749018927, 0.0015789075135279392, 0.0, -0.625, 0.16666666666666666, -0.30125528568059545, 6.2166061010848646, 0.7394341861932613, 5.707110264748875, 0.0, 0.0, 2.0),
      Array(1.309482E9, 6.398594934535208, 0.2894209655197388, 0.4150483749018927, 5.59266705260282E-4, 0.0, -0.611111111111111, 0.16666666666666666, -0.3485392903918195, 6.2166061010848646, 0.8554227710976902, 6.803505257608338, 0.0, 0.0, 2.0),
      Array(1.3094826E9, 6.398594934535208, 1.3166930096431666, 0.4150483749018927, 0.004541442630185502, 0.0, -0.5972222222222222, 0.16666666666666666, -0.013679731647585533, 6.2166061010848646, 1.3554110312101684, 7.313886831633462, 0.0, 0.0, 2.0),
      Array(1.3094832E9, 6.398594934535208, 1.1576691094610287, 0.4150483749018927, 0.0036309111727990608, 0.0, -0.5833333333333334, 0.16666666666666666, -0.020587084732611758, 6.2166061010848646, 1.4550829636395497, 7.0909098220799835, 0.0, 0.0, 2.0),
      Array(1.3094838E9, 6.398594934535208, 1.1493088896567933, 0.4150483749018927, 0.003586911009577457, 0.0, -0.5694444444444444, 0.16666666666666666, -0.010055884439657917, 6.2166061010848646, 0.7567138786556233, 6.398594934535208, 0.0, 0.0, 2.0))

    features should contain theSameElementsAs expectedFeatures
  }
}


class CountryCodeTests extends PipelineSpec with Matchers {
  "Country codes" should "be correctly parsed out from mmsis" in {
    CountryCodes.fromMmsi(0) should equal("-")
    CountryCodes.fromMmsi(10000) should equal("-")
    CountryCodes.fromMmsi(233000000) should equal("-")
    CountryCodes.fromMmsi(233453123) should equal("GB")
  }
}


