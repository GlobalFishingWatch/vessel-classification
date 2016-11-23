package org.skytruth.feature_pipeline

import io.github.karols.units._
import io.github.karols.units.SI._
import io.github.karols.units.defining._

import org.skytruth.common.AdditionalUnits._

import shapeless._
import ops.hlist._

sealed trait Annotation

final case class PointInfo(speed: DoubleU[knots],
                           course: DoubleU[degrees],
                           heading: DoubleU[degrees])
    extends Annotation

final case class Adjacency(
    numNeighbours: Int,
    closestNeighbour: Option[
      (VesselMetadata, DoubleU[kilometer], VesselLocationRecord[Resampling :: HNil])])
    extends Annotation

final case class Resampling(pointDensity: Double) extends Annotation