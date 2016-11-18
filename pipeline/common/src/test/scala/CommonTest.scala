package org.skytruth.common

import org.scalatest._

object RecordsTest {
  import shapeless._
  import record._
  import syntax.singleton._

  val nameWitness = Witness("name")
  val numberWitness = Witness("number")

  case class ValueWrapper[Annotations <: HList](value: Int, annotations: Annotations)

  def addName[Annotations <: HList](wrapper: ValueWrapper[Annotations], name: String) = {
    ValueWrapper(wrapper.value, ("name" ->> name) :: wrapper.annotations)
  }

  def addNumber[Annotations <: HList](wrapper: ValueWrapper[Annotations], number: Int) = {
    ValueWrapper(wrapper.value, ("number" ->> number) :: wrapper.annotations)
  }

  def printNameAnnotation[Annotations <: HList](annotations: Annotations)(
      implicit sel: ops.record.Selector[Annotations, nameWitness.T]) {
    println(annotations("name"))
  }

  def printNumberAnnotation[Annotations <: HList](annotations: Annotations)(
      implicit sel: ops.record.Selector[Annotations, numberWitness.T]) {
    println(annotations("number"))
  }

  def mockPipeline() {
    val rawValue = ValueWrapper(23, HNil)

    val withName = addName(rawValue, "Foo")
    val withNameAndNumber = addNumber(withName, 16)

    val withNumber = addName(rawValue, "Foo")
    val withNumberAndName = addNumber(withNumber, 16)

    printNameAnnotation(withName.annotations)
    printNameAnnotation(withNameAndNumber.annotations)
    printNameAnnotation(withNumberAndName.annotations)

    printNumberAnnotation(withNameAndNumber.annotations)
  }
}

class CommonTest extends FlatSpec with Matchers {
  import Implicits._

  "Richer iterable" should "correctly support countBy" in {
    val input = Seq(1, 4, 3, 6, 4, 1, 1, 1, 3)

    input.countBy(x => x * 2) should contain allOf (2 -> 4, 6 -> 2, 8 -> 2, 12 -> 1)
  }

  "Richer iterable" should "correctly support medianBy" in {
    val input1 = Seq(1)
    input1.medianBy(Predef.identity) should equal(1)

    val input2 = Seq(1, 2, 3)
    input2.medianBy(Predef.identity) should equal(2)

    val input3 = Seq(1, 4, 3, 6, 4, 1, 4, 1, 4, 3)
    input3.medianBy(Predef.identity) should equal(4)
  }

  "Mock pipeline" should "do Scala type magic" in {
    RecordsTest.mockPipeline()
  }
}
