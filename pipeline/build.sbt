// Enable protocol buffer builds.
import sbtprotobuf.{ProtobufPlugin => PB}
PB.protobufSettings

// Project definitions for vessel classification pipeline and modelling.
scalafmtConfig in ThisBuild := Some(file(".scalafmt"))

val tfExampleProtoFiles =
  TaskKey[Seq[File]]("tf-example-protos", "Set of protos defining TF example.")

lazy val commonSettings = Seq(
  organization := "org.skytruth",
  version := "0.0.1",
  scalaVersion := "2.11.8",
  scalacOptions ++= Seq("-optimize"),
  resolvers ++= Seq(
    "Apache commons" at "https://repository.apache.org/snapshots"
  ),
  // Main project dependencies.
  libraryDependencies ++= Seq(
    "com.spotify" % "scio-core_2.11" % "0.2.6",
    "com.jsuereth" %% "scala-arm" % "1.4",
    "com.typesafe.scala-logging" %% "scala-logging" % "3.4.0",
    "io.github.karols" %% "units" % "0.2.1",
    "joda-time" % "joda-time" % "2.9.4",
    "org.apache.commons" % "commons-math3" % "3.4",
    "com.fasterxml.jackson.dataformat" % "jackson-dataformat-yaml" % "2.8.3",
    "com.fasterxml.jackson.module" % "jackson-module-scala_2.11" % "2.8.3"
  ),
  // Test dependencies.
  libraryDependencies ++= Seq(
    "ch.qos.logback" % "logback-classic" % "1.1.7",
    "com.spotify" % "scio-test_2.11" % "0.2.6" % "test",
    "org.scalactic" %% "scalactic" % "3.0.0" % "test",
    "org.scalatest" %% "scalatest" % "3.0.0" % "test"
  )
)

// TODO(alexwilson): Download these files from github TF repo rather than having our own copy.
lazy val tfExampleProtos = project
  .in(file("tf-example-protos"))
  .settings(commonSettings: _*)
  .settings(PB.protobufSettings: _*)
  .settings(
    Seq(
      includePaths in PB.protobufConfig += (sourceDirectory in PB.protobufConfig).value
    ))

lazy val common = project.in(file("common")).settings(commonSettings: _*)

// Pipeline for annotating AIS messages with other attributes (such as when
// fishing, port visits, transhipments or AIS gaps occur).
lazy val aisAnnotatorPipeline =
  project.in(file("ais-annotator")).settings(commonSettings: _*).dependsOn(common)

// The dataflow feature generation pipeline.
lazy val featurePipeline =
  project
    .in(file("feature-pipeline"))
    .settings(commonSettings: _*)
    .settings(
      Seq(
        libraryDependencies ++= Seq("com.opencsv" % "opencsv" % "3.7",
                                    "org.json4s" %% "json4s-native" % "3.3.0",
                                    "org.jgrapht" % "jgrapht-core" % "1.0.0")
      ))
    .dependsOn(common, tfExampleProtos)

// An aggregation of all projects.
lazy val root = (project in file(".")).aggregate(common, aisAnnotatorPipeline, featurePipeline)
