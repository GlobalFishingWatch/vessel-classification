// Project definitions for vessel classification pipeline and modelling.
scalafmtConfig in ThisBuild := Some(file(".scalafmt"))

lazy val commonSettings = Seq(
  organization := "org.skytruth",
  version := "0.0.1",
  scalaVersion := "2.11.8",
  // Main project dependencies.
  libraryDependencies ++= Seq(
    "com.spotify" % "scio-core_2.11" % "0.2.1",
    "com.typesafe.scala-logging" %% "scala-logging" % "3.4.0",
    "ch.qos.logback" % "logback-classic" % "1.1.7",
    "io.github.karols" %% "units" % "0.2.1",
    "joda-time" % "joda-time" % "2.9.4"
  ),
  // Test dependencies.
  libraryDependencies ++= Seq(
    "org.scalactic" %% "scalactic" % "3.0.0" % "test",
    "org.scalatest" %% "scalatest" % "3.0.0" % "test",
    "com.spotify" % "scio-test_2.11" % "0.2.1" % "test"
  )
)

// All common code for pipeline and modelling.
lazy val common = project.in(file("common")).settings(commonSettings: _*)

// The dataflow feature generation pipeline.
lazy val featurePipeline = project.in(file("feature-pipeline")).settings(commonSettings: _*)

// An aggregation of all projects.
lazy val root = (project in file(".")).aggregate(common, featurePipeline)
