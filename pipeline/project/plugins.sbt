// Add scalafmt to allow formatting with 'sbt scalafmt'.
addSbtPlugin("com.geirsson" % "sbt-scalafmt" % "0.3.1")

// Add plugin to build Java from protocol buffers.
addSbtPlugin("com.github.gseitz" % "sbt-protobuf" % "0.5.3")

// Add plugin to support building fat jars.
addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "0.14.3")
