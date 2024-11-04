ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.12.17"

lazy val root = (project in file("."))
  .settings(
    name := "LLMDecoderSpark"
  )

assembly / assemblyMergeStrategy := {
  case PathList("META-INF", "LICENSE") => MergeStrategy.discard
  case PathList("META-INF", "License") => MergeStrategy.discard
  case PathList("META-INF", "LICENSE.txt") => MergeStrategy.discard
  case PathList("META-INF", "License.txt") => MergeStrategy.discard
  case PathList("META-INF", "license") => MergeStrategy.discard
  case PathList("META-INF", "license.txt") => MergeStrategy.discard
  case PathList("META-INF", xs @ _*) =>
    xs match {
      case "MANIFEST.MF" :: Nil => MergeStrategy.discard
      case "services" :: _ => MergeStrategy.concat
      case _ => MergeStrategy.discard
    }
  case "reference.conf" => MergeStrategy.concat
  case x if x.endsWith(".proto") => MergeStrategy.rename
  case x if x.contains("hadoop") => MergeStrategy.first
  case _ => MergeStrategy.first
}

fork := true

libraryDependencies ++= Seq(
  // https://mvnrepository.com/artifact/org.deeplearning4j/dl4j-spark
  "com.knuddels" % "jtokkit" % "1.1.0",
  "org.apache.spark" %% "spark-core" % "3.4.3",
  "org.apache.spark" %% "spark-sql"  % "3.4.3",
  "org.apache.spark" %% "spark-mllib" % "3.4.3" ,

  "org.deeplearning4j" %% "dl4j-spark" % "1.0.0-beta7",
  "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-beta7",
  "org.deeplearning4j" % "deeplearning4j-nlp" % "1.0.0-beta7",
  "org.nd4j" % "nd4j-native-platform" % "1.0.0-beta7",
  "org.slf4j" % "slf4j-simple" % "2.0.13",
  "com.typesafe" % "config" % "1.4.3",
  "org.mockito" %% "mockito-scala" % "1.17.14" % Test,
  "org.scalatest" %% "scalatest" % "3.0.8" % Test
)

resolvers += "Maven Central" at "https://repo1.maven.org/maven2/"
resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/"
