ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.12.18"

lazy val root = (project in file("."))
  .settings(
    name := "LLMDecoderSpark"
  )

libraryDependencies ++= Seq(
  // https://mvnrepository.com/artifact/org.deeplearning4j/dl4j-spark
  "org.apache.spark" %% "spark-core" % "3.1.2",
  "org.apache.spark" %% "spark-sql"  % "3.1.2",
  "org.apache.spark" %% "spark-mllib" % "3.1.2" ,

  "org.deeplearning4j" %% "dl4j-spark" % "1.0.0-beta7",
  "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-beta7",
  "org.deeplearning4j" % "deeplearning4j-nlp" % "1.0.0-beta7",
  "org.nd4j" % "nd4j-native-platform" % "1.0.0-beta7",
  "org.slf4j" % "slf4j-simple" % "2.0.13",
  "com.typesafe" % "config" % "1.4.3",
  "org.mockito" %% "mockito-scala" % "1.17.14" % Test,
  "org.scalatest" %% "scalatest" % "3.2.17" % Test
)

resolvers += "Maven Central" at "https://repo1.maven.org/maven2/"
resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/"
