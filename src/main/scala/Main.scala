import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.ModelType
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.slf4j.{Logger, LoggerFactory}

import java.io.{File, PrintWriter}
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.util.{Failure, Success, Try, Using}

object Main {
  val registry = Encodings.newDefaultEncodingRegistry
  val jtokkitEncoding = registry.getEncodingForModel(ModelType.GPT_4)

  val logger: Logger = LoggerFactory.getLogger(getClass)

  // Initialize the encoding registry and get the encoding for GPT-4

  def main(args: Array[String]): Unit = {
    logger.info("TokenizationApp started.")

    // Define input and output file paths
    val inputFilePath = "D:\\IdeaProjects\\ScalaRest\\src\\main\\resources\\test\\text\\tokenids.txt"


    val tokens: Array[Int] = Try(readTokens(inputFilePath)) match {
      case Success(arr) => arr
      case Failure(ex) =>
        println(s"Error reading tokens: ${ex.getMessage}")
        Array.empty[Int] // Return an empty array in case of failure
    }

   tokens.foreach(token => println(token))
  }

  def readTokens(filename: String): Array[Int] = {
    val source = Source.fromFile(filename)
    try {
      // Read lines, trim whitespace, filter out empty lines, and convert to Int
      source.getLines()
        .map(_.trim)               // Remove leading/trailing whitespace
        .filter(_.nonEmpty)       // Skip empty lines
        .map(_.toInt)             // Convert each token to Int
        .toArray                  // Collect into an Array[Int]
    } finally {
      source.close()
    }
  }



}