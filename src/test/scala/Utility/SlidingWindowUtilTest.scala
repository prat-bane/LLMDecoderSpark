package Utility

import org.scalatest.FunSuite
import org.scalatest.Matchers
import org.scalatest.BeforeAndAfterAll
import org.apache.spark.sql.SparkSession
import org.apache.spark.rdd.RDD
import org.scalatest.Matchers.be

import java.nio.file.{Files, Paths}
import java.nio.charset.StandardCharsets
import scala.util.{Failure, Success, Try}

class SlidingWindowUtilTest extends FunSuite with Matchers with BeforeAndAfterAll {

  private var spark: SparkSession = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    spark = SparkSession.builder()
      .appName("SlidingWindowUtilTest")
      .master("local[*]")
      .getOrCreate()
  }

  override def afterAll(): Unit = {
    if (spark != null) {
      spark.stop()
    }
    super.afterAll()
  }

  /**
   * Test the readTokens method with a valid file containing integer token IDs.
   */
  test("readTokens should correctly read and parse token IDs from a valid file") {
    // Create a temporary file with valid token IDs
    val tempFile = Files.createTempFile("valid_tokens", ".txt")
    val tokenIDs = Seq(1, 2, 3, 4, 5)
    Files.write(tempFile, tokenIDs.map(_.toString).mkString("\n").getBytes(StandardCharsets.UTF_8))

    // Invoke the readTokens method
    val tokens = Try(SlidingWindowUtil.readTokens(tempFile.toString, spark)) match {
      case Success(arr) => arr
      case Failure(ex) =>
        fail(s"readTokens threw an exception for a valid file: ${ex.getMessage}")
    }

    // Verify the output
    tokens.length shouldBe tokenIDs.length
    tokens shouldBe tokenIDs.toArray

    // Clean up
    Files.deleteIfExists(tempFile)
  }

  /**
   * Test the readTokens method with an empty file.
   */
  test("readTokens should return an empty array for an empty file") {
    // Create an empty temporary file
    val tempFile = Files.createTempFile("empty_tokens", ".txt")

    // Invoke the readTokens method
    val tokens = Try(SlidingWindowUtil.readTokens(tempFile.toString, spark)) match {
      case Success(arr) => arr
      case Failure(ex) =>
        fail(s"readTokens threw an exception for an empty file: ${ex.getMessage}")
    }

    // Verify the output
    tokens shouldBe empty

    // Clean up
    Files.deleteIfExists(tempFile)
  }
}




