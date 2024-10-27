import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object Main {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder
      .appName("SlidingWindow")
      .master("local[*]") // Run locally using all cores
      .getOrCreate()

    // Step 2: Create an RDD of integers
    val numbers = Array(123, 21, 4000, 21, 1013, 3782, 693,32,5,53,79,14,11,7,90) // Example list of integers
    val numbersRDD = spark.sparkContext.parallelize(numbers)
    // Step 3: Calculate the sum and count of the elements
    val sum = numbersRDD.sum() // Sum of all elements
    val count = numbersRDD.count() // Count of elements

    // Step 4: Calculate the average
    val average = sum / count

    // Step 5: Print the result
    println(s"The average of the numbers is: $average")

    // Step 6: Stop the Spark session
    spark.stop();

  }

}
