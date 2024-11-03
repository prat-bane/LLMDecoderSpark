package metrics

import org.apache.spark.api.java.JavaRDD
import org.apache.spark.broadcast.Broadcast
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet

import java.io.ByteArrayInputStream

object ModelMetricsCalculator {

  /**
   * Computes the accuracy of the model on the given validation dataset.
   *
   * @param modelBroadcast        Broadcast variable containing the serialized model.
   * @param validationData        The validation dataset as JavaRDD[DataSet].
   * @param embeddingsMapBroadcast Broadcast variable for embeddingsMap.
   * @param tokenToIndexBroadcast  Broadcast variable for tokenToIndex.
   * @return The accuracy as a Double.
   */
  def computeAccuracy(
                       modelBroadcast: Broadcast[Array[Byte]],
                       validationData: JavaRDD[DataSet],
                       embeddingsMapBroadcast: Broadcast[Map[Int, INDArray]],
                       tokenToIndexBroadcast: Broadcast[Map[Int, Int]]
                     ): Double = {
    // Use mapPartitions to reduce the number of times we deserialize the model
    val predictionsAndLabels = validationData.rdd.mapPartitions { iter =>
      // Deserialize the model from the broadcasted bytes
      val modelBytes = modelBroadcast.value
      val modelInputStream = new ByteArrayInputStream(modelBytes)
      val model = ModelSerializer.restoreMultiLayerNetwork(modelInputStream, false)

      val embeddingsMap = embeddingsMapBroadcast.value
      val tokenToIndex = tokenToIndexBroadcast.value
      val indexToToken = tokenToIndex.map(_.swap) // Map from index to token ID

      iter.map { ds =>
        val features = ds.getFeatures
        val trueEmbedding = ds.getLabels // Actual target embedding

        // Get the true token ID
        val trueTokenIDOption = getTrueTokenID(trueEmbedding.reshape(1, -1), embeddingsMap, indexToToken)
        if (trueTokenIDOption.isEmpty) {
          println("True token ID not found for the given embedding.")
        }

        // Get model prediction
        val predictedEmbedding = model.output(features)

        // Find the closest embedding in embeddingsMap
        val predictedTokenIDIndex = findClosestTokenID(predictedEmbedding, embeddingsMap)
        val predictedTokenID= indexToToken.getOrElse(predictedTokenIDIndex,-1)

        println(s"True Token ID: ${trueTokenIDOption.getOrElse(-1)}, Predicted Token ID: $predictedTokenID")

        // Compare predicted token ID to true token ID
        val isCorrect = trueTokenIDOption.contains(predictedTokenID)

        isCorrect
      }
    }

    // Compute accuracy
    val total = predictionsAndLabels.count()
    val correct = predictionsAndLabels.filter(identity).count()

    val accuracy = if (total > 0) correct.toDouble / total else 0.0

    accuracy
  }

  def getTrueTokenID(
                      trueEmbedding: INDArray,
                      embeddingsMap: Map[Int, INDArray],
                      indexToToken: Map[Int, Int],
                      threshold: Double = 1e-5
                    ): Option[Int] = {
    embeddingsMap.find { case (_, emb) =>
      val distance = trueEmbedding.distance2(emb)
      distance < threshold
    }.map { case (index, _) =>
      indexToToken.getOrElse(index, -1)
    }.filter(_ != -1)
  }

  def findClosestTokenID(
                          embedding: INDArray,
                          embeddingsMap: Map[Int, INDArray]
                        ): Int = {
    val embeddingVector = embedding.reshape(1, -1)
    var maxSimilarity = Double.MinValue
    var closestTokenID = -1
    val similarityThreshold =.05
    embeddingsMap.foreach { case (index, emb) =>
      val embVector = emb.reshape(1, -1)
      val similarity = cosineSimilarity(embeddingVector, embVector)
      if (similarity > maxSimilarity && similarity > similarityThreshold) {
        maxSimilarity = similarity
        closestTokenID = index
      }
    }

    closestTokenID
  }

  def cosineSimilarity(a: INDArray, b: INDArray): Double = {
    val dotProduct = a.mul(b).sumNumber().doubleValue()
    val normA = a.norm2Number().doubleValue()
    val normB = b.norm2Number().doubleValue()
    dotProduct / (normA * normB)
  }


}
