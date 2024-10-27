package Utility

import Utility.SlidingWindowTokenEmbeddingUtil.getAllTokenEmbeddings
import org.apache.spark.sql.SparkSession
import org.apache.spark.rdd.RDD
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{LSTM, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.learning.config.Adam
import org.slf4j.LoggerFactory

import scala.jdk.CollectionConverters._

object SlidingWindowUtil extends Serializable {

  private val logger = LoggerFactory.getLogger(this.getClass)

  // Configuration parameters (you can adjust these as needed)
  val learningRate: Double = 0.001
  val lstmLayerSize: Int = 128
  val batchSize: Int = 32
  val epochs: Int = 10

  /**
   * Case class representing windowed data with input tokens and target token.
   *
   * @param input  Array of input token IDs.
   * @param target Target token ID.
   */
  case class WindowedData(input: Array[Int], target: Int)

  /**
   * Computes sinusoidal positional embeddings for a given window size and embedding size.
   *
   * @param windowSize    The size of the sliding window (sequence length).
   * @param embeddingSize The size of the embedding vectors.
   * @return INDArray of positional embeddings with shape (windowSize, embeddingSize).
   */
  def computePositionalEmbedding(windowSize: Int, embeddingSize: Int): INDArray = {
    val positionalEncoding = Nd4j.zeros(windowSize, embeddingSize)

    for (pos <- 0 until windowSize) {
      for (i <- 0 until embeddingSize) {
        val angle = pos.toDouble / Math.pow(10000.0, 2.0 * (i / 2).toDouble / embeddingSize)
        if (i % 2 == 0) {
          positionalEncoding.putScalar(Array(pos, i), Math.sin(angle))
        } else {
          positionalEncoding.putScalar(Array(pos, i), Math.cos(angle))
        }
      }
    }

    positionalEncoding
  }

  /**
   * Converts an array of token IDs into their corresponding embedding vectors.
   *
   * @param tokens        Array of token IDs.
   * @param tokenToIndex  Map of token IDs to indices.
   * @param embeddingsMap Map of token indices to their corresponding embeddings.
   * @return INDArray of stacked embeddings with shape (windowSize, embeddingSize).
   */
  def tokenizeAndEmbed(
                        tokens: Array[Int],
                        tokenToIndex: Map[Int, Int],
                        embeddingsMap: Map[Int, INDArray]
                      ): INDArray = {
    // Retrieve embeddings for each token and stack them
    val embeddingsList = tokens.map { tokenID =>
      val index = tokenToIndex.getOrElse(tokenID, -1)
      if (index == -1) {
        // Handle unknown token by using a zero vector
        Nd4j.zeros(1, embeddingsMap.head._2.length().toInt)
      } else {
        embeddingsMap(index).reshape(1, -1)
      }
    }

    // Stack embeddings into an INDArray of shape (windowSize, embeddingSize)
    Nd4j.vstack(embeddingsList: _*)
  }

  /**
   * Retrieves the embedding for a given token ID.
   *
   * @param tokenID       The token ID.
   * @param tokenToIndex  Map of token IDs to indices.
   * @param embeddingsMap Map of token indices to their corresponding embeddings.
   * @return The embedding INDArray for the token.
   */
  def getEmbeddingForTokenID(
                              tokenID: Int,
                              tokenToIndex: Map[Int, Int],
                              embeddingsMap: Map[Int, INDArray]
                            ): INDArray = {
    val index = tokenToIndex.getOrElse(tokenID, -1)
    if (index == -1) {
      Nd4j.zeros(1, embeddingsMap.head._2.length().toInt) // Zero vector for unknown token
    } else {
      embeddingsMap(index)
    }
  }

  /**
   * Builds the neural network model for embedding inputs.
   *
   * @param embeddingSize Size of the embedding vectors.
   * @return Initialized MultiLayerNetwork model.
   */
  def buildModelForEmbeddingInputs(embeddingSize: Int): MultiLayerNetwork = {
    logger.info(s"Building model with embeddingSize: $embeddingSize")
    val conf = new NeuralNetConfiguration.Builder()
      .updater(new Adam(learningRate))
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .list()
      .layer(0, new LastTimeStep(
        new LSTM.Builder()
          .nIn(embeddingSize)
          .nOut(lstmLayerSize)
          .activation(Activation.TANH)
          .build()
      ))
      .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
        .activation(Activation.IDENTITY)
        .nIn(lstmLayerSize)
        .nOut(embeddingSize)
        .build())
      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()
    logger.info("Model initialized successfully")
    model
  }

  /**
   * Trains the neural network model using the provided DataSetIterator.
   *
   * @param model           The MultiLayerNetwork model to train.
   * @param dataSetIterator DataSetIterator containing the training data.
   */
  def trainModelWithIterator(model: MultiLayerNetwork, dataSetIterator: DataSetIterator): Unit = {
    logger.info("Starting model training with positional embeddings...")
    model.setListeners(new ScoreIterationListener(1))

    for (epoch <- 1 to epochs) {
      dataSetIterator.reset()
      model.fit(dataSetIterator)
      logger.info(s"Epoch $epoch completed")
    }

    logger.info("Model training completed")
  }


  /**
   * Main function demonstrating the complete workflow with Spark parallelization.
   */
  def main(args: Array[String]): Unit = {
    // Initialize SparkSession
    val spark = SparkSession.builder()
      .appName("Sliding Window DataSet Creation with Spark")
      .master("local[*]")
      .getOrCreate()

    // Example sentences (each sentence is a sequence of token IDs)
    val sentences: Array[Array[Int]] = Array(
      Array(123, 21, 4000, 21, 1013, 3782, 693),
      Array(456, 789, 1234, 5678, 91011, 1213, 1415, 1617)
      // Add more sentences as needed
    )

    // Generate embeddings and token-to-index mappings
    // Flatten all tokens to generate unique token IDs
    val allTokens = sentences.flatten
    val (embeddingsMap, tokenToIndex) = getAllTokenEmbeddings(allTokens)

    // Define window size and stride
    val windowSize = 3
    val stride = 1

    // Broadcast the embeddingsMap and tokenToIndex to all Spark executors
    val embeddingsBroadcast = spark.sparkContext.broadcast(embeddingsMap)
    val tokenToIndexBroadcast = spark.sparkContext.broadcast(tokenToIndex)

    // Parallelize the sentences RDD
    val sentencesRDD: RDD[Array[Int]] = spark.sparkContext.parallelize(sentences)

    // Generate sliding windows using flatMap
    val slidingWindowsRDD: RDD[WindowedData] = sentencesRDD.flatMap { tokens =>
      // Generate sliding windows for each sentence
      tokens.sliding(windowSize + 1, stride).collect {
        case window if window.length == windowSize + 1 =>
          val inputWindow = window.slice(0, windowSize).toArray
          val target = window(windowSize)
          println("Windowed data input "+inputWindow.mkString(", "))
          println("Windowed data target "+ target)
          WindowedData(inputWindow, target)
      }
    }

    // Map each WindowedData to a DataSet with input and target embeddings
    val dataSetsRDD: RDD[DataSet] = slidingWindowsRDD.map { windowedData =>
      val embeddingsMapLocal = embeddingsBroadcast.value
      val tokenToIndexLocal = tokenToIndexBroadcast.value

      // Convert input tokens to embeddings
      val inputEmbeddings = tokenizeAndEmbed(windowedData.input, tokenToIndexLocal, embeddingsMapLocal) // shape: (windowSize, embeddingSize)

      // Compute positional embeddings
      val embeddingSize = embeddingsMapLocal.head._2.length().toInt
      val positionalEmbeddings = computePositionalEmbedding(windowSize, embeddingSize) // shape: (windowSize, embeddingSize)

      // Add positional embeddings to word embeddings
      val positionAwareEmbedding = inputEmbeddings.add(positionalEmbeddings) // shape: (windowSize, embeddingSize)

      // Transpose to match DL4J's RNN data format (batchSize, features, timeSeriesLength)
      val inputTransposed = positionAwareEmbedding.transpose() // shape: (embeddingSize, windowSize)

      // Reshape to (1, embeddingSize, windowSize)
      val inputINDArray = inputTransposed.reshape(1, embeddingSize, windowSize)

      // Convert target token to embedding
      val targetEmbedding = getEmbeddingForTokenID(windowedData.target, tokenToIndexLocal, embeddingsMapLocal).reshape(1, embeddingSize)

      // Create DataSet
      new DataSet(inputINDArray, targetEmbedding)
    }

    // Collect the DataSets (you can also consider caching if the dataset is large)
    val dataSets: List[DataSet] = dataSetsRDD.collect().toList

    // Optional: Print out the shapes of inputs and targets for verification
    dataSets.foreach { ds =>
      println(s"Input shape: ${ds.getFeatures.shape().mkString("x")}, Target shape: ${ds.getLabels.shape().mkString("x")}")
      println(ds.getFeatures)
      println(ds.getLabels)
    }

    // Build the neural network model
    val embeddingSize = embeddingsMap.head._2.length().toInt
    val modelWithEmbeddings = buildModelForEmbeddingInputs(embeddingSize)

    // Create DataSetIterator
    val dataSetIterator: DataSetIterator = new ListDataSetIterator(dataSets.asJava, batchSize)

    // Train the model
    trainModelWithIterator(modelWithEmbeddings, dataSetIterator)

    // Stop Spark session
    spark.stop()
  }
}
