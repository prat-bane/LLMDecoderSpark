package Utility

import Utility.SlidingWindowUtil.{epochs, learningRate, logger, lstmLayerSize}
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{LSTM, OutputLayer}
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory

object SparkModelUtil {

  private val logger = LoggerFactory.getLogger(this.getClass)
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
      logger.debug("No embedding found for token")
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


}
