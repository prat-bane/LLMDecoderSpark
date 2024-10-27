package Utility

import Utility.SlidingWindowUtil.{batchSize, learningRate, logger, lstmLayerSize}
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, RNNFormat}
import org.deeplearning4j.nn.conf.layers.{EmbeddingSequenceLayer, LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory

object ModelUtil {

  private val logger = LoggerFactory.getLogger(this.getClass)

  // Configuration parameters (you can adjust these as needed)
  val learningRate: Double = 0.001
  val lstmLayerSize: Int = 128
  val batchSize: Int = 32
  val epochs: Int = 10
  /**
   * Builds the neural network model.
   *
   * @param vocabSize     Size of the vocabulary.
   * @param embeddingSize Size of the embedding vectors.
   * @return Initialized MultiLayerNetwork model.
   */
  def buildModel(vocabSize: Int, embeddingSize: Int): MultiLayerNetwork = {
    logger.info(s"Building model with vocabSize: $vocabSize, embeddingSize: $embeddingSize")
    val conf = new NeuralNetConfiguration.Builder()
      .updater(new Adam(learningRate))
      .list()
      .layer(0, new EmbeddingSequenceLayer.Builder()
        .nIn(vocabSize)
        .nOut(embeddingSize)
        .build())
      .layer(1, new LSTM.Builder()
        .nIn(embeddingSize)
        .nOut(lstmLayerSize)
        .activation(Activation.TANH)
        .build())
      .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
        .activation(Activation.SOFTMAX)
        .nIn(lstmLayerSize)
        .nOut(vocabSize)
        .dataFormat(RNNFormat.NCW)
        .build())
      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()
    logger.info("Model initialized successfully")
    model
  }

  /**
   * Trains the neural network model using the provided training data.
   *
   * @param model       The MultiLayerNetwork model to train.
   * @param inputArray  INDArray containing input data.
   * @param targetArray INDArray containing target data.
   */
  def trainModel(model: MultiLayerNetwork, inputArray: INDArray, targetArray: INDArray): Unit = {
    logger.info("Starting model training...")
    val dataSet = new DataSet(inputArray, targetArray)
    val dataSetIterator: DataSetIterator = new ListDataSetIterator(dataSet.asList(), batchSize) // Batch size from config
    model.setListeners(new ScoreIterationListener(1))

    (1 to epochs).foreach { epoch =>
      dataSetIterator.reset()
      model.fit(dataSetIterator)
      logger.info(s"Epoch $epoch completed")
    }

    logger.info("Model training completed")
  }

  /**
   * Extracts embeddings from the trained model.
   *
   * @param model The trained MultiLayerNetwork model.
   * @return Map of token indices to their corresponding embedding vectors.
   */
  def extractEmbeddings(model: MultiLayerNetwork): Map[Int, INDArray] = {
    val embeddingWeights = model.getLayer(0).getParam("W") // Shape: [vocabSize, embeddingSize]
    val vocabSize = embeddingWeights.rows()
    (1 until vocabSize).map { i =>
      val embeddingVector = embeddingWeights.getRow(i).dup()
      i -> embeddingVector
    }.toMap
  }


}
