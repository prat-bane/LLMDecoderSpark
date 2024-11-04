package Utility

import org.scalatest.{FlatSpec, Matchers}
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import SparkModelUtil._
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.nd4j.linalg.dataset.DataSet
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.conf.layers.{LSTM, OutputLayer}
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.lossfunctions.LossFunctions

import scala.collection.JavaConverters._
import org.nd4j.linalg.ops.transforms.Transforms

class SparkModelUtilTest extends FlatSpec with Matchers {

  "computePositionalEmbedding" should "compute correct positional embeddings" in {
    val windowSize = 2
    val embeddingSize = 4

    val positionalEmbedding = SparkModelUtil.computePositionalEmbedding(windowSize, embeddingSize)

    positionalEmbedding.rows() should be (windowSize)
    positionalEmbedding.columns() should be (embeddingSize)

    // Expected values for position 0
    val expectedRow0 = Nd4j.create(Array(0.0, 1.0, 0.0, 1.0))

    // Expected values for position 1
    val angle0 = 1.0 / Math.pow(10000.0, 0.0 / embeddingSize)
    val angle1 = 1.0 / Math.pow(10000.0, 2.0 / embeddingSize)
    val sinAngle0 = Math.sin(angle0)
    val cosAngle0 = Math.cos(angle0)
    val sinAngle1 = Math.sin(angle1)
    val cosAngle1 = Math.cos(angle1)
    val expectedRow1 = Nd4j.create(Array(sinAngle0, cosAngle0, sinAngle1, cosAngle1))

    // Validate position 0
    Transforms.abs(positionalEmbedding.getRow(0).sub(expectedRow0)).maxNumber().doubleValue() should be < 1e-6

    // Validate position 1
    Transforms.abs(positionalEmbedding.getRow(1).sub(expectedRow1)).maxNumber().doubleValue() should be < 1e-6
  }

  "tokenizeAndEmbed" should "convert tokens into embeddings correctly" in {
    Nd4j.setDefaultDataTypes(DataType.FLOAT, DataType.FLOAT)

    val tokens = Array(10, 99, 30)
    val tokenToIndex = Map(10 -> 0, 20 -> 1, 30 -> 2)
    val embeddingSize = 2
    val embeddingsMap = Map(
      0 -> Nd4j.create(Array(0.1f, 0.2f)),
      1 -> Nd4j.create(Array(0.3f, 0.4f)),
      2 -> Nd4j.create(Array(0.5f, 0.6f))
    )

    val result = SparkModelUtil.tokenizeAndEmbed(tokens, tokenToIndex, embeddingsMap)

    result.rows() should be (tokens.length)
    result.columns() should be (embeddingSize)

    // Expected embeddings
    val zeroVector = Nd4j.zeros(1, embeddingSize)
    val expectedEmbeddings = Nd4j.vstack(
      embeddingsMap(0).reshape(1, -1),
      zeroVector,
      embeddingsMap(2).reshape(1, -1)
    )

    // Validate embeddings
    Transforms.abs(result.sub(expectedEmbeddings)).maxNumber().doubleValue() should be < 1e-6
  }

  "getEmbeddingForTokenID" should "retrieve the correct embedding" in {
    val tokenID = 20
    val tokenToIndex = Map(10 -> 0, 20 -> 1, 30 -> 2)
    val embeddingSize = 2
    val embeddingsMap = Map(
      0 -> Nd4j.create(Array(0.1, 0.2)),
      1 -> Nd4j.create(Array(0.3, 0.4)),
      2 -> Nd4j.create(Array(0.5, 0.6))
    )

    val result = SparkModelUtil.getEmbeddingForTokenID(tokenID, tokenToIndex, embeddingsMap)
    val expectedEmbedding = embeddingsMap(1)

    Transforms.abs(result.sub(expectedEmbedding)).maxNumber().doubleValue() should be < 1e-6

    // Test for unknown tokenID
    val unknownTokenID = 99
    val resultUnknown = SparkModelUtil.getEmbeddingForTokenID(unknownTokenID, tokenToIndex, embeddingsMap)
    val zeroVector = Nd4j.zeros(1, embeddingSize)

    Transforms.abs(resultUnknown.sub(zeroVector)).maxNumber().doubleValue() should be < 1e-6
  }

  "buildModelForEmbeddingInputs" should "build a model with correct configuration" in {
    val embeddingSize = 2

    val model = SparkModelUtil.buildModelForEmbeddingInputs(embeddingSize)

    model.getnLayers should be (2)

    // Get LSTM layer configuration
    val lstmLayer = model.getLayer(0).conf().getLayer().asInstanceOf[LSTM]

    lstmLayer.getNIn should be (embeddingSize)
    lstmLayer.getActivationFn() should be (Activation.TANH.getActivationFunction())

    // Get Output layer configuration
    val outputLayer = model.getLayer(1).conf().getLayer().asInstanceOf[OutputLayer]

    outputLayer.getActivationFn() should be (Activation.IDENTITY.getActivationFunction())

    // Check that the output layer nIn matches lstmLayer nOut
    outputLayer.getNIn should be (lstmLayer.getNOut)
    outputLayer.getNOut should be (embeddingSize)


  }


  "trainModelWithIterator" should "train the model without errors" in {
    val embeddingSize = 2
    val model = SparkModelUtil.buildModelForEmbeddingInputs(embeddingSize)

    // Create dummy data
    val numSamples = 5
    val inputFeatureSize = embeddingSize
    val outputSize = embeddingSize

    val input = Nd4j.randn(numSamples, inputFeatureSize, 1)
    val labels = Nd4j.randn(numSamples, outputSize)

    val dataSet = new DataSet(input, labels)
    val dataSetIterator = new ListDataSetIterator(Seq(dataSet).asJava)

    // Adjust epochs for testing if necessary
    // Assuming you can set epochs in your SlidingWindowUtil for testing
    // SlidingWindowUtil.epochs = 1

    // Now train the model
    noException should be thrownBy SparkModelUtil.trainModelWithIterator(model, dataSetIterator)
  }
}
