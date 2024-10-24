package Utility
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, RNNFormat}
import org.deeplearning4j.nn.conf.layers.{EmbeddingSequenceLayer, LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.{DataSet, SplitTestAndTrain}
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.learning.config.Adam
import org.slf4j.LoggerFactory

object SlidingWindowUtil {
  private val logger = LoggerFactory.getLogger(this.getClass)

  // Configuration parameters (you can adjust these as needed)
  val learningRate: Double = 0.001
  val lstmLayerSize: Int = 128
  val batchSize: Int = 32
  val epochs: Int = 10

  /**
   * Generates sliding windows and corresponding target elements from token IDs.
   *
   * @param tokens Array of token IDs.
   * @param windowSize Size of each sliding window.
   * @param stride Stride with which the window moves.
   * @return Tuple containing input windows and target array.
   */
  def generateSlidingWindows(
                              tokens: Array[Int],
                              windowSize: Int,
                              stride: Int
                            ): (Array[Array[Int]], Array[Int]) = {
    // Generate sliding windows with windowSize + 1 to include the target
    val windowsWithTarget = tokens
      .sliding(windowSize + 1, stride)
      .collect {
        case window if window.length == windowSize + 1 =>
          (window.take(windowSize).toArray, window.last)
      }
      .toArray

    // Separate the inputs and targets
    val input = windowsWithTarget.map(_._1)
    val target = windowsWithTarget.map(_._2)

    (input, target)
  }

  /**
   * Maps unique tokens to unique indices and transforms input and target arrays.
   *
   * @param inputs Array of input windows (arrays of token indices).
   * @param targets Array of target tokens.
   * @return Tuple containing the token-to-index map, indexed inputs, and indexed targets.
   */
  def mapTokensToIndices(inputs: Array[Array[Int]], targets: Array[Int]): (Map[Int, Int], Array[Array[Int]], Array[Int]) = {
    // Collect all unique tokens from inputs and targets
    val uniqueTokens = (inputs.flatten ++ targets).distinct

    // Assign unique indices starting from 1 based on first occurrence
    val tokenToIndexMap: Map[Int, Int] = uniqueTokens.zipWithIndex.map { case (token, idx) => (token, idx + 1) }.toMap

    // Replace tokens in inputs and targets with their corresponding indices
    val indexedInputs: Array[Array[Int]] = inputs.map(window => window.map(token => tokenToIndexMap(token)))
    val indexedTargets: Array[Int] = targets.map(target => tokenToIndexMap(target))

    (tokenToIndexMap, indexedInputs, indexedTargets)
  }

  /**
   * Creates INDArray objects for inputs and targets from indexed input and target arrays.
   *
   * @param indexedInputs Array of indexed input windows.
   * @param indexedTargets Array of indexed target elements.
   * @return Tuple containing input INDArray and target INDArray.
   */
  def createTrainingData(indexedInputs: Array[Array[Int]], indexedTargets: Array[Int]): (INDArray, INDArray) = {
    val numExamples = indexedInputs.length
    val windowSize = indexedInputs.head.length

    // Flatten the input array and convert to Float
    val inputData = indexedInputs.flatten.map(_.toFloat)

    // Create INDArray with shape [numExamples, 1, windowSize]
    val inputINDArray: INDArray = Nd4j.create(inputData, Array(numExamples, 1, windowSize))

    // Repeat each target across the sequence length (windowSize)
    val repeatedTargets = indexedTargets.flatMap(target => Array.fill(windowSize)(target.toFloat))

    // Create INDArray with shape [numExamples, 1, windowSize]
    val targetINDArray: INDArray = Nd4j.create(repeatedTargets, Array(numExamples, 1, windowSize))

    (inputINDArray, targetINDArray)
  }

  /**
   * Builds the neural network model.
   *
   * @param vocabSize Size of the vocabulary.
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
   * @param model The MultiLayerNetwork model to train.
   * @param inputArray INDArray containing input data.
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
    (0 until vocabSize).map { i =>
      val embeddingVector = embeddingWeights.getRow(i).dup()
      i -> embeddingVector
    }.toMap
  }

  /**
   * Main function demonstrating the complete workflow.
   */
  def main(args: Array[String]): Unit = {
    // Example token IDs array
    val tokens = Array(123, 21, 4000, 21, 1013, 3782, 693)

    // Define window size and stride
    val windowSize = 3
    val stride = 1

    // Step 1: Generate sliding windows and targets
    val (inputs, targets) = generateSlidingWindows(tokens, windowSize, stride)

    println("Original Input Windows:")
    inputs.foreach(window => println(window.mkString("[", ", ", "]")))

    println("\nOriginal Target Array:")
    println(targets.mkString("[", ", ", "]"))

    // Step 2: Map tokens to indices and transform inputs and targets
    val (tokenToIndex, indexedInputs, indexedTargets) = mapTokensToIndices(inputs, targets)

    println("\nToken to Index Mapping:")
    tokenToIndex.toSeq.sortBy(_._2).foreach { case (token, index) =>
      println(s"$token : $index")
    }

    println("\nIndexed Input Windows:")
    indexedInputs.foreach(window => println(window.mkString("[", ", ", "]")))

    println("\nIndexed Target Array:")
    println(indexedTargets.mkString("[", ", ", "]"))

    // Step 3: Create training data as INDArrays
    val (inputINDArray, targetINDArray) = createTrainingData(indexedInputs, indexedTargets)

    println("\nInput INDArray Shape: " + inputINDArray.shape().mkString("x"))
    println("Input INDArray:")
    println(inputINDArray)

    println("\nTarget INDArray Shape: " + targetINDArray.shape().mkString("x"))
    println("Target INDArray:")
    println(targetINDArray)

    // Step 4: Build the model
    val vocabSize = tokenToIndex.size + 1 // +1 if using 0 as padding/index
    val embeddingSize = 50 // Example embedding size
    val model = buildModel(vocabSize, embeddingSize)

    // Step 5: Train the model
    trainModel(model, inputINDArray, targetINDArray)

    // Step 6: Extract embeddings
    val embeddings = extractEmbeddings(model)
    println("\nExtracted Embeddings:")
    embeddings.foreach { case (index, vector) =>
      println(s"Index $index: ${vector.toString()}")
    }


  }
}
