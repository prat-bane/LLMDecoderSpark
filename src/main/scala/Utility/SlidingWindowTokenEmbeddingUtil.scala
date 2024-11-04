package Utility

import Utility.ModelUtil.{buildModel, extractEmbeddings, trainModel}
import com.typesafe.config.{Config, ConfigFactory}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.LoggerFactory

object SlidingWindowTokenEmbeddingUtil {

  private val logger = LoggerFactory.getLogger(this.getClass)

  val config: Config = ConfigFactory.load()
  val windowSize: Int = config.getInt("data.windowSize")
  val stride: Int = config.getInt("data.stride")

  /**
   * Generates sliding windows and corresponding target elements from token IDs.
   *
   * @param tokens     Array of token IDs.
   * @param windowSize Size of each sliding window.
   * @param stride     Stride with which the window moves.
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
   * @param inputs  Array of input windows (arrays of token indices).
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
   * @param indexedInputs  Array of indexed input windows.
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


  def getAllTokenEmbeddings(tokens:Array[Int]):(Map[Int, INDArray],Map[Int,Int])={

    logger.debug("tokens "+tokens.mkString(", "))
    // Step 1: Generate sliding windows and targets
    val (inputs, targets) = generateSlidingWindows(tokens, windowSize, stride)

    logger.debug("Original Input Windows:")
    inputs.foreach(window => println(window.mkString("[", ", ", "]")))

    logger.debug("\nOriginal Target Array:")
    logger.debug(targets.mkString("[", ", ", "]"))

    // Step 2: Map tokens to indices and transform inputs and targets
    val (tokenToIndex, indexedInputs, indexedTargets) = mapTokensToIndices(inputs, targets)

    logger.debug("\nToken to Index Mapping:")
    tokenToIndex.toSeq.sortBy(_._2).foreach { case (token, index) =>
      logger.debug(s"$token : $index")
    }

    logger.debug("\nIndexed Input Windows:")
    indexedInputs.foreach(window => println(window.mkString("[", ", ", "]")))

    logger.debug("\nIndexed Target Array:")
    logger.debug(indexedTargets.mkString("[", ", ", "]"))

    // Step 3: Create training data as INDArrays
    val (inputINDArray, targetINDArray) = createTrainingData(indexedInputs, indexedTargets)

    logger.debug("\nInput INDArray Shape: " + inputINDArray.shape().mkString("x"))
    logger.debug("Input INDArray:")
    logger.debug(inputINDArray.toStringFull)

    logger.debug("\nTarget INDArray Shape: " + targetINDArray.shape().mkString("x"))
    logger.debug("Target INDArray:")
    logger.debug(targetINDArray.toStringFull)

    // Step 4: Build the model
    val vocabSize = tokenToIndex.size + 1 // +1 if using 0 as padding/index
    val embeddingSize = 50 // Example embedding size
    val model = buildModel(vocabSize, embeddingSize)

    // Step 5: Train the model
    trainModel(model, inputINDArray, targetINDArray)

    // Step 6: Extract embeddings
    val embeddings = extractEmbeddings(model)
    logger.info("\nExtracted Embeddings:")
    embeddings.foreach { case (index, vector) =>
      logger.info(s"Index $index: ${vector.toString()}")
    }

    (embeddings,tokenToIndex)
  }
}
