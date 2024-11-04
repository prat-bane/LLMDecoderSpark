package Utility

import Utility.SlidingWindowTokenEmbeddingUtil.getAllTokenEmbeddings
import Utility.SparkModelUtil.{WindowedData, buildModelForEmbeddingInputs, computePositionalEmbedding, getEmbeddingForTokenID, tokenizeAndEmbed}
import com.typesafe.config.{Config, ConfigFactory}
import metrics.{CustomSparkListener, EpochMetrics, GradientStatsListener}
import metrics.ModelMetricsCalculator.computeAccuracy
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.rdd.RDD
import org.deeplearning4j.nn.conf.layers.BaseLayer
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.learning.config.{Adam, Nesterovs, Sgd}
import org.slf4j.LoggerFactory

import java.io.{BufferedWriter, ByteArrayInputStream, ByteArrayOutputStream, File, FileWriter, OutputStream, OutputStreamWriter}
import java.lang.management.{ManagementFactory, OperatingSystemMXBean}
import java.net.URI
import scala.util.{Failure, Success, Try}


@SerialVersionUID(1L)
object SlidingWindowUtil extends Serializable {

  private val logger = LoggerFactory.getLogger(this.getClass)

  val config: Config = ConfigFactory.load()

  // Configuration parameters (you can adjust these as needed)
  val learningRate: Double = config.getDouble("training.learningRate")
  val lstmLayerSize: Int = config.getInt("model.lstmLayerSize")
  val batchSize: Int = config.getInt("training.batchSize")
  val epochs: Int = config.getInt("training.epochs")
  val windowSize: Int = config.getInt("data.windowSize")
  val stride: Int = config.getInt("data.stride")

  /**
   * Main function demonstrating the complete workflow with Spark parallelization.
   */
  def main(args: Array[String]): Unit = {

    val sparkConf = config.getConfig("spark")
    // Initialize SparkSession
    val spark = SparkSession.builder()
      //.master("local[*]")
      //.master("spark://192.168.0.101:7077")
      .appName(sparkConf.getString("appName"))
      .master(sparkConf.getString("master"))
     // .config("spark.jars", sparkConf.getString("jars"))
      //.config("spark.hadoop.fs.defaultFS", sparkConf.getString("hadoop.fs.defaultFS"))
      .config("spark.executor.memory", sparkConf.getString("executor.memory"))
     // .config("spark.local.dir", sparkConf.getString("local.dir"))
      //.config("spark.eventLog.dir", sparkConf.getString("eventLog.dir"))
      .config("spark.rdd.compress", sparkConf.getString("rdd.compress"))
      .config("spark.io.compression.codec", sparkConf.getString("io.compression.codec"))
      .getOrCreate()
    val sc = spark.sparkContext

    val customSparkListener = new CustomSparkListener()
    val gradientListener = new GradientStatsListener()
    sc.addSparkListener(customSparkListener)

    val inputFilePath = args(0) //tokenids.txt file path. It is present in src/main/resources/tokenids.txt


    val allTokens: Array[Int] = Try(readTokens(inputFilePath,spark)) match {
      case Success(arr) => arr
      case Failure(ex) =>
        println(s"Error reading tokens: ${ex.getMessage}")
        Array.empty[Int] // Return an empty array in case of failure
    }

    // Generate embeddings and token-to-index mappings
    logger.info("Size:"+allTokens.length)
    val (embeddingsMap, tokenToIndex) = getAllTokenEmbeddings(allTokens)

    // Broadcast the embeddingsMap and tokenToIndex to all Spark executors
    val embeddingsBroadcast = spark.sparkContext.broadcast(embeddingsMap)
    val tokenToIndexBroadcast = spark.sparkContext.broadcast(tokenToIndex)

    // Parallelize the sentences RDD
    val sentencesRDD: RDD[Array[Int]] = spark.sparkContext.parallelize(Seq(allTokens))

    // Generate sliding windows using flatMap
    val slidingWindowsRDD: RDD[WindowedData] = sentencesRDD.flatMap { tokens =>
      // Generate sliding windows for each sentence
      tokens.sliding(windowSize + 1, stride).collect {
        case window if window.length == windowSize + 1 =>
          val inputWindow = window.slice(0, windowSize).toArray
          val target = window(windowSize)
          logger.debug("Windowed data input "+inputWindow.mkString(", "))
          logger.debug("Windowed data target "+ target)
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
      logger.info(s"Input shape: ${ds.getFeatures.shape().mkString("x")}, Target shape: ${ds.getLabels.shape().mkString("x")}")
      logger.info(ds.getFeatures.toString())
      logger.info(ds.getLabels.toStringFull)
    }

    // Build the neural network model
    val embeddingSize = embeddingsMap.head._2.length().toInt
    val modelWithEmbeddings = buildModelForEmbeddingInputs(embeddingSize)

 /*   // Create DataSetIterator
    val dataSetIterator: DataSetIterator = new ListDataSetIterator(dataSets.asJava, batchSize)

    // Train the model
    trainModelWithIterator(modelWithEmbeddings, dataSetIterator)*/

    // Create the TrainingMaster
    val tm = new ParameterAveragingTrainingMaster.Builder(batchSize)
      .averagingFrequency(config.getInt("trainingMaster.avgFreq"))
      .batchSizePerWorker(batchSize)
      .workerPrefetchNumBatches(config.getInt("trainingMaster.workerPreFetchNumBatches"))
      .build()

    // Create the SparkDl4jMultiLayer
    val sparkModel = new SparkDl4jMultiLayer(spark.sparkContext, modelWithEmbeddings, tm)


    val metricsBuffer = scala.collection.mutable.ArrayBuffer[EpochMetrics]()

    val Array(trainingDataSetsRDD, validationDataSetsRDD) = dataSetsRDD.randomSplit(Array(0.8, 0.2))
    logger.info(s"Number of partitions in trainingDataSetsRDD: ${trainingDataSetsRDD.getNumPartitions}")
    logger.info(s"Number of partitions in validationDataSetsRDD: ${validationDataSetsRDD.getNumPartitions}")

    // Training loop with accuracy computation
    sparkModel.setListeners(new ScoreIterationListener(10),gradientListener)
    val modelPathStr: String = config.getString("paths.model")
    val csvPathStr: String = config.getString("paths.csv")

    //val csvHdfsPath: Path = new Path(csvPathStr)

    // Initialize FileSystem with correct HDFS URI
   /* val hdfsUri = config.getString("hdfs.uri") // Replace with your HDFS URI
    val hdfsConf = new Configuration()
    hdfsConf.set("fs.defaultFS", hdfsUri)
    val hdfs = FileSystem.get(new URI(hdfsUri), hdfsConf)*/

    // Adjust paths to local file system
    val csvLocalPath: String = csvPathStr
    val modelLocalPath: String = modelPathStr

    // Initialize the CSV file locally
    val bw = new BufferedWriter(new FileWriter(csvLocalPath))

    // Initialize the CSV file
    //val csvHdfsPath = new Path("hdfs://ip-172-31-16-56.us-east-2.compute.internal:8020/input/spark/training_metrics.csv")
   /* val hdfsOutputStream = hdfs.create(csvHdfsPath, true) buffered writer to write it on hdfs
    val bw = new BufferedWriter(new OutputStreamWriter(hdfsOutputStream))*/

    // Write header
    bw.write("Epoch,TimeStamp,TrainingLoss,TrainingAccuracy,ValidationAccuracy,meanGradientMagnitude,maxGradientMagnitude,minGradientMagnitude,gradientVariance,LearningRate,UsedMemoryMB,TotalMemoryMB,MaxMemoryMB,totalShuffleReadBytes,totalShuffleWriteBytes,maxTaskDuration,minTaskDuration,avgTaskDuration,failedTaskCount,processCpuLoad,systemCpuLoad\n")

    // Training loop with accuracy computation
    for (epoch <- 1 to epochs) {
      customSparkListener.reset()
      val epochStartTime = System.currentTimeMillis()
      logger.info(s"Starting epoch $epoch")

      // Fit the model
      sparkModel.fit(trainingDataSetsRDD)
      val epochEndTime = System.currentTimeMillis()
      val epochTimeMs = epochEndTime - epochStartTime
      logger.info(s"Completed epoch $epoch in ${epochTimeMs} ms")

      // Serialize the model and broadcast it
      val modelOutputStream = new ByteArrayOutputStream()
      ModelSerializer.writeModel(sparkModel.getNetwork, modelOutputStream, false)
      val modelBytes = modelOutputStream.toByteArray
      val modelBroadcast = spark.sparkContext.broadcast(modelBytes)

      // Evaluate on validation data
      val validationDataSetsJavaRDD: JavaRDD[DataSet] = validationDataSetsRDD.toJavaRDD()
      val validationAccuracy = computeAccuracy(modelBroadcast, validationDataSetsJavaRDD, embeddingsBroadcast, tokenToIndexBroadcast)
      logger.info(s"Epoch $epoch Validation Accuracy: $validationAccuracy")

      // Evaluate on training data
      val trainingDataSetsJavaRDD: JavaRDD[DataSet] = trainingDataSetsRDD.toJavaRDD()
      val trainingAccuracy = computeAccuracy(modelBroadcast, trainingDataSetsJavaRDD, embeddingsBroadcast, tokenToIndexBroadcast)
      logger.info(s"Epoch $epoch Training Accuracy: $trainingAccuracy")


      // Compute validation loss
      val trainingLoss = sparkModel.getScore
      logger.info(s"Epoch $epoch Validation Loss: $trainingLoss")

      val gradientStats = gradientListener.getGradientStats
      val meanGradientMagnitude = gradientStats.getOrElse("meanMagnitude",0.0)
      val maxGradientMagnitude = gradientStats.getOrElse("maxMagnitude",0.0)
      val minGradientMagnitude = gradientStats.getOrElse("minMagnitude",0.0)
      val gradientVariance = gradientStats.getOrElse("variance",0.0)


      // Get the learning rate
      val conf = sparkModel.getNetwork.getLayerWiseConfigurations
      val layerConf = conf.getConf(1)
      val layer = layerConf.getLayer

      val learningRate = layer match {
        case baseLayer: BaseLayer =>
          val iUpdater = baseLayer.getIUpdater
          iUpdater match {
            case adam: Adam => adam.getLearningRate
            case sgd: Sgd => sgd.getLearningRate
            case nesterovs: Nesterovs => nesterovs.getLearningRate
            case _ => Double.NaN
          }
        case _ => Double.NaN
      }
      logger.info(s"Current Learning Rate: $learningRate")



      // Log memory usage
      val runtime = Runtime.getRuntime
      val usedMemoryMB = (runtime.totalMemory - runtime.freeMemory) / (1024 * 1024)
      val totalMemoryMB = runtime.totalMemory / (1024 * 1024)
      val maxMemoryMB = runtime.maxMemory / (1024 * 1024)
      logger.info(s"Memory Usage - Used: $usedMemoryMB MB, Total: $totalMemoryMB MB, Max: $maxMemoryMB MB")

      // Collect metrics from customSparkListener
      val taskMetrics = customSparkListener.taskMetricsData
      val stageMetrics = customSparkListener.stageMetricsData

      // Compute total shuffle read/write
      val totalShuffleReadBytes = taskMetrics.map(_.shuffleReadBytes).sum
      val totalShuffleWriteBytes = taskMetrics.map(_.shuffleWriteBytes).sum

      logger.info(s"Epoch $epoch Total Shuffle Read Bytes: $totalShuffleReadBytes")
      logger.info(s"Epoch $epoch Total Shuffle Write Bytes: $totalShuffleWriteBytes")

      // Compute task durations
      val taskDurations = taskMetrics.map(_.duration)
      val maxTaskDuration = if (taskDurations.nonEmpty) taskDurations.max else 0L
      val minTaskDuration = if (taskDurations.nonEmpty) taskDurations.min else 0L
      val avgTaskDuration = if (taskDurations.nonEmpty) taskDurations.sum.toDouble / taskDurations.size else 0.0

      logger.info(s"Epoch $epoch Task Durations - Max: $maxTaskDuration ms, Min: $minTaskDuration ms, Avg: $avgTaskDuration ms")

      // Detect task skew
      val taskDurationThreshold = avgTaskDuration * 2 // For example
      val skewedTasks = taskMetrics.filter(_.duration > taskDurationThreshold)
      if (skewedTasks.nonEmpty) {
        logger.warn(s"Epoch $epoch Detected ${skewedTasks.size} skewed tasks")
      }

      // Log failed task count
      logger.info(s"Epoch $epoch Failed Task Count: ${customSparkListener.failedTaskCount}")

      // CPU utilization (from driver)
      val osBean = ManagementFactory.getPlatformMXBean(classOf[OperatingSystemMXBean])
      val processCpuLoad = osBean.getAvailableProcessors
      val systemCpuLoad = osBean.getSystemLoadAverage

      logger.info(s"Process CPU Load: $processCpuLoad%")
      logger.info(s"System CPU Load: $systemCpuLoad%")


      // Collect metrics
      val epochMetrics = EpochMetrics(
        epoch = epoch,
        timestamp = epochTimeMs,
        trainingLoss = trainingLoss,
        trainingAccuracy = trainingAccuracy,
        validationAccuracy = validationAccuracy,
        meanGradientMagnitude = meanGradientMagnitude,
        maxGradientMagnitude = maxGradientMagnitude,
        minGradientMagnitude = minGradientMagnitude,
        gradientVariance = gradientVariance,
        learningRate = learningRate,
        usedMemoryMB = usedMemoryMB,
        totalMemoryMB = totalMemoryMB,
        maxMemoryMB = maxMemoryMB,
        totalShuffleReadBytes = totalShuffleReadBytes,
        totalShuffleWriteBytes = totalShuffleWriteBytes,
        maxTaskDuration = maxTaskDuration,
        minTaskDuration = minTaskDuration,
        avgTaskDuration = avgTaskDuration,
        failedTaskCount = customSparkListener.failedTaskCount,
        processCpuLoad = processCpuLoad,
        systemCpuLoad = systemCpuLoad
      )
      metricsBuffer += epochMetrics

      // Write metrics to CSV
      bw.write(s"${epochMetrics.epoch},${epochMetrics.timestamp},${epochMetrics.trainingLoss},${epochMetrics.trainingAccuracy},${epochMetrics.validationAccuracy},${epochMetrics.meanGradientMagnitude},${epochMetrics.maxGradientMagnitude},${epochMetrics.minGradientMagnitude},${epochMetrics.gradientVariance},${epochMetrics.learningRate},${epochMetrics.usedMemoryMB},${epochMetrics.totalMemoryMB},${epochMetrics.maxMemoryMB},${epochMetrics.totalShuffleReadBytes},${epochMetrics.totalShuffleWriteBytes},${epochMetrics.maxTaskDuration},${epochMetrics.minTaskDuration},${epochMetrics.avgTaskDuration},${epochMetrics.failedTaskCount},${epochMetrics.processCpuLoad},${epochMetrics.systemCpuLoad}\n")
      bw.flush()
    }

    // Close the BufferedWriter
    bw.close()

    // Initialize FileSystem with correct HDFS URI
  //  val fs = FileSystem.get(new URI(hdfsUri), sc.hadoopConfiguration)

    // Define the HDFS path for the model
  /*  val modelPath: Path = new Path(modelPathStr)
  //  val modelPath = new Path("hdfs://ip-172-31-16-56.us-east-2.compute.internal:8020/input/spark/trainedModel.zip")

    // Create an OutputStream to HDFS
    val outputStream: OutputStream = fs.create(modelPath)*/

    // Save the model to HDFS using the OutputStream
    // Close the OutputStream
    //val modelpath="hdfs://localhost:9000/input/spark/model.zip"
    val modelFile = new File(modelLocalPath)
    ModelSerializer.writeModel(sparkModel.getNetwork, modelFile, true)
  /*  ModelSerializer.writeModel(sparkModel.getNetwork, outputStream, true)
    outputStream.close()*/
    // Delete temporary files and stop TrainingMaster
    tm.deleteTempFiles(spark.sparkContext)

    // Stop Spark session
    spark.stop()
  }


  def readTokens(filename: String, spark: SparkSession): Array[Int] = {
    val sc = spark.sparkContext
    val rawRDD = sc.textFile(filename)

    val rawLines = rawRDD.collect()
    logger.info("Raw lines:")
    rawLines.foreach(line => logger.info(s"'$line'"))
    logger.info(s"Total raw lines: ${rawLines.length}")

    val tokensRDD: RDD[Int] = rawRDD
      .map(_.trim)
      .filter(_.nonEmpty)
      .map(line => line.toInt)

    logger.info("File size " + rawRDD.count())
    logger.info(tokensRDD.collect().mkString(","))
    tokensRDD.collect()
  }
}






