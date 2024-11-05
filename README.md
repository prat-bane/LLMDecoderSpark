# Large Language Model Decoder on AWS EMR

### Author : Pratyay Banerjee
### Email : pbane8@uic.edu

### Youtube video : https://youtu.be/aQjFbMBpGCk

## Overview

The application reads token IDs from a text file, generates embeddings, and trains a neural network model using a sliding window approach. It leverages **Apache Spark** for distributed data processing and DL4J for deep learning tasks.

## Workflow

### 1. Sliding Window Operation on Tokens

**a. Token Embedding Generation**

- **Input**: Token IDs from `src/main/resources/tokenids.txt`.
- **Process**: Generate embeddings for each token ID and store them in a map for efficient retrieval.
- **Outcome**: A map linking each token ID to its corresponding embedding vector.

**b. Sliding Window RDD Dataset Formation**

- **Input**: Sequence of token IDs.
- **Process**:
  - Parallelize token IDs into an RDD using Spark.
  - Apply a sliding window of size `windowSize + 1` with a stride of `stride`.
  - For each window, designate the first `windowSize` tokens as input and the last token as the target.
- **Outcome**: An RDD containing input-target pairs ready for embedding and model training.

### 2. Positional Embedding Generation and Training Dataset Formation

**a. Positional Embedding Calculation**

- **Purpose**: Incorporate positional information to help the model understand the order of tokens.
- **Process**:
  - Compute positional embeddings for each token in the input window.
  - Add positional embeddings to the corresponding token embeddings to create position-aware embeddings.
- **Outcome**: Enhanced input embeddings that include both token semantics and positional context.

**b. Training Dataset Formation**

- **Input**: Position-aware embeddings and target token embeddings.
- **Process**:
  - Transpose and reshape the input embeddings to match the expected input shape for the LSTM model.
  - Pair the input embeddings with the target embeddings to form `DataSet` objects.
- **Outcome**: A training dataset comprising feature-target pairs suitable for model training.

### 3. Model Training

**a. Model Architecture**

- **Components**:
  - **LSTM Layer**: Captures sequential dependencies in the data.
  - **Output Layer**: Predicts the embedding of the target token.
- **Configuration**: Defined using `lstmLayerSize`, `learningRate`, and other hyperparameters from the configuration file.

**b. Distributed Training with Spark**

- **Process**:
  - Split the dataset into training and validation sets.
  - Configure `ParameterAveragingTrainingMaster` for distributed training.
  - Initialize `SparkDl4jMultiLayer` with the model and training master.
  - Iterate over the specified number of epochs:
    - Train the model on the training dataset.
    - Evaluate accuracy on both training and validation datasets.
    - Collect and log training metrics and system performance data.
- **Outcome**: A trained LSTM model capable of predicting the next token embedding based on input sequences.

### 4. Results

**a. Model Saving**

- **Process**: Serialize and save the trained model to the specified local path (`paths.model`).
- **Outcome**: A `.zip` file containing the trained model for future inference or analysis.

**b. Training Metrics File**

- **Process**: Log metrics such as loss, accuracy, gradient statistics, memory usage, and CPU load to a CSV file (`paths.csv`).
- **Outcome**: A comprehensive record of the training process, facilitating performance monitoring and debugging.


## Getting Started

#Youtube video Link: [https://youtu.be/QWJDm6ITcMg](https://youtu.be/aQjFbMBpGCk)

### Prerequisites

```bash
- Scala (version 2.12.17)
- Apache Spark (version 3.4.5)
- Apache Hadoop (version 3.3.4)
- SBT (Scala Build Tool, version 1.10.1)
- Java JDK (version 1.8 )
```
### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/prat-bane/LLMDecoderSpark
   cd project-directory
   ```
### Running Tests
```
sbt clean compile test
```

### Running the spark job

1) Run the SlidingWindowUtil class. It takes one argument which is the path to the tokenids.txt file which is present in src/main/resources/tokenids.txt
   

#### Configuration file
```
training {
  learningRate = 0.001
  batchSize = 32
  epochs = 10
}

model {
  lstmLayerSize = 64
}

spark {
  appName = "Spark LLM"
  master = "local[*]"
  jars = "s3://sparkllmbucket/jar/LLMDecoderSpark-assembly-0.1.0-SNAPSHOT.jar"
  hadoop {
    fs {
      defaultFS = "hdfs://localhost:9000"
    }
  }
  executor {
    memory = "4g"
  }
  local {
    dir = "hdfs://localhost:9000/spark/tmp"
  }
  eventLog {
    dir = "hdfs://localhost:9000/spark/events"
  }
  rdd {
    compress = "true"
  }
  io {
    compression {
      codec = "lz4"
    }
  }
}

hdfs {
  uri = "hdfs://localhost:9000/"
}

paths {
  model = "src/main/resources/trainedModel.zip"
  csv = "src/main/resources/training_metrics.csv"
}

data {
  windowSize = 64
  stride = 32
}

metrics {
  similarityThreshold = 0.05
}

trainingMaster{
   avgFreq=5
   workerPreFetchNumBatches=2
}

```

### General Configuration

## Configuration Parameters

| **Variable**                   | **Default Value**                                | **Description**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
|--------------------------------|--------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `window-size`                  | `64`                                             | **Purpose:** Defines the size of the sliding window used during data preparation. <br> **Impact:** Determines how many tokens are considered together, affecting the model's ability to learn from sequential data. A larger window size captures more context, allowing the model to learn longer-term dependencies, but it increases computational load and memory usage.                                                                                                                |
| `stride`                       | `32`                                             | **Purpose:** Specifies the step size by which the sliding window moves across the dataset. <br> **Impact:** Controls the overlap between consecutive windows. Smaller strides result in more overlapping windows, increasing the number of training samples and potentially improving the model's learning but also increasing processing time and resource consumption.                                                                                |
| `lstm-layer-size`              | `64`                                             | **Purpose:** Determines the number of units (neurons) in the LSTM layer of the neural network. <br> **Impact:** Affects the model's capacity to learn complex patterns and dependencies in the data. Larger sizes enhance the model's expressive power but increase the risk of overfitting and require more computational resources for training and inference.                                                                                                                             |
| `learning-rate`                | `0.001`                                          | **Purpose:** Controls the step size for updating model parameters during training. <br> **Impact:** Balances convergence speed and training stability. A learning rate that's too high can cause the model to overshoot minima, leading to divergence, while too low a rate can result in slow convergence and getting stuck in suboptimal solutions. Finding an optimal learning rate is essential for effective training.                                                        |
| `epochs`                       | `10`                                             | **Purpose:** Specifies the number of complete passes through the entire training dataset. <br> **Impact:** More epochs allow the model to learn more from the data, potentially improving accuracy. However, excessive epochs can lead to overfitting, where the model learns the training data too well and performs poorly on unseen data. It's important to monitor performance on a validation set to determine the appropriate number of epochs.                                         |
| `batch-size`                   | `32`                                             | **Purpose:** Determines the number of samples processed before updating the model's internal parameters. <br> **Impact:** Affects training stability and computational efficiency. Larger batch sizes can make better use of parallel hardware and lead to faster training times but require more memory. Smaller batch sizes provide more frequent updates but can result in noisier gradient estimates.                                               |
| `embedding-size`               | *(Defined elsewhere)*                            | **Purpose:** Sets the dimensionality of the vector embeddings generated for each token. <br> **Impact:** Higher embedding sizes capture more nuanced semantic relationships but demand more memory and computational power. Balances representation richness with resource utilization. Although not specified in `application.conf`, this parameter is crucial for defining the embedding layer's output size.                                         |
| `spark.appName`                | `"Spark LLM"`                                    | **Purpose:** Specifies the name of the Spark application. <br> **Impact:** Used for identification in the Spark UI and logs, helping in monitoring and debugging. A meaningful application name makes it easier to track and manage multiple jobs in a cluster environment.                                                                                                                                                                  |
| `spark.master`                 | `"yarn"`                                         | **Purpose:** Defines the master URL for the Spark cluster. <br> **Impact:** Determines where the Spark application will run. Setting it to `"yarn"` allows the application to run on a Hadoop YARN cluster. Changing this to `"local[*]"` runs the application locally, which is useful for development and testing but not suitable for large-scale data processing.                                                                               |
| `spark.executor.memory`        | `"4g"`                                           | **Purpose:** Allocates memory per executor process in Spark. <br> **Impact:** Affects the application's ability to handle larger datasets and perform computations efficiently. Insufficient memory may lead to out-of-memory errors, while excessive allocation can waste resources. Balancing executor memory is essential for optimal performance.                                                                                                   |
| `spark.rdd.compress`           | `"true"`                                         | **Purpose:** Enables compression of serialized RDD partitions. <br> **Impact:** Reduces the amount of memory and disk space used by RDDs, potentially improving performance when network and disk I/O are bottlenecks. However, compression adds CPU overhead, so the benefits depend on the specific workload and cluster configuration.                                                                                                        |
| `spark.io.compression.codec`   | `"lz4"`                                          | **Purpose:** Specifies the codec used for compressing internal data in Spark. <br> **Impact:** Affects the speed and efficiency of data compression and decompression. The `"lz4"` codec offers a good balance between compression speed and ratio, benefiting applications where I/O performance is critical. Selecting the appropriate codec can optimize resource utilization.                                                                |
| `hdfs.uri`                     | `"hdfs://localhost:9000/"`                       | **Purpose:** Defines the base URI for the Hadoop Distributed File System (HDFS). <br> **Impact:** Determines where the application reads input data from and writes output data to. Correct configuration is essential for successful data access and storage operations in a distributed environment.                                                                                                                                    |                                                                                                                                     
| `paths.model`                  | `"src/main/resources/trainedModel.zip"`          | **Purpose:** Specifies the local path where the trained model will be saved. <br> **Impact:** Allows the user to locate and load the trained model for inference or further analysis. Ensure that the path is writable and that sufficient storage space is available.                                                                                                                                 |
| `paths.csv`                    | `"src/main/resources/training_metrics.csv"`      | **Purpose:** Specifies the local path where the training metrics CSV file will be saved. <br> **Impact:** Enables tracking and analysis of training performance over epochs. Access to this file is important for diagnosing training issues and improving model performance.                                                                                                                         |


### 5. Training Metrics CSV Details

The `training_metrics.csv` file contains detailed metrics for each training epoch. Below is a description of each column in the CSV file:

| **Column**               | **Description**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
|--------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Epoch`                  | The current epoch number during training.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `TimeStamp`              | The duration of the epoch in milliseconds.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `TrainingLoss`           | The loss value computed on the training dataset for the current epoch.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `TrainingAccuracy`       | The accuracy of the model on the training dataset for the current epoch.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `ValidationAccuracy`     | The accuracy of the model on the validation dataset for the current epoch.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `meanGradientMagnitude`  | The mean magnitude of gradients computed during training, indicating the average size of gradient updates.                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `maxGradientMagnitude`   | The maximum gradient magnitude observed during training, useful for detecting exploding gradients.                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `minGradientMagnitude`   | The minimum gradient magnitude observed during training, useful for detecting vanishing gradients.                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `gradientVariance`       | The variance of gradient magnitudes, providing insight into the stability of gradient updates.                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `LearningRate`           | The current learning rate used by the optimizer during training.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `UsedMemoryMB`           | The amount of memory currently used by the JVM in megabytes.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `TotalMemoryMB`          | The total memory available to the JVM in megabytes.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `MaxMemoryMB`            | The maximum memory that the JVM will attempt to use, in megabytes.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `totalShuffleReadBytes`  | The total number of bytes read during shuffle operations in Spark, indicating data movement across the cluster.                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `totalShuffleWriteBytes` | The total number of bytes written during shuffle operations in Spark.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `maxTaskDuration`        | The maximum duration of any single task within the epoch, in milliseconds. Useful for identifying long-running tasks.                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `minTaskDuration`        | The minimum duration of any single task within the epoch, in milliseconds. Useful for identifying unusually short tasks.                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `avgTaskDuration`        | The average duration of tasks within the epoch, in milliseconds. Provides an overall view of task performance.                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `failedTaskCount`        | The number of tasks that failed during the epoch. Helps in monitoring the reliability of the training process.                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `processCpuLoad`         | The CPU load of the Spark driver process as a percentage. Indicates how much CPU resource the process is utilizing.                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `systemCpuLoad`          | The overall CPU load of the system as a percentage. Reflects the general CPU usage across all processes on the machine. 
