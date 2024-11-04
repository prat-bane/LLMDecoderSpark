# Large Language Model Encoder on AWS EMR

### Author : Pratyay Banerjee
### Email : pbane8@uic.edu

### Youtube video : [https://youtu.be/QWJDm6ITcMg](https://youtu.be/QWJDm6ITcMg)

## Overview

The application reads token IDs from a text file, generates embeddings, and trains a neural network model using a sliding window approach. It leverages **Apache Spark** for distributed data processing and DL4J for deep learning tasks.

## Project Workflow

1. **Sliding Window operation on Tokens**
   - **Sliding window and token embedding generation**:
     The input tokens from the token id file(src/main/resources/tokenid.txt) that we generated in hw1 has been used in this project. The token embeddings of those tokens have been calculated and stored in a
     map.
    
   - **Sliding window RDD dataset formation**: RDDs of Sliding window data has been created.
     
2. **Positional Embedding Generation and Training Dataset formation**
   - Positional embeddings of the RDD windowed data from the previous step has been calculated and added to their respective token embeddings to form the input and target embeddings,which forms the training dataset.
  

3. **Model Training**
   - The model has been trained with the input and output targe embeddings.
   - **Result**: Model zip and training metrics file.

## Getting Started

#Youtube video Link: https://youtu.be/QWJDm6ITcMg

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

1) Run the SlidingWindowUtil class.
   

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
  master = "yarn"
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
