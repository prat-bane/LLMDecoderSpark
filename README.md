# Large Language Model Encoder on AWS EMR

### Author : Pratyay Banerjee
### Email : pbane8@uic.edu

### Youtube video : [https://youtu.be/QWJDm6ITcMg](https://youtu.be/QWJDm6ITcMg)

## Overview

The application reads token IDs from a text file, generates embeddings, and trains a neural network model using a sliding window approach. It leverages **Apache Spark** for distributed data processing and DL4J for deep learning tasks.

## Project Workflow

1. **Data Sharding and Tokenization**
   - **Sharding**:
     The input text data is divided into smaller, manageable chunks to facilitate parallel processing across the EMR cluster. For sharding we have created the FileSharder class, which is used to shard by number of lines. We have also created a TextPreprocessor class which is used by the TokenizerJob to preproccess the input text. The TextPreproccesor class concats every word in the input text with its position and creates a shard. Classes used: **FileSharder.scala**,**TextPreprocessor.scala**
    
   - **Tokenization**: Each shard is tokenized using [Jtokkit](https://github.com/nocduro/jtokkit), an efficient tokenizer for large-scale text data.
     For Tokenization I have created 2 MapReduce jobs, so please don't confused.\
     i)The first **MapReduce** job I have created is the **WordToTokenJob**. It is used to generate  **word token frequency**. This was created just to meet the requirement of the first part of the project(frequency requirement). The output of this MapperReducer is not used in the rest of the project. \
     ii)The second **MapReduce** job for tokenization is the TokenizerJob which has a TokenizerMapper and TokenizerReducer. This uses an input which is preprocessed by the TextPreprocessor i.e he input shards have the positions concatenated with the words. This job generates word and token ids in  same order the input text was in. After completion, this job performs post-processing steps, which include
     extracting the token ids from the part-r-0000x files(output files of the TokenizerJob), consolidate the token ids and create a **tokenids.txt** file, and then sharding the tokenids.txt file so that it can be
     fed to the TokenEmbeddingJob to generate vector embedding from these tokens. \
     Classes used:**WordToTokenJob**,**TokenizerJob**,**TokenizerMapper**,**TokenizerReducer**
     
2. **Token Embedding Generation**
   - **MapReduce Job**: In the next MapReduce job, we generate embeddings for the tokens consolidated from the previous step.
   - **TokenEmbeddingMapper**: Processes the tokens, generates the embeddings, passes them to the reducer.
   - **TokenEmbdeddingReducer**: Averages the embeddings in the `TokenEmbedding` job to create a unified vector representation for each token.
   Classes used :**TokenEmbeddingJob**,**TokenEmbeddingMapper**,**TokenEmbdeddingReducer**

3. **Cosine Similarity Calculation**
   - **CosineSimilarity Job**: Calculates the cosine similarity of all vector embeddings. The **CosineSimilarityDriver** class adds an allEmbeddings.txt file to the Distributed cache so that all reducers
     have access to all embedding vectors. I have been manually merging the part-r-0000x files from TokenEmbeddingJob to create an alLEmbeddings.txt file using the command
     ``` hadoop fs -cat /output/embeddings/part-r-* | hadoop fs -put - /input/cosine/allEmbeddings.txt```
   - **CosineSimilarityMapper**: Processes the embeddings and passes them to the reducer.
   - **CosineSimilarityReducer**: The reducer loads the allEmbeddings.txt file from the Distributed cache and then stores it in a Map. Each reducer then computes the cosine similarity of the emebeddings
     in its input split with all other embeddings stored in the Map.
   - **Result**: Generates the top 5 similar words for each word based on the cosine similarity scores.

## Getting Started

#Youtube video Link: https://youtu.be/QWJDm6ITcMg

### Prerequisites

```bash
- Scala (version 2.13.12)
- Apache Hadoop (version 3.3.4)
- SBT (Scala Build Tool, version 1.10.1)
- Java JDK (version 1.8 )
```
### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/prat-bane/LLM-MapReduce
   cd project-directory
   ```
### Running Tests
```
sbt clean compile test
```

### Running Different Mappers and Reducers

#### Argument variables can be set in Intellij Run Configurations while running locally.

1) WordToTokenJob : It accepts 2 arguments. The first one is the path to the input shards, the second one is the output path. The input shard has to be created using the **FileSharder** class shardByLines method. FileSharder also has a main method. The shardByLines accepts inputPath,outputPath,number of lines per shard and a flag variable called skipPreprocessing. This flag variable is there because this method is used for sharding input text file for WordToTokenJob as well as sharding the tokenids.txt file,created by TokenizerJob. Sharding the tokenids.txt does not need preprocessing so the flag remains true for it.
   ```hadoop jar \path\to\jar  mapreduce.WordToTokenJob /input/shards /output/wordcount```
3) TokenizerJob : It accepts 6 arguments.\
   i) Input path (path to shards processed by TexTtProcessor class) \
   ii) Output path\
   iii) number of reducers\
   iv) Output path for creating tokenids.txt(a file which contains just the token ids extracted from the part-r-0000x files of this job)\
   v) Output path for the shards of tokenids.txt (tokenids.txt will be sharded so that it can be fed to TokenEmbeeding job to generate vector embeddings)\
   vi)Number of lines per shard\
   Sample Config(pass this as args. Set it in Intellij run config):
```E:\text\shards D:\IdeaProjects\ScalaRest\src\main\resources\test\text\output 4 D:\IdeaProjects\ScalaRest\src\main\resources\test\text\tokenids.txt D:\IdeaProjects\ScalaRest\src\main\resources\test\text\tokens\output 30000```
5) TokenEmbeddingJob : It accepts 2 arguments. The input path and the output path. The input path will be the path from point v) of 2.TokenizerJob(previous pt) i.e the output path of shards of tokenids.txt\
 ```D:\IdeaProjects\ScalaRest\src\main\resources\test\text\tokens\output D:\IdeaProjects\ScalaRest\src\main\resources\test\embedding\output```
6) CosineSimilarityDriver. It accepts 3 arguments. The input path(the output path of TokenEmbeddingJob i.e the previous job), output path, and the path to allEmbeddings.txt(the file mergeed from the part-r-0000x files of TokenEmbeddingJob)
   ```hadoop jar /path/to/jar mapreduce.CosineSimilarityDriver /output/embedding /output/cosine path/to/allEmbeddings.txt```

#### Configuration file
```
app{

    embedding-config {
      window-size = 8
      stride = 2
      embedding-size = 100
      lstm-layer-size = 128
      learning-rate = 0.001
      epochs = 500
      batch-size = 32
    }

    cosine-similarity{
       topN = 5
    }

}
```

### General Configuration

| **Variable**      | **Default Value** | **Description**                                                                                                                                                                 |
|-------------------|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `window-size`     | `8`                | **Purpose:** Defines the size of the sliding window used during tokenization or context windowing. <br> **Impact:** Determines how many tokens are considered together, affecting contextual understanding and computational load. A larger window size captures more context but requires more resources. |
| `stride`          | `2`                | **Purpose:** Specifies the step size by which the sliding window moves across the dataset. <br> **Impact:** Controls the overlap between windows. Smaller strides increase coverage and redundancy, enhancing learning but increasing processing time. |
| `embedding-size`  | `100`              | **Purpose:** Sets the dimensionality of the vector embeddings generated for each token. <br> **Impact:** Higher embedding sizes capture more nuanced semantic relationships but demand more memory and computational power. Balances representation richness with resource utilization. |
| `lstm-layer-size` | `128`              | **Purpose:** Determines the number of units (neurons) in each Long Short-Term Memory (LSTM) layer of the neural network. <br> **Impact:** Affects the model's capacity to learn complex patterns and dependencies in the data. Larger sizes enhance learning capability but increase the risk of overfitting and computational requirements. |
| `learning-rate`   | `0.001`            | **Purpose:** Controls the step size for updating model parameters during training. <br> **Impact:** Balances convergence speed and training stability. A learning rate that's too high can cause overshooting of minima, while too low a rate can result in slow convergence. Finding an optimal learning rate is essential for effective training. |
| `epochs`          | `500`              | **Purpose:** Specifies the number of complete passes through the entire training dataset. <br> **Impact:** More epochs allow the model to learn more from the data, potentially improving accuracy. However, excessive epochs can lead to overfitting, where the model performs well on training data but poorly on unseen data. |
| `batch-size`      | `32`               | **Purpose:** Determines the number of samples processed before updating the model's internal parameters. <br> **Impact:** Affects training stability and computational efficiency. Larger batch sizes make better use of parallel hardware (e.g., GPUs) but require more memory. Optimal batch sizes balance memory usage with the quality of gradient estimates. |

### Cosine Similarity Configuration

| **Variable**                 | **Default Value** | **Description**                                                                                                                                                         |
|------------------------------|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `cosine-similarity.topN`     | `5`               | **Purpose:** Specifies the number of top semantically similar words to retain for each target word based on cosine similarity scores. <br> **Impact:** Determines the breadth of similarity results. Higher values provide more related words but increase computational load during similarity calculations. Ideal for applications requiring a comprehensive set of related terms. |


