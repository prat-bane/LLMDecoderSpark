package metrics

import org.deeplearning4j.nn.api.Model
import org.deeplearning4j.nn.gradient.Gradient
import org.deeplearning4j.optimize.api.BaseTrainingListener
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer

import scala.collection.mutable

class GradientStatsListener extends BaseTrainingListener with Serializable {
  @transient private var gradientStats = mutable.Map[String, Double]()

  // Initialize gradientStats if it's null (after deserialization)
  private def ensureInitialized(): Unit = {
    if (gradientStats == null) {
      gradientStats = mutable.Map[String, Double]()
    }
  }

  override def onGradientCalculation(model: Model): Unit = {
    ensureInitialized()
    val network = model.asInstanceOf[org.deeplearning4j.nn.multilayer.MultiLayerNetwork]
    val gradient = network.gradient()
    val flattenedGradient = flattenGradient(gradient)

    if (flattenedGradient.nonEmpty) {
      val mean = flattenedGradient.sum / flattenedGradient.length
      val max = flattenedGradient.max
      val min = flattenedGradient.min

      // Calculate variance
      val variance = if (flattenedGradient.length > 1) {
        val meanDiffs = flattenedGradient.map(m => math.pow(m - mean, 2))
        meanDiffs.sum / (flattenedGradient.length - 1)
      } else 0.0

      // Store statistics
      gradientStats("meanMagnitude") = mean
      gradientStats("maxMagnitude") = max
      gradientStats("minMagnitude") = min
      gradientStats("variance") = variance
    }
  }

  private def flattenGradient(gradient: Gradient): Array[Double] = {
    val gradientMap = gradient.gradientForVariable()
    val allGradients = mutable.ArrayBuffer[Double]()

    gradientMap.values().forEach { gradientINDArray =>
      if (gradientINDArray != null) {
        val flatArray = gradientINDArray.dup().data().asDouble()
        allGradients ++= flatArray.map(math.abs)
      }
    }

    allGradients.toArray
  }

  def getGradientStats: Map[String, Double] = {
    ensureInitialized()
    gradientStats.toMap
  }
}

