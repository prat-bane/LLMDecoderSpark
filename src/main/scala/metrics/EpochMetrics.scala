package metrics

case class EpochMetrics(
                         epoch: Int,
                         timestamp: Long,
                         trainingLoss: Double,
                         trainingAccuracy: Double,
                         validationAccuracy: Double,
                         meanGradientMagnitude: Double,
                         maxGradientMagnitude: Double,
                         minGradientMagnitude: Double,
                         gradientVariance: Double,
                         learningRate: Double,
                         usedMemoryMB: Long,
                         totalMemoryMB: Long,
                         maxMemoryMB: Long,
                         totalShuffleReadBytes: Long,
                         totalShuffleWriteBytes: Long,
                         maxTaskDuration: Long,
                         minTaskDuration: Long,
                         avgTaskDuration: Double,
                         failedTaskCount: Int,
                         processCpuLoad: Double,
                         systemCpuLoad: Double
                       )
