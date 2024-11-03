package metrics

import org.apache.spark.scheduler.{SparkListener, SparkListenerStageCompleted, SparkListenerTaskEnd}

import scala.collection.mutable.ArrayBuffer

class CustomSparkListener extends SparkListener {

  val taskMetricsData = ArrayBuffer[TaskMetricsData]()
  val stageMetricsData = ArrayBuffer[StageMetricsData]()
  var failedTaskCount: Int = 0

  override def onTaskEnd(taskEnd: SparkListenerTaskEnd): Unit = {
    val metrics = taskEnd.taskMetrics
    val shuffleReadBytes = Option(metrics.shuffleReadMetrics).map(_.totalBytesRead).getOrElse(0L)
    val shuffleWriteBytes = Option(metrics.shuffleWriteMetrics).map(_.bytesWritten).getOrElse(0L)
    val taskDuration = taskEnd.taskInfo.duration

    val taskMetrics = TaskMetricsData(
      taskId = taskEnd.taskInfo.taskId,
      stageId = taskEnd.stageId,
      duration = taskDuration,
      shuffleReadBytes = shuffleReadBytes,
      shuffleWriteBytes = shuffleWriteBytes
    )

    taskMetricsData += taskMetrics

    if (taskEnd.taskInfo.failed) {
      failedTaskCount += 1
    }
  }

  override def onStageCompleted(stageCompleted: SparkListenerStageCompleted): Unit = {
    val stageInfo = stageCompleted.stageInfo
    val stageId = stageInfo.stageId
    val numTasks = stageInfo.numTasks
    val stageDuration = stageInfo.completionTime.getOrElse(0L)

    val stageMetrics = StageMetricsData(
      stageId = stageId,
      numTasks = numTasks,
      duration = stageDuration
    )

    stageMetricsData += stageMetrics
  }

  def reset(): Unit = {
    taskMetricsData.clear()
    stageMetricsData.clear()
    failedTaskCount = 0
  }
}

case class TaskMetricsData(
                            taskId: Long,
                            stageId: Int,
                            duration: Long,
                            shuffleReadBytes: Long,
                            shuffleWriteBytes: Long
                          )

case class StageMetricsData(
                             stageId: Int,
                             numTasks: Int,
                             duration: Long
                           )
