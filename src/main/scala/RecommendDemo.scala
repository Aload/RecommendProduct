import org.apache.spark.mllib.recommendation.ALS.train
import org.apache.spark.mllib.recommendation.{MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.joda.time.DateTime


object RecommendDemo {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setAppName(this.getClass.getSimpleName)
      .setMaster("local[4]")
      .set("spark.executor.memory", "1G")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
    prepareData(sc)
  }

  def prepareData(sc: SparkContext): Unit = {
    //userId,movieId,rating,timestamp
    val rowRdd = sc.textFile("E:\\大数据\\ml-latest\\ratings.csv").map(_.split(",").take(3))
    val ratingRdd = rowRdd.map({
      case Array(userId, movieId, rating) => Rating(userId.toInt, movieId.toInt, rating.toDouble)
    })
    //movieId,title
    val moviedRdd = sc.textFile("E:\\大数据\\ml-latest\\").map(_.split(",").take(2))
      .map(x => (x(0).toInt, x(1)))

    //训练推荐参数
    val Array(trainData, validationData, testData) = ratingRdd.randomSplit(Array(0.8, 0.1, 0.1))
    trainValidation(trainData, validationData)
  }

  def trainValidation(trainData: RDD[Rating], validationData: RDD[Rating]): MatrixFactorizationModel = {
    //训练rank array(5,15,20,50,100)
    evaluateParameter(trainData, validationData, "rank", Array(5, 10, 15, 20, 50, 100), Array(10), Array(0.1))
    //训练iterate Array(5,10,20,25)
    evaluateParameter(trainData, validationData, "numInterations", Array(10), Array(5, 10, 20, 25), Array(0.1))
    //训练lambda Array(0.05,0.1,1,5.0,10.0)
    evaluateParameter(trainData, validationData, "lambda", Array(10), Array(10), Array(0.05, 0.1, 1, 5.0, 10.0))
    evaluateAllParameter(trainData, validationData, Array(5, 10, 15, 20, 50, 100), Array(5, 10, 20, 25), Array(0.05, 0.1, 1, 5.0, 10.0))
  }

  def evaluateParameter(trainData: RDD[Rating], validationData: RDD[Rating], evaluateParameter: String, rankArray: Array[Int], interatorArray: Array[Int], lambdaArray: Array[Double]) = {
    for (rank <- rankArray; interator <- interatorArray; lambda <- lambdaArray) {
      val (rmes, time) = trainModel(trainData, validationData, rank, interator, lambda)
      val parameter = evaluateParameter match {
        case "rank" => rank
        case "numInterations" => interator
        case "lambda" => lambda
      }
    }
  }

  def trainModel(trainData: RDD[Rating], validationData: RDD[Rating], rank: Int, interator: Int, lambda: Double): (Double, Double) = {
    val startTime = DateTime.now()
    val model = train(trainData, rank, interator, lambda)
    val endTime = DateTime.now()
    val rmse: Double = computeRmse(model, validationData)
    val duration = startTime.getMillisOfSecond - endTime.getMillisOfSecond
    (rmse, duration)
  }

  /**
    * 计算均平方根
    *
    * @param model
    * @param validationData
    * @return
    */
  def computeRmse(model: MatrixFactorizationModel, validationData: RDD[Rating]): Double = {
    val num = validationData.count()
    val preRdd = model.predict(validationData.map(r => (r.user, r.product)))
    val predictRating = preRdd.map(p => ((p.user, p.product), p.rating))
      .join(validationData.map(r => ((r.user, r.product), r.rating)))
      .values
    math.sqrt(predictRating.map(x => (x._1 - x._2) * (x._1 - x._2)).reduce(_ + _) / num)
  }

  /**
    * 全参数训练
    *
    * @param trainData
    * @param validateData
    * @param rankArray
    * @param interatorArray
    * @param lambdaArray
    * @return
    */
  def evaluateAllParameter(trainData: RDD[Rating], validateData: RDD[Rating], rankArray: Array[Int], interatorArray: Array[Int], lambdaArray: Array[Double]): MatrixFactorizationModel = {
    val evaluation = for (rank <- rankArray; interator <- interatorArray; lambda <- lambdaArray)
      yield {
        val (rmse, time) = trainModel(trainData, validateData, rank, interator, lambda)
        (rank, interator, lambda, rmse)
      }
    val eval = evaluation.sortBy(_._4)
    val bestEva = eval(0)
    train(trainData, bestEva._1, bestEva._2, bestEva._3)
  }
}
