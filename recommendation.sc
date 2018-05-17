import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating

// Load and parse the data
val data = sc.textFile("/FileStore/tables/ratings.dat")
val ratings = data.map(_.split("::") match { case Array(user, item, rate,datestamp) =>
  Rating(user.toInt, item.toInt, rate.toDouble)
})

// Build the recommendation model using ALS
val rank = 10
val numIterations = 10
val Array(training, test) = ratings.randomSplit(Array[Double](0.6, 0.4), 18)
val model = ALS.train(training, rank, numIterations, 0.01)
// Evaluate the model on rating data
val usersProducts = test.map { case Rating(user, product, rate) =>
  (user, product)
}
val predictions =
  model.predict(usersProducts).map { case Rating(user, product, rate) =>
    ((user, product), rate)
  }
val ratesAndPreds = ratings.map { case Rating(user, product, rate) =>
  ((user, product), rate)
}.join(predictions)
val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
  val err = (r1 - r2)
  err * err
}.mean()
println(s"Mean Squared Error = $MSE")