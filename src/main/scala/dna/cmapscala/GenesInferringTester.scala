package dna.cmapscala

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

/** Genes inferring model tester. */
object GenesInferringTester {
  
  def main(args: Array[String]) {
    
    // Check arguments.
    if (args.length < 2) {
      println("GenesInferring [trainning data file path] [test data file path]")
      return
    }
    
    // Get arguments.
    val trainingDataFilePath = args(0)
    val testDataFilePath = args(1)
    
    // Initialize Spark environment.
    val conf = new SparkConf().setAppName("Genes inferring model").setMaster("local[4]")
    
    val sc = new SparkContext(conf)
    
    // Create the genes inferring model.
    val genesInferringModel = new GenesInferring(sc)
    
    // Train.
    genesInferringModel.train(trainingDataFilePath)
    
    // Test.
    // TODO
  }
}