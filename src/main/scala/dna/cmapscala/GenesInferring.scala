package dna.cmapscala

import maum.dm.Matrix
import maum.dm.ACC_CUDA
import maum.dm.ACC_NONE
import maum.dm.NeuralNetworkClassification;
import maum.dm.optimizer.Optimizer;
import java.io.FileReader
import java.io.BufferedReader
import java.io.FileWriter
import java.io.BufferedWriter
import java.lang.Double

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf


/** State vector extracting function object. */ // Inner object is valid?
object ExtractStateVector {
  def extractSV(v: scala.Double): Array[scala.Double] = {
    val stateVector = new Array[scala.Double](GenesInferring.numGeneValues)
    stateVector(v.asInstanceOf[Int]) = 1.0
    stateVector
  }
}

/** GenesInferring object. */
object GenesInferring {
  
  // Constants.
  val numLandmarkGenes = 970
  val numTargetGenes = 11350
  val numTotalGenes = 12320
  val numSamples = 100000
  val numGeneValues = 16; //?
  val numTestSamples = 1650
}

/** GenesInferrring companion class. */
class GenesInferring(sc: SparkContext) {
     
  // Training matrixes.
  var rawGeneM : Matrix[ACC_NONE] = null
  var landmarkGeneM : Matrix[ACC_NONE] = null
  var truthGeneM : Matrix[ACC_NONE] = null
  var landmarkGeneSVMs : Array[Matrix[ACC_NONE]] = null 
  
  // Neural Network Classification.
  class NNC(clusterComputingMode: Int, acceleratingComputingMode: Int, numLayers: Int, numActs: Array[Int], optimizer: Optimizer) 
    extends NeuralNetworkClassification(clusterComputingMode, acceleratingComputingMode, numLayers, numActs, optimizer) {
    
    // Judge.
    override def judge(iM: Matrix[ACC_NONE]):Matrix[ACC_NONE] = {
      val result = feedForwardC(iM)
      result
    }
  }
  
  // Unit genes inferring model.
  class UnitGeneInferringModel {
    var tensors = new Array[NNC](GenesInferring.numLandmarkGenes)
    var abstractTensor:NNC = null
  }
  
  // Genes inferring model.
  class GenesInferringModel {
    val genesInferringModels = 
      new Array[UnitGeneInferringModel](GenesInferring.numTargetGenes)
    
    // Infer genes.
    def inferGenes(testLandmarkGeneM: Matrix[ACC_NONE]): Matrix[ACC_NONE] = {
      
      // Extract a state vectors Matrix[ACC_NONE] for each landmark gene.
      var testLandmarkGeneSVMs = new Array[Matrix[ACC_NONE]](GenesInferring.numLandmarkGenes)
    
      for (i <- 1 to GenesInferring.numLandmarkGenes) {
      
        // Get a selected gene matrix.
        val range = Array[Int](1, GenesInferring.numTestSamples, i, i)
        val sGeneM = testLandmarkGeneM.getSubMatrix(range)
      
        // Make a resilient distributed data set.
        val sGeneA = new Array[scala.Double](GenesInferring.numSamples)
      
        for (k <- 0 to (GenesInferring.numSamples - 1)) {
          sGeneA(k) = sGeneM.getVal(k + 1, 1)
        }
      
        val distDataSet = sc.parallelize(sGeneA)
      
        // Transform gene values into state vectors via distribution computing.
        val stateVectorARDD = distDataSet.map(ExtractStateVector.extractSV)
      
        // Make the matrix for state vectors of a selected landmark gene.
        val stateVectorA = stateVectorARDD.take(stateVectorARDD.count().asInstanceOf[Int])
      
        for (k <- 0 until stateVectorA.length) {
          val colUnitM = new Matrix[ACC_NONE](1, GenesInferring.numGeneValues, stateVectorA(k))
        
          // Add.
          if (k == 0) {
            testLandmarkGeneSVMs(i - 1) = colUnitM // Valid??
          } else {
            testLandmarkGeneSVMs(i - 1).verticalAdd(colUnitM)
          }
        }
      }
    
      // Infer genes from landmark genes.
      var inferredGeneM: Matrix[ACC_NONE] = null
      
      for (sampleNum <- 0 until GenesInferring.numTestSamples) {
        
        // Make an unit inferred genes matrix.        
        val unitInferredGeneM = new Matrix[ACC_NONE](1, GenesInferring.numTargetGenes, 0.0)
        
        for (inferredGeneNum <- 0 until GenesInferring.numTargetGenes) {
          
          // Get an unit gene inferring model.
          val gModel = genesInferringModels(inferredGeneNum)
          
          // Infer a gene value.
          var selectedGeneInferredM: Matrix[ACC_NONE] = null 
            
          // Collect landmark genes' state vectors.
          var svs = new Array[Tuple2[Array[scala.Double], Int]](GenesInferring.numLandmarkGenes)
        
          for (m <-0 until GenesInferring.numLandmarkGenes) {
            svs(m) = new Tuple2(testLandmarkGeneSVMs(m).rowVector(sampleNum + 1), m)
          }
        
          val svsRDD = sc.parallelize(svs)
          val svsTensorTransRDD = svsRDD.map(x => gModel.tensors(x._2).judge(new Matrix[ACC_NONE](GenesInferring.numGeneValues, 1, x._1)))
          selectedGeneInferredM = svsTensorTransRDD.reduce((a, b) => a.plus(b))

          // Determine a gene value via selecting an index with a maximum probability magnitude.
          var maxIndex = 0
          var maxVal = 0.0
          
          for (index <- 0 until GenesInferring.numGeneValues) {
            if (selectedGeneInferredM.getVal(index + 1, 1) > maxVal) {
              maxIndex = index
              maxVal = selectedGeneInferredM.getVal(index + 1, 1)
            }
          }
          
          unitInferredGeneM.setVal(1, inferredGeneNum + 1, maxIndex)
        }
        
        if (sampleNum == 0) {
          inferredGeneM = unitInferredGeneM
        } 
        
        inferredGeneM.verticalAdd(unitInferredGeneM)
      }
      
      inferredGeneM
    }
  }
    
  // Genes inferring model instance.
  val geneInferringModel = new GenesInferringModel()
  
  // Train.
  def train(trainingDataFilePath: String) {
    
    // Parse training data.
    parseTrainingData(trainingDataFilePath)
    
    // Train.
    // Extract a state vectors matrix for each landmark gene.
    landmarkGeneSVMs = new Array[Matrix[ACC_NONE]](GenesInferring.numLandmarkGenes)
    
    for (i <- 1 to GenesInferring.numLandmarkGenes) {
      
      // Get a selected gene matrix.
      val range = Array[Int](1, GenesInferring.numSamples, i, i)
      val sGeneM = landmarkGeneM.getSubMatrix(range)
      
      // Make a resilient distributed data set.
      val sGeneA = new Array[scala.Double](GenesInferring.numSamples)
      
      for (k <- 0 to (GenesInferring.numSamples - 1)) {
        sGeneA(k) = sGeneM.getVal(k + 1, 1)
      }
      
      val distDataSet = sc.parallelize(sGeneA)
      
      // Transform gene values into state vectors via distribution computing.
      val stateVectorARDD = distDataSet.map(ExtractStateVector.extractSV)
      
      // Make the matrix for state vectors of a selected landmark gene.
      val stateVectorA = stateVectorARDD.take(stateVectorARDD.count().asInstanceOf[Int])
      
      for (k <- 0 until stateVectorA.length) {
        val colUnitM = new Matrix[ACC_NONE](1, GenesInferring.numGeneValues, stateVectorA(k))
        
        // Add.
        if (k == 0) {
          landmarkGeneSVMs(i - 1) = colUnitM // Valid??
        } else {
          landmarkGeneSVMs(i - 1).verticalAdd(colUnitM)
        }
      }
    }
    
    // Train each unit gene inferring model.
    var count = 1
    
    for (gModel: UnitGeneInferringModel <- geneInferringModel.genesInferringModels) {
      
      // Get a target selected gene matrix.
      var targetSVM:Matrix[ACC_NONE] = null
      
      val range = Array[Int](1, GenesInferring.numSamples, count, count)
      val sGeneM = truthGeneM.getSubMatrix(range)
      
      // Make a resilient distributed data set.
      val sGeneA = new Array[scala.Double](GenesInferring.numSamples)
      
      for (k <- 0 to (GenesInferring.numSamples - 1)) {
        sGeneA(k) = sGeneM.getVal(k + 1, 1)
      }
      
      val distDataSet = sc.parallelize(sGeneA)
      
      // Transform gene values into state vectors via distribution computing.
      val stateVectorARDD = distDataSet.map(ExtractStateVector.extractSV)
      
      // Make the matrix for state vectors of a selected landmark gene.
      val stateVectorA = stateVectorARDD.take(stateVectorARDD.count().asInstanceOf[Int])
      
      for (k <- 0 until stateVectorA.length) {
        val colUnitM = new Matrix[ACC_NONE](1, GenesInferring.numGeneValues, stateVectorA(k))
        
        // Add.
        if (k == 0) {
          targetSVM = colUnitM // Valid??
        } else {
          targetSVM.verticalAdd(colUnitM)
        }
      }  
      
      // Train each tensor.      
      for (k <- 0 until gModel.tensors.length) {
        
        // Initialize a spark neural network model.
        val numActs = Array(GenesInferring.numGeneValues, GenesInferring.numGeneValues)
        gModel.tensors(k) = new NNC(0, 0, 2, numActs)
        
        // Train the NN.
        gModel.tensors(k).train(landmarkGeneSVMs(k), targetSVM)
      }
      
      // Train an abstract tensor.
      val numActsForAbstractTensor = Array(GenesInferring.numGeneValues, GenesInferring.numGeneValues)
      gModel.abstractTensor = new NNC(2, numActsForAbstractTensor)
      
      // Calculate linear combination matrixes for all landmark genes' 
      // state vector's tensor transformation
      var lcM: Matrix[ACC_NONE] = null 
      
      for (k <- 0 until GenesInferring.numSamples) {
        
        // Collect landmark genes' state vectors.
        var svs = new Array[Tuple2[Array[scala.Double], Int]](GenesInferring.numLandmarkGenes)
        
        for (m <-0 until GenesInferring.numLandmarkGenes) {
          svs(m) = new Tuple2(landmarkGeneSVMs(m).rowVector(k + 1), m)
        }
        
        val svsRDD = sc.parallelize(svs)
        val svsTensorTransRDD = svsRDD.map(x => gModel.tensors(x._2).judge(new Matrix[ACC_NONE](GenesInferring.numGeneValues, 1, x._1)))
        lcM = svsTensorTransRDD.reduce((a, b) => a.plus(b))
      }
      
      // Train.
      gModel.abstractTensor.train(lcM, targetSVM)
      
      count += 1 
    } 
  }
  
  // Test.
  def test(testDataFilePath: String) {
    
    // Load a test data file.
    var testLandmarkGeneM: Matrix[ACC_NONE] = null
    
    val fr = new FileReader(testDataFilePath)
    val br = new BufferedReader(fr)
    
    // Check exception.
    if (!br.ready()) {
      println("Can't parse training data")
      return
    }
    
    val line = br.readLine()
     
    // Split string values and convert them into double values.
    val geneStrVals = line.split(",")
    val geneVals = new Array[scala.Double](geneStrVals.length)
     
    for (i <- 0 to (geneStrVals.length - 1)) {
      geneVals(i) = Double.valueOf(geneStrVals(i)) * 1.0 ;
    }
      
    testLandmarkGeneM = new Matrix[ACC_NONE](1, geneStrVals.length, geneVals)
    
    var count = 0
    
    while (br.ready()) {
      println(count)
      count += 1
      
      val line = br.readLine()
     
      // Split string values and convert them into double values.
      val geneStrVals = line.split(",")
      val geneVals = new Array[scala.Double](geneStrVals.length)
      
      for (i <- 0 to (geneStrVals.length - 1)) {
        geneVals(i) = Double.valueOf(geneStrVals(i)) * 1.0;
      }
      
      val tempM = new Matrix[ACC_NONE](1, geneStrVals.length, geneVals)
      
      // Add a matrix.
      testLandmarkGeneM.verticalAdd(tempM)
    }
    
    // Transpose the test landmark gene matrix.
    val samples = testLandmarkGeneM.colLength()
    
    println(samples)
    
    testLandmarkGeneM = testLandmarkGeneM.transpose()
    
    println(testLandmarkGeneM.rowLength(), testLandmarkGeneM.colLength())
    
    // Infer genes.
    val inferredGeneM = geneInferringModel.inferGenes(testLandmarkGeneM)
    
    // Save result as csv.
    saveResultAsCSV(inferredGeneM)
  }
  
  // Save result as csv.
  def saveResultAsCSV(resultM: Matrix[ACC_NONE]) {
    val fw = new FileWriter("InferredGenes.csv")
    val bw = new BufferedWriter(fw)

    // Save an inferred genes sequence for each sample.
    for (i <- 0 until resultM.rowLength()) {
      val geneValues = resultM.rowVector(i + 1)
      val sb = new StringBuilder()
      
      for (k <- 0 until (geneValues.length - 1)) {
        sb.append(geneValues(k))
        sb.append(",")
      }
      
      sb.append(geneValues(geneValues.length - 1))
      
      bw.write(sb.toString())
      bw.newLine()
    }
    
    bw.close()
  }
  
  // Parse training data.
  def parseTrainingData(trainingDataFilePath: String) {
      
    // Load a training data file.
    val fr = new FileReader(trainingDataFilePath)
    val br = new BufferedReader(fr)
    
    // Check exception.
    if (!br.ready()) {
      println("Can't parse training data")
      return
    }
    
    var count = 1
    
    println(count)
    count += 1
    
    val line = br.readLine()
     
    // Split string values and convert them into double values.
    val geneStrVals = line.split(",")
    val geneVals = new Array[scala.Double](geneStrVals.length)
     
    for (i <- 0 to (geneStrVals.length - 1)) {
      geneVals(i) = Double.valueOf(geneStrVals(i)) * 1.0 ;
    }
      
    rawGeneM = new Matrix[ACC_NONE](1, geneStrVals.length, geneVals)
        
    while (br.ready()) {
      println(count)
      count += 1
      
      val line = br.readLine()
     
      // Split string values and convert them into double values.
      val geneStrVals = line.split(",")
      val geneVals = new Array[scala.Double](geneStrVals.length)
      
      for (i <- 0 to (geneStrVals.length - 1)) {
        geneVals(i) = Double.valueOf(geneStrVals(i)) * 1.0;
      }
      
      val tempM = new Matrix[ACC_NONE](1, geneStrVals.length, geneVals)
      
      // Add a matrix.
      rawGeneM.verticalAdd(tempM)
    }
    
    br.close()
     
    // Extract landmark genes and truth genes.
    // Transpose the raw gene matrix.
    val samples = rawGeneM.colLength()
    
    println(samples)
    
    rawGeneM = rawGeneM.transpose()
    
    println(rawGeneM.rowLength(), rawGeneM.colLength())
    
    // Landmark.
    val rangeL = Array(1, samples, 1, GenesInferring.numLandmarkGenes)
    landmarkGeneM = rawGeneM.getSubMatrix(rangeL)
    
    // Truth.
    val rangeT = Array(1, samples, GenesInferring.numLandmarkGenes + 1, GenesInferring.numTotalGenes)
    truthGeneM = rawGeneM.getSubMatrix(rangeT)
  }
}