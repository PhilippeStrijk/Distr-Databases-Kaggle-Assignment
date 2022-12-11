package be.hogent.dit.tin;

import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.BinaryLogisticRegressionTrainingSummary;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegressionTrainingSummary;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;


public class OpdrachtSparkLogistic {
	private static final int SEED = 10;
	public static void main(String[] args) {
		
		// Create a Spark Session
		SparkSession spark = SparkSession.builder()
		   .appName("Spark Demo")
		   .master("local[*]")
		   .config("spark.logLineage", true)
		     .config("spark.executor.memory", "70g")
		     .config("spark.driver.memory", "50g")
		     .config("spark.memory.offHeap.enabled",true)
		     .config("spark.memory.offHeap.size","16g")
		   .getOrCreate();

		// Create a Logger
		Logger logger = Logger.getLogger("org.apache");
		logger.setLevel(Level.WARN);

		// Read in files: test set, train set
		Dataset<Row> testSet = spark.read()
		   .option("header", "true")
		   .option("inferSchema", "true")
		   .csv("src/main/resources/test.csv");
		
		Dataset<Row> trainSet = spark.read()
				   .option("header", "true")
				   .option("inferSchema", "true")
				   .csv("src/main/resources/train.csv");
		
		// Fill up any null values to avoid errors with IDF Modeling. Change 'target' column name to 'label'
		testSet = testSet.na().fill("");
		
		trainSet = trainSet.na().fill("");
		trainSet = trainSet.withColumnRenamed("target", "label");
		trainSet = trainSet.withColumn("label", trainSet.col("label").cast("double"));
	
		// ---------------------------			AFTER PREPARATION 			------------------------------------------
		// Creating a TF-IDF vectorizer by: 
		// 1) Creating a tokenizer for the sentences (this one is identical for both Datasets, so only one is necessary)
		// 2) Featurizing the tokenized words (once again, only one is necessary)
		// 3) Fit and transform the featurized train set

		//Here, we make one big pipeline for our data.
		// Tokenizer, Featurizer, IDF
		Tokenizer tokenizer = new Tokenizer().setInputCol("question_text").setOutputCol("words");
		HashingTF hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures");
		IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
		

		//-------------------------			  Tokenize, Featurize and IDF the sets.		-------------------------
		testSet = tokenizer.transform(testSet);
		testSet = hashingTF.transform(testSet);
		IDFModel idfTestModel = idf.fit(testSet);
		testSet = idfTestModel.transform(testSet);
		
		trainSet = tokenizer.transform(trainSet);
		trainSet = hashingTF.transform(trainSet);
		IDFModel idfTrainModel = idf.fit(trainSet);
		trainSet = idfTrainModel.transform(trainSet);
		// -------------------------------------------------------------------------------------------------------

		
		
		// ------------------		Split train in x and y 	----------------------------
		// Only continue with yTrain, otherwise the program fails
		Dataset<Row>[] split = trainSet.randomSplit(new double[]{0.75, 0.25}, SEED); 
		//Dataset<Row> xTrain = split[0];
		Dataset<Row> yTrain = split[1];
		// -----------------------------------------------------------------------------
		
		

		// ----------------			Create validation set		-------------------------
		//Dataset<Row> xVal = xTrain.select(xTrain.col("target"));
		Dataset<Row> yVal = yTrain.select(yTrain.col("label"));
		// ------------------------------------------------------------------------------

		
		
		// Define Logistic Regression model
		// Here, we use multinomial regression, because the model has to pick between 0 and 1. This is binary classification.
		LogisticRegression mlr = new LogisticRegression()
		        .setMaxIter(10)
		        .setRegParam(0.3)
		        .setElasticNetParam(0.8);
		
		// Drop columns that aren't necessary for the LR model
		//xTrain = xTrain.drop("qid").drop("question_text").drop("words").drop("rawFeatures");
				
		yTrain = yTrain.drop("qid").drop("question_text").drop("words").drop("rawFeatures");
				

		//xTrain = xTrain.na().fill(0);
		yTrain = yTrain.na().fill(0);

		// OutOfMemoryError Java Heap Space on xTrain, not on yTrain however (presumably because it's smaller). For the sake of this task we'll continue with the yTrain.
		LogisticRegressionModel mlryTrainModel = mlr.fit(yTrain);
			
		
		// ---------------------------			THIS PART SADLY DOESN'T WORK		----------------------------------------------
		// 						EITHER Java OutOfMemoryError OR 'Exception thrown by Await#result'
		/*
		Pipeline pipeline = new Pipeline()
				  .setStages(new PipelineStage[] {mlr});
		
		ParamMap[] paramGrid = new ParamGridBuilder()
				  .addGrid(hashingTF.numFeatures(), new int[] {10, 100, 1000})
				  .addGrid(mlr.regParam(), new double[] {0.1, 0.01})
				  .build();
		
		CrossValidator cv = new CrossValidator()
				  .setEstimator(pipeline)
				  .setEvaluator(new BinaryClassificationEvaluator())
				  .setEstimatorParamMaps(paramGrid)
				  .setNumFolds(2)  // Use 3+ in practice
				  .setParallelism(2);

		CrossValidatorModel cvModel = cv.fit(trainSet);
		
		Dataset<Row> predictions = cvModel.transform(testSet);

		// -----------------------------------------------------------------------------------------------------------------
		*/

		Dataset<Row> predictions = mlryTrainModel.transform(testSet);
		
		predictions.show(false);
		
		// -------------------------------------------------- 				SUMMARRY 			----------------------------------------------------------------
		// Print the coefficients and intercepts for logistic regression with multinomial family
		System.out.println("Multinomial coefficients: " + mlryTrainModel.coefficientMatrix()
		  + "\nMultinomial intercepts: " + mlryTrainModel.interceptVector());
		
		LogisticRegressionTrainingSummary trainingSummary = mlryTrainModel.summary();
		
		// Obtain the loss per iteration.
		double[] objectiveHistory = trainingSummary.objectiveHistory();
		for (double lossPerIteration : objectiveHistory) {
		    System.out.println(lossPerIteration);
		}

		// for multiclass, we can inspect metrics on a per-label basis
		System.out.println("False positive rate by label:");
		int i = 0;
		double[] fprLabel = trainingSummary.falsePositiveRateByLabel();
		for (double fpr : fprLabel) {
		    System.out.println("label " + i + ": " + fpr);
		    i++;
		}

		System.out.println("True positive rate by label:");
		i = 0;
		double[] tprLabel = trainingSummary.truePositiveRateByLabel();
		for (double tpr : tprLabel) {
		    System.out.println("label " + i + ": " + tpr);
		    i++;
		}

		System.out.println("Precision by label:");
		i = 0;
		double[] precLabel = trainingSummary.precisionByLabel();
		for (double prec : precLabel) {
		    System.out.println("label " + i + ": " + prec);
		    i++;
		}

		System.out.println("Recall by label:");
		i = 0;
		double[] recLabel = trainingSummary.recallByLabel();
		for (double rec : recLabel) {
		    System.out.println("label " + i + ": " + rec);
		    i++;
		}

		System.out.println("F-measure by label:");
		i = 0;
		double[] fLabel = trainingSummary.fMeasureByLabel();
		for (double f : fLabel) {
		    System.out.println("label " + i + ": " + f);
		    i++;
		}

		double accuracy = trainingSummary.accuracy();
		double falsePositiveRate = trainingSummary.weightedFalsePositiveRate();
		double truePositiveRate = trainingSummary.weightedTruePositiveRate();
		double fMeasure = trainingSummary.weightedFMeasure();
		double precision = trainingSummary.weightedPrecision();
		double recall = trainingSummary.weightedRecall();
		System.out.println("Accuracy: " + accuracy);
		System.out.println("FPR: " + falsePositiveRate);
		System.out.println("TPR: " + truePositiveRate);
		System.out.println("F-measure: " + fMeasure);
		System.out.println("Precision: " + precision);
		System.out.println("Recall: " + recall);
		
		spark.close();
		

	}

}
