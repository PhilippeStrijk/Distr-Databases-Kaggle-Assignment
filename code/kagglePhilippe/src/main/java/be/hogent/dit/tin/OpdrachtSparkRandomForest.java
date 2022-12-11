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
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.BinaryLogisticRegressionTrainingSummary;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegressionTrainingSummary;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.VectorIndexer;
import org.apache.spark.ml.feature.VectorIndexerModel;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;


public class OpdrachtSparkRandomForest {
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
		testSet = testSet.withColumnRenamed("target", "label");
		testSet = testSet.drop("label");
		
		
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

		
		//-------------------------			  Tokenize, Featurize, IDF and VectorIndex the sets.		-------------------------
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
		
		// Tried running the Random Forest Classifier on only 1% of the dataset; but it's still awfully slow.
		// Exception: java.lang.OutOfMemoryError thrown from the UncaughtExceptionHandler in thread "RemoteBlock-temp-file-clean-thread"
		// Exception: java.lang.OutOfMemoryError thrown from the UncaughtExceptionHandler in thread "Spark Context Cleaner"
		// Exception in thread "spark-listener-group-appStatus" java.lang.OutOfMemoryError: Java heap space

		Dataset<Row>[] split = trainSet.randomSplit(new double[]{0.99, 0.01}, SEED); 
		//Dataset<Row> xTrain = split[0];
		Dataset<Row> yTrain = split[1];
		yTrain = yTrain.na().fill(0);
		// -----------------------------------------------------------------------------
		
		StringIndexerModel labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(yTrain);
		VectorIndexerModel featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(yTrain);

		// ----------------			Create validation set		-------------------------
		//Dataset<Row> xVal = xTrain.select(xTrain.col("target"));
		Dataset<Row> yVal = yTrain.select(yTrain.col("label"));
		// ------------------------------------------------------------------------------

		
		
		// Define Random Forest
		RandomForestClassifier rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures");
		
		IndexToString labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labelsArray()[0]);
		
		Pipeline pipeline = new Pipeline()
				  .setStages(new PipelineStage[] {labelIndexer, featureIndexer, rf, labelConverter});
		
		PipelineModel model = pipeline.fit(yTrain);
		
		Dataset<Row> predictions = model.transform(testSet);
		
		predictions.select("predictedLabel", "label", "features").show(5);

		
		// Select (prediction, true label) and compute test error
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
		  .setLabelCol("indexedLabel")
		  .setPredictionCol("prediction")
		  .setMetricName("accuracy");
		double accuracy = evaluator.evaluate(predictions);
		System.out.println("Test Error = " + (1.0 - accuracy));

		RandomForestClassificationModel rfModel = (RandomForestClassificationModel)(model.stages()[2]);
		System.out.println("Learned classification forest model:\n" + rfModel.toDebugString());
		
		

		// Drop columns that aren't necessary for the LR model
		//xTrain = xTrain.drop("qid").drop("question_text").drop("words").drop("rawFeatures");
				
		//yTrain = yTrain.drop("qid").drop("question_text").drop("words").drop("rawFeatures");
				

		//xTrain = xTrain.na().fill(0);

		
		spark.close();
		

	}

}
