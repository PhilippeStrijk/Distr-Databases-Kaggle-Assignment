package be.hogent.dit.tin;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.catalyst.encoders.RowEncoder;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.util.LongAccumulator;

import scala.Tuple2;
import scala.collection.mutable.WrappedArray;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.BinaryLogisticRegressionTrainingSummary;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegressionTrainingSummary;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import static org.apache.spark.sql.functions.callUDF;


public class LogisticPaperVersion {
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
				
		trainSet = trainSet.na().fill("");
		testSet = testSet.na().fill("");
		long sizeTestSet = testSet.count();
		long sizeTrainSet = trainSet.count();
		// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Now we want to remove question marks, comma's and dots. We Declare, Register and Apply the UDF 
		// Declare the UDF
		UDF1<String, String> replaceQM = new UDF1<String, String>() {
		    public String call (final String str) throws Exception {
		        return str.replaceAll("[\\?,.]", "");
		    }
		};
		// Register the UDF
		spark.udf().register("replaceQM", replaceQM, DataTypes.StringType);
		// Apply the UDF
		trainSet.select(functions.callUDF("replaceQM", trainSet.col("question_text"))
		    .alias("column_name_without_QM"));
		
		trainSet = trainSet.withColumn("column_name_without_QM", functions.callUDF("replaceQM", trainSet.col("question_text")));
		trainSet = trainSet.drop("question_text").withColumnRenamed("column_name_without_QM", "question_text");
	//--------------------------------------		Above here is the TrainSet where the column 'question_text' doesn't have any question marks, comma's or dots. ------------------------------------------- 
		
		
		
		//Check how many incinsere questions we  have
		Dataset<Row> filteredDataset = trainSet.filter("target = 1");
		long sizeFilteredDs = filteredDataset.count();
		
		// Before we remove the stopwords, we have to tokenize the column 'question_text'
		Tokenizer tokenizer = new Tokenizer().setInputCol("question_text").setOutputCol("tokens");
		trainSet = tokenizer.transform(trainSet);
	
		// Now, we can remove all stopwords
		StopWordsRemover remover = new StopWordsRemover()
				  .setInputCol("tokens")
				  .setOutputCol("tokens w/o stopwords");	
		String[] originalStopwords = remover.getStopWords();
		String[] stopwords = new String[] {"one ", "br", "Po", "th", "sayi", "fo", "unknown"};
		String[] allStopwords = combineObjects(stopwords, originalStopwords);
		remover.setStopWords(allStopwords);
		trainSet = remover.transform(trainSet);		

		trainSet = trainSet.drop("question_text", "tokens");
		trainSet = trainSet.withColumnRenamed("tokens w/o stopwords", "tokens");
		
		//------------------------				Above here is the TrainSet where the column 'question_text' is changed to 'tokens' and all seperate words are correctly tokenized -----------------------------

		//Hashing
		HashingTF hashingTF = new HashingTF().setInputCol("tokens").setOutputCol("rawFeatures");
		trainSet = hashingTF.transform(trainSet);
		trainSet = trainSet.drop("tokens");
	
		//IDF
		IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
		IDFModel idfTrainModel = idf.fit(trainSet);
		trainSet = idfTrainModel.transform(trainSet);
		
		//Some cleaning
		trainSet = trainSet.drop("rawFeatures");
		trainSet = trainSet.na().fill("");
		trainSet = trainSet.withColumnRenamed("target", "label");
		trainSet = trainSet.withColumn("label", trainSet.col("label").cast("double"));
		trainSet = trainSet.na().fill(0);
		

		//Continue with smaller trainSet
		Dataset<Row>[] split = trainSet.randomSplit(new double[]{0.8, 0.2}, SEED); 
		trainSet = split[1];

		//Before we apply the model to find our predictions, we have to clean our testSet as well.
		//However, we don't need to remove stopwords or question marks, comma's or dots. 
		//The model isn't trained to pick up on these things. We only need to hash and vectorize.
		testSet = tokenizer.transform(testSet);
		testSet = hashingTF.transform(testSet);
		testSet = testSet.drop("tokens");
		IDFModel idfTestModel = idf.fit(testSet);
		testSet = idfTestModel.transform(testSet);
		testSet.show();
		
		
		//Logistic Regression
		LogisticRegression mlr = new LogisticRegression()
		        .setMaxIter(10)
		        .setRegParam(0.3)
		        .setElasticNetParam(0.8);
		
		LogisticRegressionModel mlryTrainModel = mlr.fit(trainSet);

		Dataset<Row> predictions = mlryTrainModel.transform(testSet);
		predictions.show(false);
		
		
		// --------------			AFTER SPARK			--------------------------
		
		System.out.println("Size test set: " + sizeTestSet + "   Size train set: " + sizeTrainSet);

		System.out.println("Size with target = 1: " + sizeFilteredDs);
	
		StructType schema = trainSet.schema();
		StructField[] schemaFields = schema.fields();
		List<StructField> schemaFieldsList = Arrays.asList(schemaFields);
		for(StructField field : schemaFieldsList) {
		    System.out.println("Column Name: " + field.name() + " DataType: " + field.dataType());
		}
		
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
	
	public static String[] combineObjects(String[] obj1, String[] obj2) {
		   String[] combinedObject = new String[obj1.length + obj2.length];
		   int i = 0; 
		   for (String s : obj1) {
		      combinedObject[i++] = s;
		   }
		   for (String s : obj2) {
		      combinedObject[i++] = s;
		   }
		   return combinedObject;
		}
	
	

}
