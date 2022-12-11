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
		trainSet.show(false);
		
		//------------------------				Above here is the TrainSet where the column 'question_text' is changed to 'tokens' and all seperate words are correctly tokenized -----------------------------

		
		// Create a Dataset<Row> for the output data
		Dataset<Row> outputData = spark.createDataFrame(new ArrayList<Row>(), trainSet.schema());

	
			
		spark.close();
		// --------------			AFTER SPARK			--------------------------
		
		System.out.println("Size test set: " + sizeTestSet + "   Size train set: " + sizeTrainSet);

		System.out.println("Size with target = 1: " + sizeFilteredDs);
	
		StructType schema = trainSet.schema();
		StructField[] schemaFields = schema.fields();
		List<StructField> schemaFieldsList = Arrays.asList(schemaFields);
		for(StructField field : schemaFieldsList) {
		    System.out.println("Column Name: " + field.name() + " DataType: " + field.dataType());
		}

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
