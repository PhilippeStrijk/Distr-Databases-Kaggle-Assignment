package main;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

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


import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
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
import org.apache.spark.ml.feature.PCA;
import org.apache.spark.ml.feature.PCAModel;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.mllib.linalg.VectorUDT;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.sql.types.Metadata;

import static org.apache.spark.sql.functions.callUDF;

public class Kaggle {
	
	private static final int SEED = 42;
	
	public static void main(String[] args) {
		//create spark session
		SparkSession spark = SparkSession.
				builder()
				.appName("Spark Demo")
				.master("local[*]")
				.config("spark.logLineage", true)
				.getOrCreate();
		
		// Create a Logger
		Logger logger = Logger.getLogger("org.apache");
		logger.setLevel(Level.WARN);
		
		//read the files				
		String path = "src/main/resources/tabular-playground-series-nov-2022";
		
		//------------------ load x_train
		/*Dataset<Row> trainSet= spark.read()
				.option("header", true)
				.csv(path + "/submission_files.csv");*/
		
		Dataset<Row> trainSet= spark.read()
				.option("header", true)
				.option("inferSchema","true")
				.csv(path + "/submission_files/0.6222863195.csv");
		//------------------- load y
		Dataset<Row> y= spark.read()
				.option("header", true)
				.csv(path + "/train_labels.csv");

		
		//------------------ feature selection
		List<Row> data = Arrays.asList(
				  RowFactory.create(Vectors.sparse(5, new int[]{1, 3}, new double[]{1.0, 7.0})),
				  RowFactory.create(Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0)),
				  RowFactory.create(Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0))
				);
		StructType schema = new StructType(new StructField[]{
				  new StructField("pred", new VectorUDT(), false, Metadata.empty()),
				  new StructField("label", DataTypes.IntegerType,false,Metadata.empty())
				});
		
		Dataset<Row> df = spark.createDataFrame(data, schema);
		
		PCA pca = new PCA()
				  .setInputCol("pred")
				  .setOutputCol("pcaFeatures")
				  .setK(3);
				 // .fit(df);
		//trainSet = pca.transform(trainSet).select("pcaFeatures");
		
		//splitting
		Dataset<Row>[] split = trainSet.randomSplit(new double[] {0.80,.20}, SEED);
		
		Dataset<Row> x_train = split[0];
		Dataset<Row> x_test = split[1];
		
		Dataset<Row> y_train = y.select(y.col("label"));
		x_train.show(10);
		y.show(10);
		y_train.show(10);
		
		//-------------------- creating the pipeline
		LogisticRegression lr = new LogisticRegression()
				.setMaxIter(10)
				.setRegParam(0.1)
				.setElasticNetParam(0.8)
				.setFeaturesCol("pred");
		
		
		Pipeline model_logsitic= new Pipeline()
										.setStages(new PipelineStage[] {pca,lr,});
		
		PipelineModel pipeMod = model_logsitic.fit(x_train);
		
		//------------ testing
		List<Row> dataTest = Arrays.asList(
			    RowFactory.create(1.0, Vectors.dense(-1.0, 1.5, 1.3)),
			    RowFactory.create(0.0, Vectors.dense(3.0, 2.0, -0.1)),
			    RowFactory.create(1.0, Vectors.dense(0.0, 2.2, -1.5))
			);
		Dataset<Row> test = spark.createDataFrame(dataTest, schema);
		

		Dataset<Row> rows = test.select("features", "label", "myProbability", "prediction");
		for (Row r: rows.collectAsList()) {
		  System.out.println("(" + r.get(0) + ", " + r.get(1) + ") -> prob=" + r.get(2)
		    + ", prediction=" + r.get(3));
		}
		
		spark.close();
	}

}
