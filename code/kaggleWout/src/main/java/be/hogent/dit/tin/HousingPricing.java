package be.hogent.dit.tin;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.OneHotEncoderModel;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import static org.apache.spark.sql.functions.*;

import java.io.File;
import java.io.PrintWriter;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang.ArrayUtils;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;

public class HousingPricing {
	public static void main(String[] args) {
		//https://www.kaggle.com/code/saumyadeepm/analysis-house-price-predictor/notebook?scriptVersionId=111412196
		
		// Create a Logger
		Logger logger = Logger.getLogger("org.apache");
		logger.setLevel(Level.WARN);
		
		// Create Spark Session
		SparkSession spark = SparkSession.builder()
				.appName("HousingPricing").master("local[*]")
				.config("spark.driver.host", "localhost")
				.config("spark.sql.debug.maxToStringFields", 1000)
				.getOrCreate();
					
		Dataset<Row> train = spark.read()
			.option("header", true)
			.option("inferschema", true)
			.csv("src/main/resources/train.csv");
		Dataset<Row> test = spark.read()
				.option("header", true)
				.option("inferschema", true)
				.csv("src/main/resources/test.csv");
	
		
		// Everything above 4500 is to far from the average and we drop
		train = train.where(col("GrLivArea").lt(4500));
		
		
		// Count number of empty values for each column
		List<Row> total = new ArrayList<>();
		
		for (String c : test.columns()){
			total.add(RowFactory.create(c, test.where(col(c).equalTo("NA")).count()));
		}
		List<StructField> fields = Arrays.asList(
				DataTypes.createStructField("column", DataTypes.StringType, true),
				DataTypes.createStructField("Total", DataTypes.LongType, true)
				);
		StructType schema = DataTypes.createStructType(fields);
		Dataset<Row> missingData = spark.createDataFrame(total, schema);
		
		
		// Create the new dataframe where there are no columns with missing values
		missingData.orderBy(desc("Total")).show(45);

		for (Row c : missingData.select(col("column")).where(col("Total").gt(0)).collectAsList() ) {
			train = train.drop(c.getString(0));
			
		}
		
		
		// test where there are no columns with missing data
		for (Row c : missingData.select(col("column")).where(col("Total").gt(0)).collectAsList() ) {
			test = test.drop(c.getString(0));
			
		}
		test = test.drop("Electrical");
		
//		test = test.withColumn("Electrical", lit("NA"));
		train = train.drop("Electrical");
		train = train.withColumnRenamed("SalePrice", "label");
		train.show();
		
		
		//create linear regression model
		LinearRegression lr = new LinearRegression()
				.setMaxIter(10);
		
		//map categorical features
		StringIndexer indexer = new StringIndexer()
				.setInputCols(new String[] {"Street","LotShape","LandContour","LotConfig","LandSlope","Neighborhood","Condition1","Condition2","BldgType","HouseStyle","OverallQual","OverallCond","YearBuilt","YearRemodAdd","RoofStyle","RoofMatl","ExterQual","ExterCond","Foundation","Heating","HeatingQC","CentralAir","PavedDrive","MoSold","YrSold","SaleCondition"})
				.setOutputCols(new String[] {"StreetIndex","LotShapeIndex","LandContourIndex","LotConfigIndex","LandSlopeIndex","NeighborhoodIndex","Condition1Index","Condition2Index","BldgTypeIndex","HouseStyleIndex","OverallQualIndex","OverallCondIndex","YearBuiltIndex","YearRemodAddIndex","RoofStyleIndex","RoofMatlIndex","ExterQualIndex","ExterCondIndex","FoundationIndex","HeatingIndex","HeatingQCIndex","CentralAirIndex","PavedDriveIndex","MoSoldIndex","YrSoldIndex","SaleConditionIndex"});
		OneHotEncoder encoder = new OneHotEncoder()
				.setInputCols(indexer.getOutputCols())
				.setOutputCols(new String[] {"StreetVec","LotShapeVec","LandContourVec","LotConfigVec","LandSlopeVec","NeighborhoodVec","Condition1Vec","Condition2Vec","BldgTypeVec","HouseStyleVec","OverallQualVec","OverallCondVec","YearBuiltVec","YearRemodAddVec","RoofStyleVec","RoofMatlVec","ExterQualVec","ExterCondVec","FoundationVec","HeatingVec","HeatingQCVec","CentralAirVec","PavedDriveVec","MoSoldVec","YrSoldVec","SaleConditionVec"});
		VectorAssembler assembler = new VectorAssembler()
				  .setInputCols(encoder.getOutputCols())
				  .setOutputCol("features");
		
		
		Dataset<Row> transform = train.select("Street","LotShape","LandContour","LotConfig","LandSlope","Neighborhood","Condition1","Condition2","BldgType","HouseStyle","OverallQual","OverallCond","YearBuilt","YearRemodAdd","RoofStyle","RoofMatl","ExterQual","ExterCond","Foundation","Heating","HeatingQC","CentralAir","PavedDrive","MoSold","YrSold","SaleCondition");
		Dataset<Row> transform_test = test.select("Street","LotShape","LandContour","LotConfig","LandSlope","Neighborhood","Condition1","Condition2","BldgType","HouseStyle","OverallQual","OverallCond","YearBuilt","YearRemodAdd","RoofStyle","RoofMatl","ExterQual","ExterCond","Foundation","Heating","HeatingQC","CentralAir","PavedDrive","MoSold","YrSold","SaleCondition");

		transform = indexer.fit(transform).transform(transform);
		transform = encoder.fit(transform).transform(transform);
		train = train.drop(indexer.getInputCols());
		train = train.join(transform.selectExpr(encoder.getOutputCols()));
		train.show();
		train = assembler.transform(train);
		train = train.select("label","features");
		train.show();
		
		transform_test = indexer.fit(transform_test).transform(transform_test);
		transform_test = encoder.fit(transform_test).transform(transform_test);
		test = test.drop(indexer.getInputCols());
		test = test.join(transform.selectExpr(encoder.getOutputCols()));
		test = assembler.transform(test);
		test = test.select("features");
		

		//split train in train en validation data
		ParamMap[] paramGrid = new ParamGridBuilder()
				  .addGrid(lr.regParam(), new double[] {0.1})
				  .addGrid(lr.fitIntercept())
				  .addGrid(lr.elasticNetParam(), new double[] {0.5})
				  .build();
		TrainValidationSplit tvs = new TrainValidationSplit()
				.setEstimator(lr)
				.setEvaluator(new RegressionEvaluator())
				.setEstimatorParamMaps(paramGrid)
				.setTrainRatio(0.80)
				.setParallelism(2);
		
		TrainValidationSplitModel model = tvs.fit(train);
		Dataset<Row> predictions = model.transform(test);
		predictions.show();
		predictions.select("features","prediction").show();
		
		Dataset<Row> result = predictions.select("prediction");
		//output to csv
		result.write().option("header", true).csv("~/Downloads/submission.csv");
		
	}
}
