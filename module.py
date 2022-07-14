from pyspark.sql import SparkSession

from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel
import pyspark.sql.functions as F
import pandas as pd

spark = SparkSession.builder.master("local[*]").getOrCreate()
spark.conf.set("spark.sql.repl.eagerEval.enabled", True)  # Property used to format output tables bettern

model = LogisticRegressionModel.load('436-model')
pipeline = PipelineModel.load('436-pipeline')


def evaluate_csv(pd_data):
    df = spark.createDataFrame(pd_data)
    datapipe = pipeline.transform(df)
    data = datapipe.select(F.col("VectorAssembler_features").alias("features"),
                           F.col("isFraud").alias("predictedIsFraud"))

    out = model.transform(data)
    formatted = pd.concat([out.toPandas().reset_index(drop=True), pd_data], axis=1)
    return formatted

