import pandas
import numpy
import os
import matplotlib.pyplot as plt
import  seaborn as sns
plt.rc('font',family = "AppleGothic")
plt.rcParams['axes.unicode_minus'] = False

# PySpark - SQL
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.functions import mean,col,split,regexp_extract,when,lit

# Pyspark - ML
from pyspark.ml import pipeline
from pyspark.ml.feature import StringIndexer,VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import QuantileDiscretizer

# Spark Session 만들기

spark = SparkSession.builder\
    .appName("Titanic Data pysaprk ML")\
    .getOrCreate()

# data load
df = spark.read.csv("/Users/sunho99/PycharmProjects/python_Project/PySpark/PySpark_MLlib/titanic_project/titanic/train.csv", header=True,inferSchema=True)

# to Pandas()를 이용하여 pandas에서 제공하는 df형태로 출력
# df.limit(3).toPandas()

pandas_df = df.toPandas()
print("pandas_df 타입: ", type(pandas_df))

plt.figure(figsize = (15,10))
plt.title("타이타닉 탑승객의 Age KDE 분포")
sns.displot(pandas_df["Age"])
plt.show()