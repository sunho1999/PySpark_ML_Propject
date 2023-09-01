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
from pyspark.sql.functions import mean,col,split,regexp_extract,when,lit,isnan,count

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

# when 메소드는 filter()와 비슷한 기능, when (조건 A,조건 A가 True일때, value),otherwise(조건 A가 False 일 때 value)로 사용
# 결측치 개수 확인
df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]) \
    .show()
# 결측치가 50퍼가 넘기 때문에 드랍
df = df.drop('Cabin') # 재할당

# 탑승객의 이름을 따 Initial 칼럼을 이용패 파생변수 생성. withColumn()을 이용하는데, 이때 PySpark 정규표현식 API를 사용.

# 새로운 파생변수를 생성한 데이터 프레임을 새로 할당!
df = df.withColumn("Initial",regexp_extract(col("Name"),"([A-Za-z]+)\.",1))
df.limit(3).show()

# salutation 값에 오탈자가 있어 값 수정

df = df.replace(["Mlle","Mme","Ms","Dr","Major","Lady","Countess","Jonkheer","Col","Rev","Capt","Sir","Don"],
                ["Miss","Miss","Miss","Mr","Mr","Mrs","Mrs","Other","Other","Other","Mr","Mr","Mr"])

# Initial 변수 값들로 그룹핑 후 평균 Age 구하기
print(df.groupby('Initial').avg("Age").collect())

# 평균 Age를 이용하여 결측치 대체

df = df.withColumn("Age",when((df['Initial'] == "Miss") & (df['Age'].isNull()),22).otherwise(df['Age']))
df = df.withColumn("Age",when((df['Initial'] == "Other") & (df['Age'].isNull()),46).otherwise(df['Age']))
df = df.withColumn("Age",when((df['Initial'] == "Master") & (df['Age'].isNull()),5).otherwise(df['Age']))
df = df.withColumn("Age",when((df['Initial'] == "Mr") & (df['Age'].isNull()),33).otherwise(df['Age']))
df = df.withColumn("Age",when((df['Initial'] == "Mrs") & (df['Age'].isNull()),36).otherwise(df['Age']))

# Ebarked 결측치 확인
df.groupBy('Embarked').count().show()

# 최빈값인 S로 대체.
df = df.na.fill({"Embarked":"S"})
# 결측치가 채워졌는지 확인
df.groupBy('Embarked').count().show()