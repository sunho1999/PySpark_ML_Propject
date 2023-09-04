import pandas
import numpy
import os
import matplotlib.pyplot as plt
import  seaborn as sns
plt.rc('font',family = "AppleGothic")
plt.rcParams['axes.unicode_minus'] = False

# PySpark - SQL
from pyspark.sql import SparkSession
from pyspark.sql.functions import mean,col,split,regexp_extract,when,lit,isnan,count

# Pyspark - ML 파이프라인
from pyspark.ml import Pipeline

# Feature
from pyspark.ml.feature import StringIndexer,VectorAssembler

# 분류 모델
from pyspark.ml.classification import LogisticRegression
# 파라미터 튜닝 & 교차 검즘
from pyspark.ml.tuning import ParamGridBuilder,TrainValidationSplit
from pyspark.ml.tuning import CrossValidator

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import QuantileDiscretizer

# Matric
from pyspark.mllib.evaluation import BinaryClassificationMetrics as metric
from pyspark import SparkContext
# ROC AUC
from sklearn.metrics import roc_curve, auc

# Spark Session 만들기

spark = SparkSession.builder\
    .appName("Titanic Data pysaprk ML")\
    .getOrCreate()

# data load
df = spark.read.csv("/Users/sunho99/PycharmProjects/python_Project/PySpark_MLlib/titanic_project/titanic/train.csv", header=True,inferSchema=True)

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
df.groupby('Initial').avg("Age").collect()

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

# Family Size 파생 변수 생성
df = df.withColumn("Family_Size", df["SibSp"] + df["Parch"])
# Alone이라는 Binary 파생변수 생성, 0으로 초기화
df = df.withColumn("Alone",lit(0))
# 조건에 맞게 Alone 변수값 변경
df = df.withColumn("Alone",when(df["Family_Size"] == 0,1)\
                   .otherwise(df["Alone"]))
# Label Encoding 진행
covert_cols = ["Sex","Embarked","Initial"]

# IndexToString을 사용하기 위해 indexer객체 생성

indexer = [StringIndexer(inputCol=col,outputCol=col+'_index').fit(df) for col in covert_cols] # column 별로 StringIndexer를 만듬

for i in indexer:
    print(i)
    print("-"*80)


# Pipeline 이용하여 stage에다가 실행 과정 담아 넘기기

pipeline = Pipeline(stages=indexer)
df = pipeline.fit(df).transform(df)

# 불필요한 칼럼들 삭제 후 최종 Feature들을 Vector로 변환

un_cols = ["PassengerId","Name","Ticket","Cabin","Embarked","Sex","Initial"]

df = df.drop(*un_cols) # Native Python 기능

# 남은 Feature들을 Vector로 변환하여 머신러닝 모델에 입력시킬 준비
# VectorAssembler()와 StringIndexer()의 차이는 파라미터 인자중 inputCol에 s를 붙일 수 있음, fit을 하지 않고, 바로 transform 수행

feature = VectorAssembler(inputCols=df.columns[1:],outputCol="features")
feature_vector = feature.transform(df) # 데이터 프레임 형태로 변환

titanic_df = feature_vector.select(["features","Survived"])

# split trian,test
(train_df,test_df) = titanic_df.randomSplit([0.8,0.2],seed = 42) # 데이터 분류

lr = LogisticRegression(labelCol="Survived")

# 튜닝할 파라미터 grid 정의
paramGrid = ParamGridBuilder().addGrid(lr.regParam,(0.01,0.1))\
                              .addGrid(lr.maxIter,(5,10)) \
                              .addGrid(lr.tol,(1e-4,1e-5))\
                              .addGrid(lr.elasticNetParam,
                                       (0.25,0.75))\
                              .build()

# 교차 검증 정의 - pipeline식으로 정의

tvs = TrainValidationSplit(estimator=lr,
                           estimatorParamMaps=paramGrid,
                           evaluator=MulticlassClassificationEvaluator(labelCol="Survived"),
                           trainRatio=0.8)

# 학습은 fit!
model = tvs.fit(train_df)
# 평가는 transform!
model_prediction = model.transform(test_df)

# metric 평가
print("Accuracy: ",MulticlassClassificationEvaluator(labelCol="Survived",metricName="accuracy").evaluate(model_prediction))

print("Precision: ",MulticlassClassificationEvaluator(labelCol="Survived",metricName="weightedPrecision").evaluate(model_prediction))

model_prediction.show(10)
# rawPrediction : 해당 feature을 넣었을 때, 계산되어 나오는 raw 값
# probability : rawPrediction값에 Logistic 함수를 적용한 후 변환 된값, 0-1사이
# Prediction: 특정 임계값 기준으로 1 또는 0으로 분류된 클래스(label)

# SparkContext를 만들기
sc = SparkContext.getOrCreate()

# ROC 점수인 AUC 계산을 위해 Logistic을 적용해 나온 확률값과 레이블만 가져오기
results = model_prediction.select(['probability','Survived'])

# 확률 값- 레이블 set
# collect()로 모든 데이터 row retrieve(반환)
result_collect = results.collect()

# print(result_collect[0])
# print("Probability: ",result_collect[0].probability)
# print("Survived: ",result_collect[0].Survived)

results_list = [(float(i.probability[1]),float(i.Survived)) for i in result_collect]

# 여러개의 튜플이 담긴 list를 RDD자료구조로 변경
scoreAndLabels = sc.parallelize(results_list)
# ROC metric 계산하기
metrics = metric(scoreAndLabels)

# Visualize
fpr = []
tpr = []
roc_auc = []

y_test = [i[1] for  i in results_list]
y_proba = [i[0] for i in results_list]

fpr,tpr,_ = roc_curve(y_test,y_proba)
roc_auc = auc(fpr,tpr)

plt.figure()
plt.plot(fpr,tpr,label= "ROC Curve(area = %0.2f)" % roc_auc)
plt.plot([0,1],[0,1],"k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Area under the ROC Curve")
plt.legend(loc="lower right")
plt.show()