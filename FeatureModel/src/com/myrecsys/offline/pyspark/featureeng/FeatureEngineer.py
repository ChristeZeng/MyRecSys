from pyspark import SparkConf
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, QuantileDiscretizer, MinMaxScaler
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import functions as F


def oneHotEncoderExample(mvsamples):
    samplesWithIdNumber = mvsamples.withColumn('movieIdNumber', F.col("movieId").cast(IntegerType()))
    encoder = OneHotEncoder(inputCols=['movieIdNumber'], outputCols=['movieIdVector'], dropLast=False)
    oneHotEncoderSamples = encoder.fit(samplesWithIdNumber).transform(samplesWithIdNumber)
    oneHotEncoderSamples.printSchema()
    oneHotEncoderSamples.show(10)


def array2vec(genreIndexes, indexSize):
    genreIndexes.sort()
    fill_list = [1.0 for _ in range(len(genreIndexes))]
    return Vectors.sparse(indexSize, genreIndexes, fill_list)


def multiHotEncoderExample(mvsamples):
    samplesWithGenre = mvsamples.select("movieId", "title", explode(
        split(F.col("genres"), "\\|").cast(ArrayType(StringType()))).alias('genre'))
    samplesWithGenre.show(10)
    samplesWithGenre.printSchema()
    genreIndexer = StringIndexer(inputCols="genre", outputCol="genreIndex")
    StringIndexerModel = genreIndexer.fit(samplesWithGenre)
    genreIndexSamples = StringIndexerModel.transform(samplesWithGenre).withColumn('genreIndexInt',
                                                                                  F.col('genreIndex').cast(IntegerType()))
    indexSize = genreIndexSamples.agg(max(F.col('genreIndexInt'))).head()[0] + 1
    processedSamples = genreIndexSamples.groupBy('movieId').agg(
        F.collect_list('genreIndexInt').alias('genreIndexes')).withColumn('indexSize', F.col(indexSize))
    finalSample = processedSamples.withColumn('vector',
                                            udf(array2vec, VectorUDT())(F.col('genreIndexes'), F.col('indexSize')))
    finalSample.printSchema()
    finalSample.show(10)


if __name__ == '__main__':
    conf = SparkConf().setAppName('featureEngineer').setMaster('local')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    filePath = '/Users/christezeng/Documents/MyRecSys/dataset'
    mvResourcePath = filePath + '/ml-latest-small/movies.csv'
    mvsamples = spark.read.format('csv').option('header', 'true').load(mvResourcePath)
    print("Raw Movie Samples: ")
    mvsamples.show(10)
    mvsamples.printSchema()
    print("OneHotEncoder Example: ")
    oneHotEncoderExample(mvsamples)
    print("MultiHotEncoder Examples: ")
    multiHotEncoderExample(mvsamples)




