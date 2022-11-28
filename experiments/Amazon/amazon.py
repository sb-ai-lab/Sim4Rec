import os
os.environ["CUDA_VISIBLE_DEVICES"]=""

import pandas as pd
import numpy as np
import requests
import gzip
import json
import re

import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from typing import List, Optional

from bs4 import BeautifulSoup
from nltk.tokenize import TreebankWordTokenizer, WhitespaceTokenizer

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')
words = set(nltk.corpus.words.words())
words = set([w.lower() for w in words])
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
from nltk.tokenize import sent_tokenize

import gensim
from gensim.downloader import load
from gensim.models import Word2Vec
w2v_model = gensim.downloader.load('word2vec-google-news-300')

import pyspark 
from pyspark.sql.types import *
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import functions as sf
from pyspark.ml.feature import Tokenizer, RegexTokenizer, StopWordsRemover
from pyspark.ml import Pipeline
from pyspark.sql.functions import expr 


def get_spark_session(
    mode
) -> SparkSession: 
    """
    The function creates spark session
    :param mode: session mode
    :type mode: int
    :return: SparkSession
    :rtype: SparkSession
    """
    if mode == 1:
        
        SPARK_MASTER_URL = 'spark://spark:7077'
        SPARK_DRIVER_HOST = 'jupyterhub'
        
        conf = SparkConf().setAll([
        ('spark.master', SPARK_MASTER_URL),
        ('spark.driver.bindAddress', '0.0.0.0'),
        ('spark.driver.host', SPARK_DRIVER_HOST),
        ('spark.driver.blockManager.port', '12346'),
        ('spark.driver.port', '12345'),
        ('spark.driver.memory', '8g'), #4
        ('spark.driver.memoryOverhead', '2g'),
        ('spark.executor.memory', '14g'), #14
        ('spark.executor.memoryOverhead', '2g'),
        ('spark.app.name', 'simulator'),
        ('spark.submit.deployMode', 'client'),
        ('spark.ui.showConsoleProgress', 'true'),
        ('spark.eventLog.enabled', 'false'),
        ('spark.logConf', 'false'),
        ('spark.network.timeout', '10000000'),
        ('spark.executor.heartbeatInterval', '10000000'),
        ('spark.sql.shuffle.partitions', '4'),
        ('spark.default.parallelism', '4'),
        ("spark.kryoserializer.buffer","1024"),
        ('spark.sql.execution.arrow.pyspark.enabled', 'true'),
        ('spark.rpc.message.maxSize', '1000'),
        ("spark.driver.maxResultSize", "2g")    
        ])
        spark = SparkSession.builder\
            .config(conf=conf)\
            .getOrCreate()
    
    elif mode == 0:
        spark = SparkSession.builder\
            .appName('simulator')\
            .master('local[4]')\
            .config('spark.sql.shuffle.partitions', '4')\
            .config('spark.default.parallelism', '4')\
            .config('spark.driver.extraJavaOptions', '-XX:+UseG1GC')\
            .config('spark.executor.extraJavaOptions', '-XX:+UseG1GC')\
            .config('spark.sql.autoBroadcastJoinThreshold', '-1')\
            .config('spark.sql.execution.arrow.pyspark.enabled', 'true')\
            .getOrCreate()
               
    return spark

def clean_text(text: str) -> str:
    
    """
    Cleaning and preprocessing of the tags text with help of regular expressions.
    :param text: initial text
    :type text: str
    :return: cleaned text
    :rtype: str
    """

    text = re.sub("[^a-zA-Z]", " ",text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+$", "", text)
    text = re.sub(r"^\s+", "", text)
    text = text.lower()

    return text


def string_embedding(arr: list) -> np.ndarray:
    """
    Processing each word in the string with word2vec and return their aggregation (mean).
    
    :param arr: words
    :type text: List[str]
    :return: average vector of word2vec words representations
    :rtype: np.ndarray
    """
    
    vec = 0
    cnt = 0
    for i in arr:
        try:
            vec += w2v_model[i]
            cnt += 1
        except:
            pass
    if cnt == 0:
        vec = np.zeros((300,))
    else:
        vec /= cnt
    return vec

@sf.pandas_udf(StringType(), sf.PandasUDFType.SCALAR)
def clean_udf(str_series):
    """
    pandas udf of the clean_text function
    """
    result = []
    for x in str_series:
        x_procc = clean_text(x)
        result.append(x_procc)
    return pd.Series(result)

@sf.pandas_udf(ArrayType(DoubleType()), sf.PandasUDFType.SCALAR)
def embedding_udf(str_series):
    """
    pandas udf of the string_embedding function
    """
    result = []
    for x in str_series:
        x_procc = string_embedding(x)
        result.append(x_procc)
    return pd.Series(result)


def data_processing(df_sp: pyspark.sql.DataFrame):
    df_procc = df.withColumnRenamed("item_id", "item_idx")\
             .withColumnRenamed("user_id", "user_idx")\
             .withColumn("review_clean", clean_udf(sf.col("review"))).drop("review", "__index_level_0__")
    df_procc = tokenizer.transform(df_procc).drop("review_clean")
    df_procc = remover.transform(df_procc).drop("tokens")
    df_procc = df_procc.withColumn("embedding", embedding_udf(sf.col("tokens_clean"))).drop("tokens_clean")
    
    df_items = df_sp.groupby("item_idx").agg(sf.array(*[sf.mean(sf.col("embedding")[i]) for i in range(300)]).alias("embedding"),
                                                sf.mean("helpfulness").alias("helpfulness"),
                                                sf.mean("score").alias("rating_avg"),
                                                sf.count("score").alias("rating_cnt"))

    df_users = df_sp.groupby("user_idx").agg(sf.array(*[sf.mean(sf.col("embedding")[i]) for i in range(300)]).alias("embedding"),
                                                sf.mean("helpfulness").alias("helpfulness"),
                                                sf.mean("score").alias("rating_avg"),
                                                sf.count("score").alias("rating_cnt"))

    df_rating = df_sp.groupby("user_idx", "item_idx", "timestamp").agg(sf.mean("score").alias("relevance"), sf.count("score").alias("rating_cnt"))
    
    df_items = df_items.select(['item_idx', 'helpfulness', 'rating_avg', 'rating_cnt']+[expr('embedding[' + str(x) + ']') for x in range(0, 300)])
    new_colnames = ['item_idx', 'helpfulness', 'rating_avg', 'rating_cnt'] + ['w2v_' + str(i) for i in range(0, 300)] 
    df_items = df_items.toDF(*new_colnames)

    df_users = df_users.select(['user_idx', 'helpfulness', 'rating_avg', 'rating_cnt']+[expr('embedding[' + str(x) + ']') for x in range(0, 300)])
    new_colnames = ['user_idx', 'helpfulness', 'rating_avg', 'rating_cnt'] + ['w2v_' + str(i) for i in range(0, 300)] 
    df_users = df_users.toDF(*new_colnames)
    
    return [df_users, df_items, df_rating]
