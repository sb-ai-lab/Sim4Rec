# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import
import pytest
from pyspark.sql import functions as sf
from pyspark.sql import SparkSession
from datetime import datetime
from notebooks.ThompsonSampling import ThompsonSampling
from sim4rec.utils.session_handler import get_spark_session

from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StructField,
    StructType,
    TimestampType,
)
import pandas as pd


spark = SparkSession.builder.getOrCreate()


LOG_SCHEMA = StructType(
    [
        StructField("user_idx", IntegerType()),
        StructField("item_idx", IntegerType()),
        StructField("timestamp", TimestampType()),
        StructField("relevance", DoubleType()),
    ]
)


REC_SCHEMA = StructType(
    [
        StructField("user_idx", IntegerType()),
        StructField("item_idx", IntegerType()),
        StructField("relevance", DoubleType()),
    ]
)



@pytest.fixture
def preprocessed_log(log):
    return log.withColumn(
        "relevance", sf.when(sf.col("relevance") < 3, 0).otherwise(1)
    )

# @pytest.fixture
# def spark():
#     session = get_spark_session(1, 1)
#     session.sparkContext.setLogLevel("ERROR")
#     return session


@pytest.fixture
def log(spark):
    return spark.createDataFrame(
        data=[
            [0, 0, datetime(2019, 8, 22), 4.0],
            [0, 2, datetime(2019, 8, 23), 3.0],
            [0, 1, datetime(2019, 8, 27), 2.0],
            [1, 3, datetime(2019, 8, 24), 3.0],
            [1, 0, datetime(2019, 8, 25), 4.0],
            [2, 1, datetime(2019, 8, 26), 5.0],
            [2, 0, datetime(2019, 8, 26), 5.0],
            [2, 2, datetime(2019, 8, 26), 3.0],
            [3, 1, datetime(2019, 8, 26), 5.0],
            [3, 0, datetime(2019, 8, 26), 5.0],
            [3, 0, datetime(2019, 8, 26), 1.0],
        ],
        schema=LOG_SCHEMA,
    )


@pytest.fixture
def model():
    model = ThompsonSampling(seed=42)
    return model


@pytest.fixture
def fitted_model(preprocessed_log, model):
    model.fit(preprocessed_log)
    return model


@pytest.fixture
def test_data():
    return spark.createDataFrame([
        (1, 1, 1),
        (1, 2, 0),
        (2, 1, 1),
        (2, 2, 1),
        (2, 3, 1),
        (3, 1, 0),
        (3, 2, 0),
        (3, 3, 1),
        (3, 4, 0),
    ], ["user_idx", "item_idx", "relevance"])

@pytest.fixture
def partial_data():
    return spark.createDataFrame([
        (1, 3, 1),
        (1, 4, 0),
        (2, 1, 1),
        (2, 3, 1),
        (2, 4, 1),
        (3, 1, 1),
        (3, 4, 0),
    ], ["user_idx", "item_idx", "relevance"])


def test_works(preprocessed_log, model):
    model.fit(preprocessed_log)
    model.item_popularity.count()


def test_predict(preprocessed_log, model):
    model.fit(preprocessed_log)
    users = spark.createDataFrame([1, 0], IntegerType()).toDF("user_idx")
    items = spark.createDataFrame([1, 0], IntegerType()).toDF("item_idx")
    recs = model.predict(k=1, users=users, items=items)
    assert recs.count() == 2
    assert (
        recs.select(
            sf.sum(sf.col("user_idx").isin([1, 0]).astype("int"))
        ).collect()[0][0]
        == 2
    )
    

def test_partial_fit(test_data, partial_data):
    model = ThompsonSampling()
    model.fit(test_data)
    item_popularity = model.item_popularity
    item_popularity_ = model.create_item_popularity(partial_data)
    model.partial_fit(partial_data)
    ip_first = item_popularity.toPandas()
    ip_second = item_popularity_.toPandas()
    count = ip_first.loc[ip_first["item_idx"] != 2].reset_index() + ip_second.loc[ip_second["item_idx"] != 2].reset_index()
    
    model_ip = model.item_popularity.toPandas()
    
    pd.testing.assert_series_equal(model_ip[model_ip["item_idx"] != 2].sort_values("item_idx").reset_index().positive, 
                                   count.positive)
    pd.testing.assert_series_equal(model_ip[model_ip["item_idx"] != 2].sort_values("item_idx").reset_index().negative, 
                                   count.negative)


def test_create_item_popularity(test_data):
    model = ThompsonSampling()
    item_popularity = model._create_item_popularity(test_data)
    assert item_popularity.count() > 0
    assert item_popularity.columns == ['item_idx', 'positive', 'negative', 'relevance']