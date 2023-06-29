import pandas as pd
from pyspark.sql import SparkSession, DataFrame

from replay.session_handler import State


def pandas_to_spark(
    df: pd.DataFrame,
    schema=None,
    spark_session : SparkSession = None) -> DataFrame:
    """
    Converts pandas DataFrame to spark DataFrame

    :param df: DataFrame to convert
    :param schema: Schema of the dataframe, defaults to None
    :param spark_session: Spark session to use, defaults to None
    :returns: data converted to spark DataFrame
    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError('df must be an instance of pd.DataFrame')

    if len(df) == 0:
        raise ValueError('Dataframe is empty')

    if spark_session is not None:
        spark = spark_session
    else:
        spark = State().session

    return spark.createDataFrame(df, schema=schema)
