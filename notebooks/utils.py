from typing import List, Optional, Union, Iterable

from pyspark.ml.linalg import DenseVector, Vectors, VectorUDT
from pyspark.sql import Column, DataFrame, Window, functions as sf


from IPython.display import clear_output
import matplotlib.pyplot as plt

from sim4rec.utils.session_handler import State


def plot_metric(metrics):
    clear_output(wait=True)
    plt.plot(metrics)
    plt.grid()
    plt.xlabel('iteration')
    plt.ylabel('# of clicks')
    plt.show()


def calc_metric(response_df):
    return (response_df
            .groupBy("user_idx").agg(sf.sum("response").alias("num_positive"))
            .select(sf.mean("num_positive")).collect()[0][0]
           )


def get_top_k(
    dataframe: DataFrame,
    partition_by_col: Column,
    order_by_col: List[Column],
    k: int,
) -> DataFrame:
    """
    Return top ``k`` rows for each entity in ``partition_by_col`` ordered by
    ``order_by_col``.

    >>> from replay.utils.session_handler import State
    >>> spark = State().session
    >>> log = spark.createDataFrame([(1, 2, 1.), (1, 3, 1.), (1, 4, 0.5), (2, 1, 1.)]).toDF("user_id", "item_id", "relevance")
    >>> log.show()
    +-------+-------+---------+
    |user_id|item_id|relevance|
    +-------+-------+---------+
    |      1|      2|      1.0|
    |      1|      3|      1.0|
    |      1|      4|      0.5|
    |      2|      1|      1.0|
    +-------+-------+---------+
    <BLANKLINE>
    >>> get_top_k(dataframe=log,
    ...    partition_by_col=sf.col('user_id'),
    ...    order_by_col=[sf.col('relevance').desc(), sf.col('item_id').desc()],
    ...    k=1).orderBy('user_id').show()
    +-------+-------+---------+
    |user_id|item_id|relevance|
    +-------+-------+---------+
    |      1|      3|      1.0|
    |      2|      1|      1.0|
    +-------+-------+---------+
    <BLANKLINE>

    :param dataframe: spark dataframe to filter
    :param partition_by_col: spark column to partition by
    :param order_by_col: list of spark columns to orted by
    :param k: number of first rows for each entity in ``partition_by_col`` to return
    :return: filtered spark dataframe
    """
    return (
        dataframe.withColumn(
            "temp_rank",
            sf.row_number().over(
                Window.partitionBy(partition_by_col).orderBy(*order_by_col)
            ),
        )
        .filter(sf.col("temp_rank") <= k)
        .drop("temp_rank")
    )
    

# def get_unique_entities(
#     df: Union[Iterable, DataFrame],
#     column: str,
# ) -> DataFrame:
#     """
#     Get unique values from ``df`` and put them into dataframe with column ``column``.
#     :param df: spark dataframe with ``column`` or python iterable
#     :param column: name of a column
#     :return:  spark dataframe with column ``[`column`]``
#     """
#     spark = State().session
#     if isinstance(df, DataFrame):
#         unique = df.select(column).distinct()
#     elif isinstance(df, collections.abc.Iterable):
#         unique = spark.createDataFrame(
#             data=pd.DataFrame(pd.unique(list(df)), columns=[column])
#         )
#     else:
#         raise ValueError(f"Wrong type {type(df)}")
#     return unique
