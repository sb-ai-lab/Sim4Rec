from typing import Optional, Set, Union, Iterable
import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
import pandas as pd

from utils import get_top_k

from numpy.random import default_rng
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StructField,
    StructType
)

from sim4rec.utils.session_handler import State


REC_SCHEMA = StructType(
    [
        StructField("user_idx", IntegerType()),
        StructField("item_idx", IntegerType()),
        StructField("relevance", DoubleType()),
    ]
)


class ThompsonSampling:
    """
    Thompson Sampling recommender.

    Bandit model with `efficient exploration-exploitation balance
    <https://proceedings.neurips.cc/paper/2011/file/e53a0a2978c28872a4505bdb51db06dc-Paper.pdf>`_.
    The reward probability of each of the K arms is modeled by a Beta distribution
    which is updated after an arm is selected. The initial prior distribution is Beta(1,1).
    """
    
    item_popularity: DataFrame
    
    def __init__(
        self,
        seed: int = 1234,
    ):
        self.seed = seed
    
    
    def fit(self, log: DataFrame) -> None:
        """
        Fit a recommendation model

        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :return:
        """
        
        num_positive = log.filter(
            log.relevance == sf.lit(1)
        ).groupby("item_idx").agg(
            sf.count("relevance").alias("positive")
        )
        num_negative = log.filter(
            log.relevance == sf.lit(0)
        ).groupby("item_idx").agg(
            sf.count("relevance").alias("negative")
        )

        self.item_popularity = num_positive.join(
            num_negative, how="inner", on="item_idx"
        )

        self.item_popularity = self.item_popularity.withColumn(
            "relevance",
            sf.udf(np.random.beta, "double")("positive", "negative")
        ).drop("positive", "negative")
        self.item_popularity.cache().count()
        self.fill = np.random.beta(1, 1)

    
    def predict(
        self,
        k: int,
        users: DataFrame,
        items: DataFrame
    ) -> DataFrame:
        """
        Get recommendations

        :param k: number of recommendations for each user
        :param users: users to create recommendations for
            dataframe containing ``[user_idx]`` or ``array-like``;
            if ``None``, recommend to all users from ``log``
        :param items: candidate items for recommendations
            dataframe containing ``[item_idx]`` or ``array-like``;
            if ``None``, take all items from ``log``.
            If it contains new items, ``relevance`` for them will be ``0``.
        """
        users = users.select("user_idx").distinct()
        items = items.select("item_idx").distinct()

        selected_item_popularity = self.item_popularity.join(
            items,
            on="item_idx",
            how="right"
        ).fillna(value=self.fill, subset=["relevance"])
        
        selected_item_popularity = selected_item_popularity.withColumn(
            "relevance",
            sf.when(sf.col("relevance") == sf.lit(0.0), 0.1**6).otherwise(
                sf.col("relevance")
            ),
        )

        items_pd = selected_item_popularity.withColumn(
            "probability",
            sf.col("relevance")
            / selected_item_popularity.select(sf.sum("relevance")).first()[0],
        ).toPandas()

        if items_pd.shape[0] == 0:
            return State().session.createDataFrame([], REC_SCHEMA)

        seed = self.seed

        def grouped_map(pandas_df: pd.DataFrame) -> pd.DataFrame:
            user_idx = pandas_df["user_idx"][0]
            cnt = pandas_df["cnt"][0]

            local_rng = default_rng(seed + user_idx)

            items_positions = local_rng.choice(
                np.arange(items_pd.shape[0]),
                size=cnt,
                p=items_pd["probability"].values,
                replace=False,
            )

            relevance = items_pd["probability"].values[items_positions]

            return pd.DataFrame(
                {
                    "user_idx": cnt * [user_idx],
                    "item_idx": items_pd["item_idx"].values[items_positions],
                    "relevance": relevance,
                }
            )

        recs = users.withColumn("cnt", sf.lit(min(k, items_pd.shape[0])))
        recs = recs.groupby("user_idx").applyInPandas(grouped_map, REC_SCHEMA)
        recs = get_top_k(
            dataframe=recs,
            partition_by_col=sf.col("user_idx"),
            order_by_col=[sf.col("relevance").desc()],
            k=k
        ).select(
            "user_idx", "item_idx", "relevance"
        )

        return recs  
