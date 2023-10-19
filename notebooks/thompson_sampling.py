from typing import Optional, Set, Union, Iterable
import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
import pandas as pd

from utils import (
# get_unique_entities,
get_top_k
)

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
    
    cached_dfs: Optional[Set] = None
    item_popularity: DataFrame
    
    def __init__(
        self,
        sample: bool = False,
        seed: Optional[int] = None,
    ):
        self.sample = sample
        self.seed = seed
        self.add_cold_items=True
        self.cold_weight = 1
    
    
    def fit(self, log: DataFrame) -> None:
        """
        Fit a recommendation model

        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :return:
        """

        users = log.select("user_idx").distinct()
        items = log.select("item_idx").distinct()
                
        self.fit_users = sf.broadcast(users)
        self.fit_items = sf.broadcast(items)
        self._num_users = self.fit_users.count()
        self._num_items = self.fit_items.count()
        self._user_dim_size = (
            self.fit_users.agg({"user_idx": "max"}).collect()[0][0] + 1
        )
        self._item_dim_size = (
            self.fit_items.agg({"item_idx": "max"}).collect()[0][0] + 1
        )
        
        self._check_relevance(log)

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
        log: DataFrame,
        k: int,
        users: Optional[Union[DataFrame, Iterable]] = None,
        items: Optional[Union[DataFrame, Iterable]] = None,
        filter_seen_items: bool = True
    ) -> Optional[DataFrame]:
        """
        Get recommendations

        :param log: historical log of interactions
            ``[user_idx, item_idx, timestamp, relevance]``
        :param k: number of recommendations for each user
        :param users: users to create recommendations for
            dataframe containing ``[user_idx]`` or ``array-like``;
            if ``None``, recommend to all users from ``log``
        :param items: candidate items for recommendations
            dataframe containing ``[item_idx]`` or ``array-like``;
            if ``None``, take all items from ``log``.
            If it contains new items, ``relevance`` for them will be ``0``.
        :param filter_seen_items: flag to remove seen items from recommendations based on ``log``.
        :param recs_file_path: save recommendations at the given absolute path as parquet file.
            If None, cached and materialized recommendations dataframe  will be returned
        :return: cached recommendation dataframe with columns ``[user_idx, item_idx, relevance]``
            or None if `file_path` is provided
        """
        users = users.select("user_idx").distinct()
        items = items.select("item_idx").distinct()

        recs = self._predict_with_sampling(
                log=log,
                k=k,
                users=users,
                items=items, 
                filter_seen_items=filter_seen_items
        )
        
        recs = get_top_k(
            dataframe=recs,
            partition_by_col=sf.col("user_idx"),
            order_by_col=[sf.col("relevance").desc()],
            k=k
        ).select(
            "user_idx", "item_idx", "relevance"
        )

        if filter_seen_items and log:
            recs = self._filter_seen(recs=recs, log=log, users=users, k=k)


        output = recs.cache()
        output.count()

        self._clear_model_temp_view("filter_seen_users_log")
        self._clear_model_temp_view("filter_seen_num_seen")
        
        return output

    
    @staticmethod
    def _check_relevance(log: DataFrame):

        vals = log.select("relevance").where(
            (sf.col("relevance") != 1) & (sf.col("relevance") != 0)
        )
        if vals.count() > 0:
            raise ValueError("Relevance values in log must be 0 or 1")    

    
    def _predict_with_sampling(
        self,
        log: DataFrame,
        k: int,
        users: DataFrame,
        items: DataFrame,
        filter_seen_items: bool = True
    ) -> DataFrame:
        """
        Randomized prediction for popularity-based models,
        top-k items from `items` are sampled for each user based with
        probability proportional to items' popularity
        """
        selected_item_popularity = self.item_popularity.join(
            items,
            on="item_idx",
            how="right" if self.add_cold_items else "inner",
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
        class_name = self.__class__.__name__

        
        def grouped_map(pandas_df: pd.DataFrame) -> pd.DataFrame:
            user_idx = pandas_df["user_idx"][0]
            cnt = pandas_df["cnt"][0]

            if seed is not None:
                local_rng = default_rng(seed + user_idx)
            else:
                local_rng = default_rng()

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

        if log is not None and filter_seen_items:
            recs = (
                log.select("user_idx", "item_idx")
                .distinct()
                .join(users, how="right", on="user_idx")
                .groupby("user_idx")
                .agg(sf.countDistinct("item_idx").alias("cnt"))
                .selectExpr(
                    "user_idx",
                    f"LEAST(cnt + {k}, {items_pd.shape[0]}) AS cnt",
                )
            )
        else:
            recs = users.withColumn("cnt", sf.lit(min(k, items_pd.shape[0])))


        return recs.groupby("user_idx").applyInPandas(grouped_map, REC_SCHEMA)

    
    def _clear_model_temp_view(self, df_name: str) -> None:
        """
        Temp view to replace will be constructed as
        "id_<python object id>_model_<RePlay model name>_<df_name>"
        """
        full_name = f"id_{id(self)}_model_{str(self)}_{df_name}"
        spark = State().session
        spark.catalog.dropTempView(full_name)
        if self.cached_dfs is not None:
            self.cached_dfs.discard(full_name)


    def _filter_seen(
        self, recs: DataFrame, log: DataFrame, k: int, users: DataFrame
    ):
        """
        Filter seen items (presented in log) out of the users' recommendations.
        For each user return from `k` to `k + number of seen by user` recommendations.
        """
        users_log = log.join(users, on="user_idx")
        self._cache_model_temp_view(users_log, "filter_seen_users_log")
        num_seen = users_log.groupBy("user_idx").agg(
            sf.count("item_idx").alias("seen_count")
        )
        self._cache_model_temp_view(num_seen, "filter_seen_num_seen")

        # count maximal number of items seen by users
        max_seen = 0
        if num_seen.count() > 0:
            max_seen = num_seen.select(sf.max("seen_count")).collect()[0][0]

        # crop recommendations to first k + max_seen items for each user
        recs = recs.withColumn(
            "temp_rank",
            sf.row_number().over(
                Window.partitionBy("user_idx").orderBy(
                    sf.col("relevance").desc()
                )
            ),
        ).filter(sf.col("temp_rank") <= sf.lit(max_seen + k))

        # leave k + number of items seen by user recommendations in recs
        recs = (
            recs.join(num_seen, on="user_idx", how="left")
            .fillna(0)
            .filter(sf.col("temp_rank") <= sf.col("seen_count") + sf.lit(k))
            .drop("temp_rank", "seen_count")
        )

        # filter recommendations presented in interactions log
        recs = recs.join(
            users_log.withColumnRenamed("item_idx", "item")
            .withColumnRenamed("user_idx", "user")
            .select("user", "item"),
            on=(sf.col("user_idx") == sf.col("user"))
            & (sf.col("item_idx") == sf.col("item")),
            how="anti",
        ).drop("user", "item")

        return recs


