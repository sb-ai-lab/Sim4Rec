from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

import pyspark.sql.functions as sf

from sim4rec.response import BernoulliResponse,  ActionModelTransformer

class PopBasedTransformer(ActionModelTransformer):
    def __init__(
            self,
            spark: SparkSession,
            outputCol: str = None,
            pop_df_path: str = None,
    ):
        """
        :param outputCol: Name of the response probability column
        :param pop_df_path: path to a spark dataframe with items' popularity
        """
        self.pop_df = sf.broadcast(spark.read.parquet(pop_df_path))
        self.outputCol = outputCol

    def _transform(self, dataframe):
        return (dataframe
                .join(self.pop_df, on='item_idx')
                .drop(*set(self.pop_df.columns).difference(["item_idx", self.outputCol]))
                )


class ResponseModel:
    def __init__(self, spark, pop_df_path="./data/popular_items_popularity.parquet", seed=123):
        pop_resp = PopBasedTransformer(spark=spark, outputCol="popularity", pop_df_path=pop_df_path)
        br = BernoulliResponse(seed=seed, inputCol='popularity', outputCol='response')
        self.model = PipelineModel(
            stages=[pop_resp, br])

    def transform(self, df):
        return self.model.transform(df).drop("popularity")


if __name__ == '__main__':
    import pandas as pd
    num_threads = 4
    spark = SparkSession.builder \
        .appName('simulator') \
        .master(f'local[{num_threads}]') \
        .config('spark.sql.shuffle.partitions', f'{num_threads * 3}') \
        .config('spark.default.parallelism', f'{num_threads * 3}') \
        .config('spark.driver.extraJavaOptions', '-XX:+UseG1GC') \
        .config('spark.executor.extraJavaOptions', '-XX:+UseG1GC') \
        .getOrCreate()

    task = TaskOneResponse(spark, seed=123)
    task.model.stages[0].pop_df.show()
    test_df = spark.createDataFrame(pd.DataFrame({"item_idx": [3, 5, 1], "user_idx": [5, 2, 1]}))
    task.transform(test_df).show()
    task.transform(test_df).show()
    task.transform(test_df).show()
    