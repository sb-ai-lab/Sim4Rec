import pytest
from pyspark.sql import DataFrame, SparkSession

SEED = 1234

@pytest.fixture(scope="session")
def spark() -> SparkSession:
    return SparkSession.builder\
        .appName('simulator_test')\
        .master('local[4]')\
        .config('spark.sql.shuffle.partitions', '4')\
        .config('spark.default.parallelism', '4')\
        .config('spark.driver.extraJavaOptions', '-XX:+UseG1GC')\
        .config('spark.executor.extraJavaOptions', '-XX:+UseG1GC')\
        .config('spark.sql.autoBroadcastJoinThreshold', '-1')\
        .config('spark.driver.memory', '4g')\
        .getOrCreate()


@pytest.fixture(scope="session")
def users_df(spark: SparkSession) -> DataFrame:
    data = [
        (0, 1.25, -0.75, 0.5),
        (1, 0.2, -1.0, 0.0),
        (2, 0.85, -0.5, -1.5),
        (3, -0.33, -0.33, 0.33),
        (4, 0.1, 0.2, 0.3)
    ]

    return spark.createDataFrame(
        data=data,
        schema=['user_id', 'user_attr_1', 'user_attr_2', 'user_attr_3']
    )


@pytest.fixture(scope="session")
def items_df(spark: SparkSession) -> DataFrame:
    data = [
        (0, 0.45, -0.45, 1.2),
        (1, -0.3, 0.75, 0.25),
        (2, 1.25, -0.75, 0.5),
        (3, -1.0, 0.0, -0.5),
        (4, 0.5, -0.5, -1.0)
    ]

    return spark.createDataFrame(
        data=data,
        schema=['item_id', 'item_attr_1', 'item_attr_2', 'item_attr_3']
    )

@pytest.fixture(scope="session")
def log_df(spark: SparkSession) -> DataFrame:
    data = [
        (0, 2, 1.0, 0),
        (1, 1, 0.0, 0),
        (1, 2, 0.0, 0),
        (2, 0, 1.0, 0),
        (2, 2, 1.0, 0)
    ]
    return spark.createDataFrame(
        data=data,
        schema=['user_id', 'item_id', 'relevance', 'timestamp']
    )
