import pyspark.sql.types as st
import pyspark.sql.functions as sf
from pyspark.sql import DataFrame
from pyspark.ml.feature import Bucketizer, VectorAssembler
from pyspark.ml.clustering import KMeans


USER_PREFIX = 'user_'
ITEM_PREFIX = 'item_'

MOVIELENS_USER_SCHEMA = st.StructType(
    [st.StructField('user_idx', st.IntegerType())] +\
    [st.StructField(f'genre{i}', st.DoubleType()) for i in range(19)] +\
    [st.StructField('rating_avg', st.DoubleType())] +\
    [st.StructField(f'w2v_{i}', st.DoubleType()) for i in range(300)]
)
MOVIELENS_ITEM_SCHEMA = st.StructType(
    [st.StructField('item_idx', st.IntegerType())] +\
    [st.StructField('year', st.IntegerType())] +\
    [st.StructField('rating_avg', st.DoubleType())] +\
    [st.StructField(f'genre{i}', st.DoubleType()) for i in range(19)] +\
    [st.StructField(f'w2v_{i}', st.DoubleType()) for i in range(300)]
)
MOVIELENS_LOG_SCHEMA = st.StructType([
    st.StructField('user_idx', st.IntegerType()),
    st.StructField('item_idx', st.IntegerType()),
    st.StructField('relevance', st.DoubleType()),
    st.StructField('timestamp', st.IntegerType())
])

NETFLIX_USER_SCHEMA = st.StructType(
    [st.StructField('user_idx', st.IntegerType())] +\
    [st.StructField('rating_avg', st.DoubleType())] +\
    [st.StructField(f'w2v_{i}', st.DoubleType()) for i in range(300)] +\
    [st.StructField('rating_cnt', st.IntegerType())]
)
NETFLIX_ITEM_SCHEMA = st.StructType(
    [st.StructField('item_idx', st.IntegerType())] +\
    [st.StructField('rating_avg', st.DoubleType())] +\
    [st.StructField('rating_cnt', st.IntegerType())] +\
    [st.StructField('year', st.IntegerType())] +\
    [st.StructField(f'w2v_{i}', st.DoubleType()) for i in range(300)]
)
NETFLIX_LOG_SCHEMA = st.StructType([
    st.StructField('item_idx', st.IntegerType()),
    st.StructField('user_idx', st.IntegerType()),
    st.StructField('relevance', st.DoubleType()),
    st.StructField('timestamp', st.DoubleType())
])

MOVIELENS_CLUSTER_COLS = [
    'genre0', 'genre1', 'genre2', 'genre3', 'genre4',
    'genre5', 'genre6', 'genre7', 'genre8', 'genre9',
    'genre10', 'genre11', 'genre12', 'genre13', 'genre14',
    'genre15', 'genre16', 'genre17', 'genre18'
]
NETFLIX_STRAT_COL = 'rating_cnt'
NETFLIX_CLUSTER_COLS = [f'w2v_{i}' for i in range(300)]


def read_movielens(base_path, type, spark_session):
    if type not in ['train', 'val', 'test']:
        raise ValueError('Wrong dataset type')

    users = spark_session\
        .read.csv(f'{base_path}/{type}/users.csv', header=True, schema=MOVIELENS_USER_SCHEMA)\
        .withColumnRenamed('user_idx', 'user_id')
    items = spark_session\
        .read.csv(f'{base_path}/{type}/items.csv', header=True, schema=MOVIELENS_ITEM_SCHEMA)\
        .withColumnRenamed('item_idx', 'item_id')
    log = spark_session\
        .read.csv(f'{base_path}/{type}/rating.csv', header=True, schema=MOVIELENS_LOG_SCHEMA)\
        .withColumnRenamed('user_idx', 'user_id')\
        .withColumnRenamed('item_idx', 'item_id')

    log = log\
        .join(users, on='user_id', how='leftsemi')\
        .join(items, on='item_id', how='leftsemi')

    for c in users.columns:
        if not c.startswith('user_'):
            users = users.withColumnRenamed(c, 'user_' + c)
        
    for c in items.columns:
        if not c.startswith('item_'):
            items = items.withColumnRenamed(c, 'item_' + c)

    log = log.withColumn('relevance', sf.when(sf.col('relevance') >= 3, 1).otherwise(0))

    users = users.na.drop()
    items = items.na.drop()
    log = log.na.drop()

    return users, items, log

def read_netflix(base_path, type, spark_session):
    if type not in ['train', 'val', 'test']:
        raise ValueError('Wrong dataset type')

    users = spark_session\
        .read.csv(f'{base_path}/{type}/users.csv', header=True, schema=NETFLIX_USER_SCHEMA)\
        .withColumnRenamed('user_idx', 'user_id')
    items = spark_session\
        .read.csv(f'{base_path}/{type}/items.csv', header=True, schema=NETFLIX_ITEM_SCHEMA)\
        .withColumnRenamed('item_idx', 'item_id')
    log = spark_session\
        .read.csv(f'{base_path}/{type}/rating.csv', header=True, schema=NETFLIX_LOG_SCHEMA)\
        .withColumnRenamed('user_idx', 'user_id')\
        .withColumnRenamed('item_idx', 'item_id')

    log = log\
        .join(users, on='user_id', how='leftsemi')\
        .join(items, on='item_id', how='leftsemi')

    for c in users.columns:
        if not c.startswith('user_'):
            users = users.withColumnRenamed(c, 'user_' + c)
        
    for c in items.columns:
        if not c.startswith('item_'):
            items = items.withColumnRenamed(c, 'item_' + c)

    log = log.withColumn('relevance', sf.when(sf.col('relevance') >= 3, 1).otherwise(0))

    users = users.na.drop()
    items = items.na.drop()
    log = log.na.drop()

    return users, items, log

def read_amazon(base_path, type, spark_session):
    if type not in ['train', 'val', 'test']:
        raise ValueError('Wrong dataset type')

    users = spark_session\
        .read.parquet(f'{base_path}/{type}/users.parquet').withColumnRenamed('user_idx', 'user_id')
    items = spark_session\
        .read.parquet(f'{base_path}/{type}/items.parquet').withColumnRenamed('item_idx', 'item_id')
    log = spark_session\
        .read.parquet(f'{base_path}/{type}/rating.parquet')\
        .withColumnRenamed('user_idx', 'user_id')\
        .withColumnRenamed('item_idx', 'item_id')

    log = log\
        .join(users, on='user_id', how='leftsemi')\
        .join(items, on='item_id', how='leftsemi')

    for c in users.columns:
        if not c.startswith('user_'):
            users = users.withColumnRenamed(c, 'user_' + c)
        
    for c in items.columns:
        if not c.startswith('item_'):
            items = items.withColumnRenamed(c, 'item_' + c)

    log = log.withColumn('relevance', sf.when(sf.col('relevance') >= 3, 1).otherwise(0))

    users = users.na.drop()
    items = items.na.drop()
    log = log.na.drop()

    return users, items, log


def netflix_cluster_users(
    df : DataFrame,
    outputCol : str = 'cluster',
    column_prefix : str = '',
    seed : int = None
):
    cluster_cols = [f'{column_prefix}{c}' for c in NETFLIX_CLUSTER_COLS]
    assembler = VectorAssembler(
        inputCols=cluster_cols, outputCol='__features'
    )
    kmeans = KMeans(
        k=10, featuresCol='__features',
        predictionCol=outputCol, maxIter=300, seed=seed
    )

    df = assembler.transform(df)
    kmeans_model = kmeans.fit(df)
    df = kmeans_model.transform(df)

    return df.drop('__features')

def movielens_cluster_users(
    df : DataFrame,
    outputCol : str = 'cluster',
    column_prefix : str = '',
    seed : int = None
):
    cluster_cols = [f'{column_prefix}{c}' for c in MOVIELENS_CLUSTER_COLS]
    assembler = VectorAssembler(
        inputCols=cluster_cols, outputCol='__features'
    )
    kmeans = KMeans(
        k=10, featuresCol='__features',
        predictionCol=outputCol, maxIter=300, seed=seed
    )

    df = assembler.transform(df)
    kmeans_model = kmeans.fit(df)
    df = kmeans_model.transform(df)

    return df.drop('__features')
