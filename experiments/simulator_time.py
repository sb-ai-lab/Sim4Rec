import warnings
warnings.filterwarnings("ignore")

import sys
import time
import random

import pandas as pd
import numpy as np

from pyspark.sql import SparkSession
from replay.session_handler import State
from simulator.utils import pandas_to_spark
from simulator.modules import SDVDataGenerator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import PipelineModel
from simulator.response import CosineSimilatiry, BernoulliResponse, NoiseResponse, ParametricResponseFunction
from simulator.utils import VectorElementExtractor

from replay.data_preparator import Indexer
from ucb import UCB

NUM_JOBS = int(sys.argv[1])

SPARK_LOCAL_DIR = '/data/home/anthony/tmp'
CHECKPOINT_DIR = '/data/home/anthony/tmp/checkpoints'
MODELS_PATH = '../bin'

spark = SparkSession.builder\
    .appName('simulator_validation')\
    .master(f'local[{NUM_JOBS}]')\
    .config('spark.sql.shuffle.partitions', f'{NUM_JOBS}')\
    .config('spark.default.parallelism', f'{NUM_JOBS}')\
    .config('spark.driver.extraJavaOptions', '-XX:+UseG1GC')\
    .config('spark.executor.extraJavaOptions', '-XX:+UseG1GC')\
    .config('spark.sql.autoBroadcastJoinThreshold', '-1')\
    .config('spark.driver.memory', '64g')\
    .config('spark.local.dir', SPARK_LOCAL_DIR)\
    .getOrCreate()

State(spark)

users_df = pd.DataFrame(data=np.random.normal(1, 1, size=(10000, 100)), columns=[f'user_attr_{i}' for i in range(100)]) 
items_df = pd.DataFrame(data=np.random.normal(-1, 1, size=(2000, 100)), columns=[f'item_attr_{i}' for i in range(100)]) 
items_df.loc[random.sample(range(2000), 1000)] = np.random.normal(1, 1, size=(1000, 100))
users_df['user_id'] = np.arange(len(users_df))
items_df['item_id'] = np.arange(len(items_df))
history_df_all = pd.DataFrame()
history_df_all['user_id'] = np.random.randint(0, 10000, size=33000)
history_df_all['item_id'] = np.random.randint(0, 2000, size=33000)
history_df_all['relevance'] = 0

users_matrix = users_df.values[history_df_all.values[:, 0], :-1]
items_matrix = items_df.values[history_df_all.values[:, 1], :-1]
dot = np.sum(users_matrix * items_matrix, axis=1)
history_df_all['relevance'] = np.where(dot >= 0.5, 1, 0)
history_df_all = history_df_all.drop_duplicates(subset=['user_id', 'item_id'], ignore_index=True)

history_df_train = history_df_all.iloc[:30000]
history_df_val = history_df_all.iloc[30000:]

users_df = pandas_to_spark(users_df)
items_df = pandas_to_spark(items_df)
history_df_train = pandas_to_spark(history_df_train)
history_df_val = pandas_to_spark(history_df_val)

user_generator = SDVDataGenerator.load(f'{MODELS_PATH}/cycle_scale_users_gen.pkl')
item_generator = SDVDataGenerator.load(f'{MODELS_PATH}/cycle_scale_items_gen.pkl')

user_generator.setDevice('cpu')
item_generator.setDevice('cpu')
user_generator.setParallelizationLevel(NUM_JOBS)
item_generator.setParallelizationLevel(NUM_JOBS)

syn_users = user_generator.generate(1000000).cache()
syn_items = item_generator.generate(10000).cache()

va_users_items = VectorAssembler(
    inputCols=users_df.columns[:-1] + items_df.columns[:-1],
    outputCol='features'
)

lr = LogisticRegression(
    featuresCol='features',
    labelCol='relevance',
    probabilityCol='__lr_prob'
)

vee = VectorElementExtractor(inputCol='__lr_prob', outputCol='__lr_prob', index=1)

lr_train_df = history_df_train\
    .join(users_df, 'user_id', 'left')\
    .join(items_df, 'item_id', 'left')

lr_model = lr.fit(va_users_items.transform(lr_train_df))


va_users = VectorAssembler(
    inputCols=users_df.columns[:-1],
    outputCol='features_usr'
)

va_items = VectorAssembler(
    inputCols=items_df.columns[:-1],
    outputCol='features_itm'
)

cos_sim = CosineSimilatiry(
    inputCols=["features_usr", "features_itm"],
    outputCol="__cos_prob"
)

noise_resp = NoiseResponse(mu=0.5, sigma=0.2, outputCol='__noise_prob', seed=1234)

param_resp = ParametricResponseFunction(
    inputCols=['__lr_prob', '__cos_prob', '__noise_prob'],
    outputCol='__proba',
    weights=[1/3, 1/3, 1/3]
)

br = BernoulliResponse(inputCol='__proba', outputCol='response')

pipeline = PipelineModel(
    stages=[
        va_users_items,
        lr_model,
        vee,
        va_users,
        va_items,
        cos_sim,
        noise_resp,
        param_resp,
        br
    ]
)

from simulator.modules import Simulator, EvaluateMetrics
from replay.metrics import NDCG

sim = Simulator(
    user_gen=user_generator,
    item_gen=item_generator,
    user_key_col='user_id',
    item_key_col='item_id',
    spark_session=spark,
    data_dir=f'{CHECKPOINT_DIR}/cycle_load_test_{NUM_JOBS}',
)

evaluator = EvaluateMetrics(
    userKeyCol='user_id',
    itemKeyCol='item_id',
    predictionCol='relevance',
    labelCol='response',
    replay_label_filter=1.0,
    replay_metrics={NDCG() : 100}
)

indexer = Indexer(user_col='user_id', item_col='item_id')
indexer.fit(users=syn_users, items=syn_items)

ucb = UCB(sample=True)
ucb.fit(log=indexer.transform(history_df_train.limit(1)))

items_replay = indexer.transform(syn_items).cache()

ucb_metrics = []

time_list = []
for i in range(30):
    cycle_time = {}
    iter_start = time.time()

    start = time.time()
    users = sim.sample_users(0.02).cache()
    users.count()
    cycle_time['sample_users_time'] = time.time() - start
    
    start = time.time()
    log = sim.get_log(users)
    if log is not None:
        log = indexer.transform(log).cache()
    else:
        log = indexer.transform(history_df_train.limit(1)).cache()
    log.count()
    cycle_time['get_log_time'] = time.time() - start

    start = time.time()
    recs_ucb = ucb.predict(
        log=log,
        k=100,
        users=indexer.transform(users),
        items=items_replay
    )
    recs_ucb = indexer.inverse_transform(recs_ucb).cache()
    recs_ucb.count()
    cycle_time['model_predict_time'] = time.time() - start

    start = time.time()
    resp_ucb = sim.sample_responses(
        recs_df=recs_ucb,
        user_features=users,
        item_features=syn_items,
        action_models=pipeline
    ).select('user_id', 'item_id', 'relevance', 'response').cache()
    resp_ucb.count()
    cycle_time['sample_responses_time'] = time.time() - start

    start = time.time()
    sim.update_log(resp_ucb, iteration=i)
    cycle_time['update_log_time'] = time.time() - start

    start = time.time()
    ucb_metrics.append(evaluator(resp_ucb))
    cycle_time['metrics_time'] = time.time() - start

    start = time.time()
    ucb._clear_cache()
    ucb_train_log = sim.log.cache()
    cycle_time['log_size'] = ucb_train_log.count()
    ucb.fit(
        log=indexer.transform(
            ucb_train_log\
                .select('user_id', 'item_id', 'response')\
                .withColumnRenamed('response', 'relevance')
        )
    )
    cycle_time['model_train'] = time.time() - start

    users.unpersist()
    if log is not None:
        log.unpersist()
    recs_ucb.unpersist()
    resp_ucb.unpersist()
    ucb_train_log.unpersist()

    cycle_time['iter_time'] = time.time() - iter_start
    cycle_time['iteration'] = i
    cycle_time['num_threads'] = NUM_JOBS

    time_list.append(cycle_time)

    print(f'Iteration {i} ended in {cycle_time["iter_time"]} seconds')

items_replay.unpersist()

import os

if os.path.isfile(f'{MODELS_PATH}/cycle_time.csv'):
    pd.concat([pd.read_csv(f'{MODELS_PATH}/cycle_time.csv'), pd.DataFrame(time_list)]).to_csv(f'{MODELS_PATH}/cycle_time.csv', index=False)
else:
    pd.DataFrame(time_list).to_csv(f'{MODELS_PATH}/cycle_time.csv', index=False)
