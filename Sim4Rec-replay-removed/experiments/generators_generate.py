import os
import sys
import time
import pandas as pd
import torch
from pyspark.sql import SparkSession
from replay.session_handler import State
from sim4rec.modules import SDVDataGenerator

SPARK_LOCAL_DIR = '/data/home/anthony/tmp'
RESULT_DIR = '../bin'


NUM_JOBS = int(sys.argv[1])
torch.set_num_threads(8)

spark = SparkSession.builder\
    .appName('simulator')\
    .master(f'local[{NUM_JOBS}]')\
    .config('spark.sql.shuffle.partitions', f'{NUM_JOBS}')\
    .config('spark.default.parallelism', f'{NUM_JOBS}')\
    .config('spark.driver.extraJavaOptions', '-XX:+UseG1GC')\
    .config('spark.executor.extraJavaOptions', '-XX:+UseG1GC')\
    .config('spark.sql.autoBroadcastJoinThreshold', '-1')\
    .config('spark.driver.memory', '256g')\
    .config('spark.local.dir', SPARK_LOCAL_DIR)\
    .getOrCreate()

State(spark)

def generate_time(generator, num_samples):
    start = time.time()
    df = generator.generate(num_samples).cache()
    df.count()
    result_time = time.time() - start
    df.unpersist()

    return (generator.getLabel(), result_time, num_samples, NUM_JOBS)

generators = [SDVDataGenerator.load(f'{RESULT_DIR}/genscale_{g}_{10000}.pkl') for g in ['copulagan', 'ctgan', 'gaussiancopula', 'tvae']]
for g in generators:
    g.setParallelizationLevel(NUM_JOBS)

NUM_TEST_SAMPLES = [10, 100, 1000, 10000, 100000, 1000000, 10000000]

result_df = pd.DataFrame(columns=['model_label', 'generate_time', 'num_samples', 'num_threads'])

for g in generators:
    _ = g.generate(100).cache().count()
    for n in NUM_TEST_SAMPLES:
        print(f'Generating with {g.getLabel()} {n} samples')
        result_df.loc[len(result_df)] = generate_time(g, n)

old_df = None
if os.path.isfile(f'{RESULT_DIR}/gens_sample_time.csv'):
    old_df = pd.read_csv(f'{RESULT_DIR}/gens_sample_time.csv')
else:
    old_df = pd.DataFrame(columns=['model_label', 'generate_time', 'num_samples', 'num_threads'])

pd.concat([old_df, result_df], ignore_index=True).to_csv(f'{RESULT_DIR}/gens_sample_time.csv', index=False)
