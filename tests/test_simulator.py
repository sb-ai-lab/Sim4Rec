import os
import shutil
import pytest
import pyspark.sql.functions as sf
from pyspark.sql import DataFrame, SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler

from sim4rec.modules import (
    Simulator,
    RealDataGenerator,
    SDVDataGenerator,
    CompositeGenerator,
    CrossJoinItemEstimator,
    CrossJoinItemTransformer
)
from sim4rec.response import CosineSimilatiry


SEED = 1234

@pytest.fixture(scope="module")
def real_users_gen(users_df : DataFrame) -> RealDataGenerator:
    gen = RealDataGenerator(label='real', seed=SEED)
    gen.fit(users_df)
    gen.generate(5)

    return gen

@pytest.fixture(scope="module")
def synth_users_gen(users_df : DataFrame) -> SDVDataGenerator:
    gen = SDVDataGenerator(
        label='synth',
        id_column_name='user_id',
        model_name='gaussiancopula',
        parallelization_level=2,
        device_name='cpu',
        seed=SEED
    )
    gen.fit(users_df)
    gen.generate(5)

    return gen

@pytest.fixture(scope="module")
def comp_users_gen(
    real_users_gen : RealDataGenerator,
    synth_users_gen : SDVDataGenerator
) -> CompositeGenerator:
    return CompositeGenerator(
        generators=[real_users_gen, synth_users_gen],
        label='composite',
        weights=[0.5, 0.5]
    )

@pytest.fixture(scope="module")
def real_items_gen(items_df : DataFrame) -> RealDataGenerator:
    gen = RealDataGenerator(label='real', seed=SEED)
    gen.fit(items_df)
    gen.generate(5)

    return gen

@pytest.fixture(scope="module")
def selector(items_df : DataFrame) -> CrossJoinItemTransformer:
    estimator = CrossJoinItemEstimator(
        k=3,
        userKeyColumn='user_id',
        itemKeyColumn='item_id',
        seed=SEED
    )
    return estimator.fit(items_df)

@pytest.fixture(scope="module")
def pipeline() -> PipelineModel:
    va_left = VectorAssembler(inputCols=['user_attr_1', 'user_attr_2'], outputCol='__v1')
    va_right = VectorAssembler(inputCols=['item_attr_1', 'item_attr_2'], outputCol='__v2')

    c = CosineSimilatiry(inputCols=['__v1', '__v2'], outputCol='response')

    return PipelineModel(stages=[va_left, va_right, c])

@pytest.fixture(scope="function")
def simulator_empty(
    comp_users_gen : CompositeGenerator,
    real_items_gen : RealDataGenerator,
    spark : SparkSession,
    tmp_path
) -> Simulator:
    shutil.rmtree(str(tmp_path / 'sim_empty'), ignore_errors=True)
    return Simulator(
        user_gen=comp_users_gen,
        item_gen=real_items_gen,
        log_df=None,
        user_key_col='user_id',
        item_key_col='item_id',
        data_dir=str(tmp_path / 'sim_empty'),
        spark_session=spark
    )

@pytest.fixture(scope="function")
def simulator_with_log(
    comp_users_gen : CompositeGenerator,
    real_items_gen : RealDataGenerator,
    log_df : DataFrame,
    spark : SparkSession,
    tmp_path
) -> Simulator:
    shutil.rmtree(str(tmp_path / 'sim_with_log'), ignore_errors=True)
    return Simulator(
        user_gen=comp_users_gen,
        item_gen=real_items_gen,
        log_df=log_df,
        user_key_col='user_id',
        item_key_col='item_id',
        data_dir=str(tmp_path / 'sim_with_log'),
        spark_session=spark
    )


def test_simulator_init(
    simulator_empty : Simulator,
    simulator_with_log : Simulator
):
    assert os.path.isdir(simulator_empty._data_dir)
    assert os.path.isdir(simulator_with_log._data_dir)

    assert simulator_empty._log is None
    assert Simulator.ITER_COLUMN in simulator_with_log._log.columns

    assert simulator_with_log._log.count() == 5
    assert os.path.isdir(f'{simulator_with_log._data_dir}/{simulator_with_log.log_filename}/{Simulator.ITER_COLUMN}=start')

def test_simulator_clearlog(
    simulator_with_log : Simulator
):
    simulator_with_log.clear_log()

    assert simulator_with_log.log is None
    assert simulator_with_log._log_schema is None

def test_simulator_updatelog(
    simulator_empty : Simulator,
    simulator_with_log : Simulator,
    log_df : DataFrame
):
    simulator_empty.update_log(log_df, iteration=0)
    simulator_with_log.update_log(log_df, iteration=0)

    assert simulator_empty.log.count() == 5
    assert simulator_with_log.log.count() == 10

    assert set(simulator_empty.log.toPandas()[Simulator.ITER_COLUMN].unique()) == set([0])
    assert set(simulator_with_log.log.toPandas()[Simulator.ITER_COLUMN].unique()) == set(['0', 'start'])

    assert os.path.isdir(f'{simulator_empty._data_dir}/{simulator_empty.log_filename}/{Simulator.ITER_COLUMN}=0')
    assert os.path.isdir(f'{simulator_with_log._data_dir}/{simulator_with_log.log_filename}/{Simulator.ITER_COLUMN}=start')
    assert os.path.isdir(f'{simulator_with_log._data_dir}/{simulator_with_log.log_filename}/{Simulator.ITER_COLUMN}=0')

def test_simulator_sampleusers(
    simulator_empty : Simulator
):
    sampled1 = simulator_empty.sample_users(0.5)\
        .toPandas().sort_values(['user_id'])
    sampled2 = simulator_empty.sample_users(0.5)\
        .toPandas().sort_values(['user_id'])

    assert not sampled1.equals(sampled2)

    assert len(sampled1) == 2
    assert len(sampled2) == 4

def test_simulator_sampleitems(
    simulator_empty : Simulator
):
    sampled1 = simulator_empty.sample_items(0.5)\
        .toPandas().sort_values(['item_id'])
    sampled2 = simulator_empty.sample_items(0.5)\
        .toPandas().sort_values(['item_id'])

    assert not sampled1.equals(sampled2)

    assert len(sampled1) == 2
    assert len(sampled2) == 2

def test_simulator_getuseritem(
    simulator_with_log : Simulator,
    selector : CrossJoinItemTransformer,
    users_df : DataFrame
):
    users = users_df.filter(sf.col('user_id').isin([0, 1, 2]))
    pairs, log = simulator_with_log.get_user_items(users, selector)

    assert pairs.count() == users.count() * selector._k
    assert log.count() == 5

    assert 'user_id' in pairs.columns
    assert 'item_id' in pairs.columns

    assert set(log.toPandas()['user_id']) == set([0, 1, 2])

def test_simulator_responses(
    simulator_empty : Simulator,
    pipeline : PipelineModel,
    users_df : DataFrame,
    items_df : DataFrame,
    log_df : DataFrame
):
    resp = simulator_empty.sample_responses(
        recs_df=log_df,
        user_features=users_df,
        item_features=items_df,
        action_models=pipeline
    ).drop('__v1', '__v2').toPandas().sort_values(['user_id'])

    assert 'user_id' in resp.columns
    assert 'item_id' in resp.columns
    assert 'response' in resp.columns
    assert len(resp) == log_df.count()
    assert resp['response'].values[0] == 1
