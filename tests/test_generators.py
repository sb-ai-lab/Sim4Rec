import os
import pytest
import numpy as np
import pandas as pd
import pyspark.sql.functions as sf
from pyspark.sql import DataFrame

from simulator.modules import (
    RealDataGenerator,
    SDVDataGenerator,
    CompositeGenerator
)

SEED = 1234

@pytest.fixture(scope="function")
def real_gen() -> RealDataGenerator:
    return RealDataGenerator(label='real', seed=SEED)

@pytest.fixture(scope="function")
def synth_gen() -> SDVDataGenerator:
    return SDVDataGenerator(
        label='synth',
        id_column_name='user_id',
        model_name='gaussiancopula',
        parallelization_level=2,
        device_name='cpu',
        seed=SEED
    )

@pytest.fixture(scope="function")
def comp_gen(real_gen : RealDataGenerator, synth_gen : SDVDataGenerator) -> CompositeGenerator:
    return CompositeGenerator(
        generators=[real_gen, synth_gen],
        label='composite',
        weights=[0.5, 0.5]
    )


def test_realdatagenerator_fit(real_gen : RealDataGenerator, users_df : DataFrame):
    real_gen.fit(users_df)

    assert real_gen._fit_called
    assert real_gen._source_df.count() == users_df.count()

def test_sdvdatagenerator_fit(synth_gen : SDVDataGenerator, users_df : DataFrame):
    synth_gen.fit(users_df)

    assert synth_gen._fit_called
    assert isinstance(synth_gen._model.sample(100), pd.DataFrame)


def test_realdatagenerator_generate(real_gen : RealDataGenerator, users_df : DataFrame):
    real_gen.fit(users_df)

    assert real_gen.generate(5).count() == 5
    assert real_gen._df.count() == 5
    assert real_gen.getDataSize() == 5

def test_sdvdatagenerator_generate(synth_gen : SDVDataGenerator, users_df : DataFrame):
    synth_gen.fit(users_df)

    assert synth_gen.generate(100).count() == 100
    assert synth_gen._df.count() == 100
    assert synth_gen.getDataSize() == 100

def test_compositegenerator_generate(
    real_gen : RealDataGenerator,
    synth_gen : SDVDataGenerator,
    comp_gen : CompositeGenerator,
    users_df : DataFrame
):
    real_gen.fit(users_df)
    synth_gen.fit(users_df)
    comp_gen.generate(10)

    assert real_gen.getDataSize() == 5
    assert synth_gen.getDataSize() == 5

    comp_gen.setWeights([1.0, 0.0])
    comp_gen.generate(5)

    assert real_gen.getDataSize() == 5
    assert synth_gen.getDataSize() == 0

    comp_gen.setWeights([0.0, 1.0])
    comp_gen.generate(10)

    assert real_gen.getDataSize() == 0
    assert synth_gen.getDataSize() == 10


def test_realdatagenerator_sample(real_gen : RealDataGenerator, users_df : DataFrame):
    real_gen.fit(users_df)
    _ = real_gen.generate(5)

    assert real_gen.sample(1.0).count() == 5
    assert real_gen.sample(0.5).count() == 2
    assert real_gen.sample(0.0).count() == 0

def test_sdvdatagenerator_sample(synth_gen : SDVDataGenerator, users_df : DataFrame):
    synth_gen.fit(users_df)
    _ = synth_gen.generate(100)

    assert synth_gen.sample(1.0).count() == 100
    assert synth_gen.sample(0.5).count() == 46
    assert synth_gen.sample(0.0).count() == 0

def test_compositegenerator_sample(
    real_gen : RealDataGenerator,
    synth_gen : SDVDataGenerator,
    comp_gen : CompositeGenerator,
    users_df : DataFrame
):
    real_gen.fit(users_df)
    synth_gen.fit(users_df)
    comp_gen.generate(10)

    assert comp_gen.sample(1.0).count() == 10
    assert comp_gen.sample(0.5).count() == 4
    assert comp_gen.sample(0.0).count() == 0

    df = comp_gen.sample(1.0).toPandas()
    assert df['user_id'].str.startswith('synth').sum() == 5

    comp_gen.setWeights([1.0, 0.0])
    comp_gen.generate(5)
    df = comp_gen.sample(1.0).toPandas()
    assert df['user_id'].str.startswith('synth').sum() == 0

    comp_gen.setWeights([0.0, 1.0])
    comp_gen.generate(10)
    df = comp_gen.sample(1.0).toPandas()
    assert df['user_id'].str.startswith('synth').sum() == 10


def test_realdatagenerator_iterdiff(real_gen : RealDataGenerator, users_df : DataFrame):
    real_gen.fit(users_df)
    generated_1 = real_gen.generate(5).toPandas()
    sampled_1 = real_gen.sample(0.5).toPandas()

    generated_2 = real_gen.generate(5).toPandas()
    sampled_2 = real_gen.sample(0.5).toPandas()

    assert not generated_1.equals(generated_2)
    assert not sampled_1.equals(sampled_2)

def test_sdvdatagenerator_iterdiff(synth_gen : SDVDataGenerator, users_df : DataFrame):
    synth_gen.fit(users_df)

    generated_1 = synth_gen.generate(100).toPandas()
    sampled_1 = synth_gen.sample(0.1).toPandas()

    generated_2 = synth_gen.generate(100).toPandas()
    sampled_2 = synth_gen.sample(0.1).toPandas()

    assert not generated_1.equals(generated_2)
    assert not sampled_1.equals(sampled_2)

def test_compositegenerator_iterdiff(
    real_gen : RealDataGenerator,
    synth_gen : SDVDataGenerator,
    comp_gen : CompositeGenerator,
    users_df : DataFrame
):
    real_gen.fit(users_df)
    synth_gen.fit(users_df)
    comp_gen.generate(10)

    sampled_1 = comp_gen.sample(0.5).toPandas()
    sampled_2 = comp_gen.sample(0.5).toPandas()

    assert not sampled_1.equals(sampled_2)


def test_sdvdatagenerator_partdiff(synth_gen : SDVDataGenerator, users_df : DataFrame):
    synth_gen.fit(users_df)

    generated = synth_gen.generate(100)\
        .drop('user_id')\
        .withColumn('__partition_id', sf.spark_partition_id())
    df_1 = generated.filter(sf.col('__partition_id') == 0)\
                    .drop('__partition_id').toPandas()
    df_2 = generated.filter(sf.col('__partition_id') == 1)\
                    .drop('__partition_id').toPandas()

    assert not df_1.equals(df_2)

def test_sdv_save_load(
    synth_gen : SDVDataGenerator,
    users_df : DataFrame,
    tmp_path
):
    synth_gen.fit(users_df)
    synth_gen.save_model(f'{tmp_path}/generator.pkl')
    
    assert os.path.isfile(f'{tmp_path}/generator.pkl')

    g = SDVDataGenerator.load(f'{tmp_path}/generator.pkl')

    assert g.getLabel() == 'synth'
    assert g._id_col_name == 'user_id'
    assert g._model_name == 'gaussiancopula'
    assert g.getParallelizationLevel() == 2
    assert g.getDevice() == 'cpu'
    assert g.getInitSeed() == 1234
    assert g._fit_called
    assert hasattr(g, '_model')
    assert hasattr(g, '_schema')
