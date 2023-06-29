import pytest
import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import DenseVector

from sim4rec.response import (
    BernoulliResponse,
    NoiseResponse,
    ConstantResponse,
    CosineSimilatiry,
    ParametricResponseFunction
)

SEED = 1234


@pytest.fixture(scope="function")
def users_va_left() -> VectorAssembler:
    return VectorAssembler(
        inputCols=[f'user_attr_{i}' for i in range(0, 5)],
        outputCol='__v1'
    )


@pytest.fixture(scope="function")
def users_va_right() -> VectorAssembler:
    return VectorAssembler(
        inputCols=[f'user_attr_{i}' for i in range(5, 10)],
        outputCol='__v2'
    )


@pytest.fixture(scope="function")
def bernoulli_resp() -> BernoulliResponse:
    return BernoulliResponse(
        inputCol='__proba',
        outputCol='relevance',
        seed=SEED
    )


@pytest.fixture(scope="function")
def noise_resp() -> NoiseResponse:
    return NoiseResponse(
        mu=0.5,
        sigma=0.2,
        outputCol='__noise',
        seed=SEED
    )


@pytest.fixture(scope="function")
def const_resp() -> ConstantResponse:
    return ConstantResponse(
        value=0.5,
        outputCol='__const',
    )


@pytest.fixture(scope="function")
def cosine_resp() -> CosineSimilatiry:
    return CosineSimilatiry(
        inputCols=['__v1', '__v2'],
        outputCol='__cosine'
    )


@pytest.fixture(scope="function")
def param_resp() -> ParametricResponseFunction:
    return ParametricResponseFunction(
        inputCols=['__const', '__cosine'],
        outputCol='__proba',
        weights=[0.5, 0.5]
    )


@pytest.fixture(scope="module")
def random_df(spark : SparkSession) -> DataFrame:
    data = [
        (0, 0.0),
        (1, 0.2),
        (2, 0.4),
        (3, 0.6),
        (4, 1.0)
    ]
    return spark.createDataFrame(data=data, schema=['id', '__proba'])


@pytest.fixture(scope="module")
def vector_df(spark : SparkSession) -> DataFrame:
    data = [
        (0, DenseVector([1.0, 0.0]), DenseVector([0.0, 1.0])),
        (1, DenseVector([-1.0, 0.0]), DenseVector([1.0, 0.0])),
        (2, DenseVector([1.0, 0.0]), DenseVector([1.0, 0.0])),
        (3, DenseVector([0.5, 0.5]), DenseVector([-0.5, 0.5])),
        (4, DenseVector([0.0, 0.0]), DenseVector([1.0, 0.0]))
    ]
    return spark.createDataFrame(data=data, schema=['id', '__v1', '__v2'])


def test_bernoulli_transform(
    bernoulli_resp : BernoulliResponse,
    random_df : DataFrame
):
    result = bernoulli_resp.transform(random_df).toPandas().sort_values(['id'])

    assert 'relevance' in result.columns
    assert len(result) == 5
    assert set(result['relevance']) == set([0, 1])
    assert list(result['relevance'][:5]) == [0, 0, 0, 1, 1]


def test_bernoulli_iterdiff(
    bernoulli_resp : BernoulliResponse,
    random_df : DataFrame
):
    result1 = bernoulli_resp.transform(random_df).toPandas()
    result1 = result1.sort_values(['id'])
    result2 = bernoulli_resp.transform(random_df).toPandas()
    result2 = result2.sort_values(['id'])

    assert list(result1['relevance'][:5]) != list(result2['relevance'][:5])


def test_noise_transform(
    noise_resp : NoiseResponse,
    random_df : DataFrame
):
    result = noise_resp.transform(random_df).toPandas().sort_values(['id'])

    assert '__noise' in result.columns
    assert len(result) == 5
    assert np.allclose(result['__noise'][0], 0.6117798825975235)


def test_noise_iterdiff(
    noise_resp : NoiseResponse,
    random_df : DataFrame
):
    result1 = noise_resp.transform(random_df).toPandas().sort_values(['id'])
    result2 = noise_resp.transform(random_df).toPandas().sort_values(['id'])

    assert result1['__noise'][0] != result2['__noise'][0]


def test_const_transform(
    const_resp : ConstantResponse,
    random_df : DataFrame
):
    result = const_resp.transform(random_df).toPandas().sort_values(['id'])

    assert '__const' in result.columns
    assert len(result) == 5
    assert list(result['__const']) == [0.5] * 5


def test_cosine_transform(
    cosine_resp : CosineSimilatiry,
    vector_df : DataFrame
):
    result = cosine_resp.transform(vector_df)\
        .drop('__v1', '__v2')\
        .toPandas()\
        .sort_values(['id'])

    assert '__cosine' in result.columns
    assert len(result) == 5
    assert list(result['__cosine']) == [0.5, 0.0, 1.0, 0.5, 0.0]


def test_paramresp_transform(
    param_resp : ParametricResponseFunction,
    const_resp : ConstantResponse,
    cosine_resp : CosineSimilatiry,
    vector_df : DataFrame
):
    df = const_resp.transform(vector_df)
    df = cosine_resp.transform(df)

    result = param_resp.transform(df)\
        .drop('__v1', '__v2')\
        .toPandas()\
        .sort_values(['id'])

    assert '__proba' in result.columns
    assert len(result) == 5
    assert list(result['__proba']) == [0.5, 0.25, 0.75, 0.5, 0.25]

    r = result['__const'][0] / 2 +\
        result['__cosine'][0] / 2
    assert result['__proba'][0] == r

    param_resp.setWeights([1.0, 0.0])
    result = param_resp.transform(df)\
        .drop('__v1', '__v2')\
        .toPandas()\
        .sort_values(['id'])
    assert result['__proba'][0] == result['__const'][0]

    param_resp.setWeights([0.0, 1.0])
    result = param_resp.transform(df)\
        .drop('__v1', '__v2')\
        .toPandas()\
        .sort_values(['id'])
    assert result['__proba'][0] == result['__cosine'][0]
