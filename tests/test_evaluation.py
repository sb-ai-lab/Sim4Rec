import pytest

from pyspark.sql import DataFrame, SparkSession

from sim4rec.modules import (
    evaluate_synthetic,
    EvaluateMetrics,
    ks_test,
    kl_divergence,
    QualityControlObjective
)
from sim4rec.response import ConstantResponse

from replay.metrics import NDCG, Precision

@pytest.fixture(scope="function")
def evaluator() -> EvaluateMetrics:
    return EvaluateMetrics(
        userKeyCol='user_id',
        itemKeyCol='item_id',
        predictionCol='relevance',
        labelCol='response',
        replay_label_filter=1.0,
        replay_metrics={NDCG() : [2, 3], Precision() : 2},
        mllib_metrics=['mse', 'f1', 'areaUnderROC']
    )

@pytest.fixture(scope='module')
def objective() -> QualityControlObjective:
    return QualityControlObjective(
        userKeyCol='user_id',
        itemKeyCol='item_id',
        predictionCol='relevance',
        labelCol='response',
        relevanceCol='relevance',
        response_function=ConstantResponse(0.5, 'response'),
        replay_metrics={NDCG() : 3, Precision() : 2},
    )

@pytest.fixture(scope="module")
def response_df(spark : SparkSession) -> DataFrame:
    data = [
        (0, 0, 0.0, 0.0),
        (0, 1, 0.5, 1.0),
        (0, 2, 1.0, 1.0),
        (1, 0, 1.0, 0.0),
        (1, 1, 0.5, 0.0),
        (1, 2, 0.0, 1.0)
    ]
    return spark.createDataFrame(data=data, schema=['user_id', 'item_id', 'relevance', 'response'])

def test_evaluate_metrics(
    evaluator : EvaluateMetrics,
    response_df : DataFrame
):
    result = evaluator(response_df)
    assert 'NDCG@2' in result
    assert 'NDCG@3' in result
    assert 'Precision@2' in result
    assert 'mse' in result
    assert 'f1' in result
    assert 'areaUnderROC' in result

    evaluator._replay_metrics = {}

    result = evaluator(response_df)
    assert 'mse' in result
    assert 'f1' in result
    assert 'areaUnderROC' in result

    evaluator._mllib_metrics = []
    evaluator._replay_metrics = {NDCG() : [2, 3], Precision() : 2}

    result = evaluator(response_df)
    assert 'NDCG@2' in result
    assert 'NDCG@3' in result
    assert 'Precision@2' in result

def test_evaluate_synthetic(
    users_df : DataFrame
):
    import pandas as pd
    pd.options.mode.chained_assignment = None

    result = evaluate_synthetic(
        users_df.sample(0.5).drop('user_id'),
        users_df.sample(0.5).drop('user_id')
    )

    assert result['LogisticDetection'] is not None
    assert result['SVCDetection'] is not None
    assert result['KSTest'] is not None
    assert result['ContinuousKLDivergence'] is not None

def test_kstest(
    users_df : DataFrame
):  
    result = ks_test(
        df=users_df.select('user_attr_1', 'user_attr_2'),
        predCol='user_attr_1',
        labelCol='user_attr_2'
    )

    assert isinstance(result, float)

def test_kldiv(
    users_df : DataFrame
):  
    result = kl_divergence(
        df=users_df.select('user_attr_1', 'user_attr_2'),
        predCol='user_attr_1',
        labelCol='user_attr_2'
    )

    assert isinstance(result, float)

def test_qualitycontrolobjective(
    log_df : DataFrame,
    users_df : DataFrame,
    items_df : DataFrame,
    objective : QualityControlObjective
):
    result = objective(
        test_log=log_df,
        user_features=users_df,
        item_features=items_df,
        real_recs=log_df,
        real_ground_truth=log_df,
        synthetic_recs=log_df,
        synthetic_ground_truth=log_df
    )

    assert isinstance(result, float)
