import pytest
from pyspark.sql import DataFrame

from simulator.modules import (
    CrossJoinItemEstimator,
    CrossJoinItemTransformer
)

SEED = 1234
K = 5

@pytest.fixture(scope="function")
def estimator() -> CrossJoinItemEstimator:
    return CrossJoinItemEstimator(
        k=K,
        userKeyColumn='user_id',
        itemKeyColumn='item_id',
        seed=SEED
    )

@pytest.fixture(scope="function")
def transformer(
    estimator : CrossJoinItemEstimator,
    items_df : DataFrame
) -> CrossJoinItemTransformer:
    return estimator.fit(items_df)


def test_crossjoinestimator_fit(
    transformer : CrossJoinItemTransformer,
    items_df : DataFrame
):
    assert transformer._item_df is not None
    assert transformer._item_df.toPandas().equals(items_df.toPandas())


def test_crossjointransformer_transform(
    transformer : CrossJoinItemTransformer,
    users_df : DataFrame
):
    sample = transformer.transform(users_df).toPandas()
    sample = sample.sort_values(['user_id', 'item_id'])

    assert 'user_id' in sample.columns[0]
    assert 'item_id' in sample.columns[1]
    assert len(sample) == users_df.count() * K
    assert sample.iloc[K, 0] == 1
    assert sample.iloc[K, 1] == 0


def test_crossjointransformer_iterdiff(
    transformer : CrossJoinItemTransformer,
    users_df : DataFrame
):
    sample_1 = transformer.transform(users_df).toPandas()
    sample_2 = transformer.transform(users_df).toPandas()
    sample_1 = sample_1.sort_values(['user_id', 'item_id'])
    sample_2 = sample_2.sort_values(['user_id', 'item_id'])

    assert not sample_1.equals(sample_2)


def test_crossjointransformer_fixedseed(
    transformer : CrossJoinItemTransformer,
    users_df : DataFrame,
    items_df : DataFrame
):
    e = CrossJoinItemEstimator(
        k=K,
        userKeyColumn='user_id',
        itemKeyColumn='item_id',
        seed=SEED
    )
    t = e.fit(items_df)

    sample_1 = transformer.transform(users_df).toPandas()
    sample_2 = t.transform(users_df).toPandas()
    sample_1 = sample_1.sort_values(['user_id', 'item_id'])
    sample_2 = sample_2.sort_values(['user_id', 'item_id'])

    assert sample_1.equals(sample_2)
