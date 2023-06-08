import pytest
from pyspark.sql import DataFrame

from sim4rec.modules import (
    EncoderEstimator,
    EncoderTransformer
)

SEED = 1234

@pytest.fixture(scope="module")
def estimator(users_df : DataFrame) -> EncoderEstimator:
    return EncoderEstimator(
        inputCols=users_df.columns,
        outputCols=[f'encoded_{i}' for i in range(5)],
        hidden_dim=10,
        lr=0.001,
        batch_size=32,
        num_loader_workers=2,
        max_iter=10,
        device_name='cpu',
        seed=SEED
    )

@pytest.fixture(scope="module")
def transformer(
    users_df : DataFrame,
    estimator : EncoderEstimator
) -> EncoderTransformer:
    return estimator.fit(users_df)


def test_estimator_fit(
    estimator : EncoderEstimator,
    transformer : EncoderTransformer
):
    assert estimator._input_dim == len(estimator.getInputCols())
    assert estimator._latent_dim == len(estimator.getOutputCols())
    assert estimator.getDevice() == transformer.getDevice()
    assert str(next(transformer._encoder.parameters()).device) == transformer.getDevice()

def test_transformer_transform(
    users_df : DataFrame,
    transformer : EncoderTransformer
):
    result = transformer.transform(users_df)

    assert result.count() == users_df.count()
    assert len(result.columns) == 5
    assert set(result.columns) == set([f'encoded_{i}' for i in range(5)])
