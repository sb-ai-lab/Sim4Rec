from abc import ABC
from typing import List, Union, Dict, Optional

import numpy as np
from scipy.stats import kstest
# TL;DR scipy.special is a C library, pylint needs python source code
# https://github.com/pylint-dev/pylint/issues/3703
# pylint: disable=no-name-in-module
from scipy.special import kl_div

import pyspark.sql.functions as sf
from pyspark.sql import DataFrame
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    RegressionEvaluator,
    MulticlassClassificationEvaluator
)
from pyspark.ml import Transformer

from sdv.evaluation import evaluate

from replay.metrics import Metric
from replay.experiment import Experiment


def evaluate_synthetic(
    synth_df : DataFrame,
    real_df : DataFrame
) -> dict:
    """
    Evaluates the quality of synthetic data against real. The following
    metrics will be calculated:

    - LogisticDetection: The metric evaluates how hard it is to distinguish the synthetic
    data from the real data by using a Logistic regression model
    - SVCDetection: The metric evaluates how hard it is to distinguish the synthetic data
    from the real data by using a C-Support Vector Classification model
    - KSTest: This metric uses the two-sample Kolmogorov-Smirnov test to compare
    the distributions of continuous columns using the empirical CDF
    - ContinuousKLDivergence: This approximates the KL divergence by binning the continuous values
    to turn them into categorical values and then computing the relative entropy

    :param synth_df: Synthetic data without any identifiers
    :param real_df: Real data without any identifiers
    :return: Dictionary with metrics on synthetic data quality
    """

    result = evaluate(
        synthetic_data=synth_df.toPandas(),
        real_data=real_df.toPandas(),
        metrics=[
            'LogisticDetection',
            'SVCDetection',
            'KSTest',
            'ContinuousKLDivergence'
        ],
        aggregate=False
    )

    return {
        row['metric'] : row['normalized_score']
        for _, row in result.iterrows()
    }


def ks_test(
    df : DataFrame,
    predCol : str,
    labelCol : str
) -> float:
    """
    Kolmogorov-Smirnov test on two dataframe columns

    :param df: Dataframe with two target columns
    :param predCol: Column name with values to test
    :param labelCol: Column name with values to test against
    :return: Result of KS test
    """

    pdf = df.select(predCol, labelCol).toPandas()
    rvs, cdf = pdf[predCol].values, pdf[labelCol].values

    return kstest(rvs, cdf).statistic


def kl_divergence(
    df : DataFrame,
    predCol : str,
    labelCol : str
) -> float:
    """
    Normalized Kullback–Leibler divergence on two dataframe columns. The normalization is
    as follows:

    .. math::
            \\frac{1}{1 + KL\_div}

    :param df: Dataframe with two target columns
    :param predCol: First column name
    :param labelCol: Second column name
    :return: Result of KL divergence
    """

    pdf = df.select(predCol, labelCol).toPandas()
    predicted, ground_truth = pdf[predCol].values, pdf[labelCol].values

    f_obs, edges = np.histogram(ground_truth)
    f_exp, _ = np.histogram(predicted, bins=edges)

    f_obs = f_obs.flatten() + 1e-5
    f_exp = f_exp.flatten() + 1e-5

    return 1 / (1 + np.sum(kl_div(f_obs, f_exp)))


# pylint: disable=too-few-public-methods
class QualityControlObjective(ABC):
    """
    QualityControlObjective is designed to evaluate the quality of response
    function by calculating the similarity degree between results of the
    model, which was trained on real data and a model, trained with
    simulator. The calculated function is

    .. math::
        1 - KS(predictionCol, labelCol) + DKL_{norm}(predictionCol, labelCol)

        - \\frac{1}{N} \\sum_{n=1}^{N} |QM_{syn}^{i}(recs_{synthetic},
        ground\_truth_{synthetic}) - QM_{real}^{i}(recs_{real}, ground\_truth_{real})|,

    where

    .. math::
        KS = supx||Q(x) - P(x)||\ (i.e.\ KS\ test\ statistic)

        DKL_{norm} = \\frac{1}{1 + DKL}

    The greater value indicates more similarity between models' result
    and lower value shows dissimilarity. As a predicted value for KS test
    and KL divergence it takes the result of `response_function` on a
    pairs from real log and compares the distributions similarity between
    real responses and predicted. For calculating QM from formula above
    the metrics from RePlay library are used. Those take ground truth and
    predicted values for both models and measures how close are metric
    values to each other.
    """
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        userKeyCol : str,
        itemKeyCol : str,
        predictionCol : str,
        labelCol : str,
        relevanceCol : str,
        response_function : Transformer,
        replay_metrics : Optional[Dict[Metric, Union[int, List[int]]]],
    ):
        """
        :param userKeyCol: User identifier column name
        :param itemKeyCol: Item identifier column name
        :param predictionCol: Prediction column name, which `response_function`
            will create
        :param labelCol: Column name with ground truth response values
        :param relevanceCol: Relevance column name for RePlay metrics. For
            ground truth dataframe it should be response score and for
            dataframe with recommendations it should be the predicted relevance
            from recommendation algorithm
        :param response_function: Spark's transformer which predict response
            value
        :param replay_metrics: Dictionary with replay metrics. See
            https://sb-ai-lab.github.io/RePlay/pages/modules/metrics.html for
            infromation about available metrics and their descriptions. The
            dictionary format is the same as in Experiment class of the RePlay
            library. Those metrics will be used as QM in the objective above
        """

        super().__init__()

        self._userKeyCol = userKeyCol
        self._itemKeyCol = itemKeyCol
        self._predictionCol = predictionCol
        self._relevanceCol = relevanceCol
        self._labelCol = labelCol

        self._resp_func = response_function
        self._replay_metrics = replay_metrics

    # pylint: disable=too-many-arguments
    def __call__(
        self,
        test_log : DataFrame,
        user_features : DataFrame,
        item_features : DataFrame,
        real_recs : DataFrame,
        real_ground_truth : DataFrame,
        synthetic_recs : DataFrame,
        synthetic_ground_truth : DataFrame
    ) -> float:
        """
        Calculates the models similarity value. Note, that dataframe with
        recommendations for both synthetic and real data must include only
        users from ground truth dataframe

        :param test_log: Real log dataframe with response values
        :param user_features: Users features dataframe with identifier
        :param item_features: Items features dataframe with identifier
        :param real_recs: Recommendations dataframe from model trained on
            real dataset
        :param real_ground_truth: Real log dataframe with only positive
            responses
        :param synthetic_recs: Recommendations dataframe from model trained
            with simulator
        :param synthetic_ground_truth: Simulator's log dataframe with only
            positive responses
        :return: Function value
        """

        objective = 0

        feature_df = test_log.join(user_features, on=self._userKeyCol, how='left')\
                             .join(item_features, on=self._itemKeyCol, how='left')

        pred_df = self._resp_func.transform(feature_df)
        objective = (1
                     - ks_test(pred_df, self._predictionCol, self._labelCol)
                     + kl_divergence(pred_df, self._predictionCol, self._labelCol))

        metrics_values = []
        for r, t in zip((real_recs, synthetic_recs), (real_ground_truth, synthetic_ground_truth)):
            r = r.select(self._userKeyCol, self._itemKeyCol, self._relevanceCol)\
                 .withColumnRenamed(self._userKeyCol, 'user_idx')\
                 .withColumnRenamed(self._itemKeyCol, 'item_idx')\
                 .withColumnRenamed(self._relevanceCol, 'relevance')
            t = t.select(self._userKeyCol, self._itemKeyCol, self._relevanceCol)\
                 .withColumnRenamed(self._userKeyCol, 'user_idx')\
                 .withColumnRenamed(self._itemKeyCol, 'item_idx')\
                 .withColumnRenamed(self._relevanceCol, 'relevance')

            exp = Experiment(test=t, metrics=self._replay_metrics)
            exp.add_result('_dummy', r)
            metrics_values.append(exp.results.values)

        objective -= np.abs(metrics_values[0] - metrics_values[1]).mean()

        return objective


# pylint: disable=too-few-public-methods
class EvaluateMetrics(ABC):
    """
    Recommendation systems and response function metric evaluator class.
    The class allows you to evaluate the quality of a response function on
    historical data or a recommender system on historical data or based on
    the results of an experiment in a simulator. Provides simultaneous
    calculation of several metrics using metrics from the Spark MLlib and
    RePlay libraries.
    A created instance is callable on a dataframe with ``user_id, item_id,
    predicted relevance/response, true relevance/response`` format, which
    you can usually retrieve from simulators sample_responses() or log data
    with recommendation algorithm scores. In case when the RePlay metrics
    are needed it additionally apply filter on a passed dataframe to take
    only necessary responses (e.g. when response is equal to 1).
    """

    REGRESSION_METRICS = set(['rmse', 'mse', 'r2', 'mae', 'var'])
    MULTICLASS_METRICS = set([
        'f1', 'accuracy', 'weightedPrecision', 'weightedRecall',
        'weightedTruePositiveRate', 'weightedFalsePositiveRate',
        'weightedFMeasure', 'truePositiveRateByLabel', 'falsePositiveRateByLabel',
        'precisionByLabel', 'recallByLabel', 'fMeasureByLabel',
        'logLoss', 'hammingLoss'
    ])
    BINARY_METRICS = set(['areaUnderROC', 'areaUnderPR'])

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        userKeyCol : str,
        itemKeyCol : str,
        predictionCol : str,
        labelCol : str,
        replay_label_filter : float = 1.0,
        replay_metrics : Optional[Dict[Metric, Union[int, List[int]]]] = None,
        mllib_metrics : Optional[Union[str, List[str]]] = None
    ):
        """
        :param userKeyCol: User identifier column name
        :param itemKeyCol: Item identifier column name
        :param predictionCol: Predicted scores column name
        :param labelCol: True label column name
        :param replay_label_filter: RePlay metrics assume that only positive
            responses are presented in ground truth data. All user-item pairs with
            col(labelCol) >= replay_label_filter condition are treated as positive
            responses during RePlay metrics calculation, defaults to 1.0
        :param replay_metrics: Dictionary with replay metrics. See
            https://sb-ai-lab.github.io/RePlay/pages/modules/metrics.html for
            infromation about available metrics and their descriptions. The
            dictionary format is the same as in Experiment class of the RePlay
            library, defaults to None
        :param mllib_metrics: Metrics to calculate from spark's mllib. See
            REGRESSION_METRICS, MULTICLASS_METRICS, BINARY_METRICS for available
            values, defaults to None
        """

        super().__init__()

        self._userKeyCol = userKeyCol
        self._itemKeyCol = itemKeyCol
        self._predictionCol = predictionCol
        self._labelCol = labelCol
        self._filter = sf.col(self._labelCol) >= replay_label_filter

        if isinstance(mllib_metrics, str):
            mllib_metrics = [mllib_metrics]

        if replay_metrics is None:
            replay_metrics = {}
        if mllib_metrics is None:
            mllib_metrics = []

        self._replay_metrics = replay_metrics
        self._mllib_metrics = mllib_metrics

    def __call__(
        self,
        df : DataFrame
    ) -> Dict[str, float]:
        """
        Performs metrics calculations on passed dataframe

        :param df: Spark dataframe with userKeyCol, itemKeyCol, predictionCol
            and labelCol columns
        :return: Dictionary with metrics
        """

        df = df.withColumnRenamed(self._userKeyCol, 'user_idx')\
               .withColumnRenamed(self._itemKeyCol, 'item_idx')

        result = {}

        if len(self._replay_metrics) > 0:
            exp = Experiment(
                test=df.filter(self._filter)
                       .drop(self._predictionCol)
                       .withColumnRenamed(self._labelCol, 'relevance'),
                metrics=self._replay_metrics
            )
            exp.add_result(
                '__dummy',
                df.drop(self._labelCol).withColumnRenamed(self._predictionCol, 'relevance')
            )
            result.update(exp.results.to_dict(orient='records')[0])

        for m in self._mllib_metrics:
            evaluator = self._get_evaluator(m)
            result[m] = evaluator.evaluate(df)

        return result

    def _reg_or_multiclass_params(self):
        return {'predictionCol' : self._predictionCol, 'labelCol' : self._labelCol}

    def _binary_params(self):
        return {'rawPredictionCol' : self._predictionCol, 'labelCol' : self._labelCol}

    def _get_evaluator(self, metric):
        if metric in self.REGRESSION_METRICS:
            return RegressionEvaluator(
                metricName=metric, **self._reg_or_multiclass_params())
        if metric in self.BINARY_METRICS:
            return BinaryClassificationEvaluator(
                metricName=metric, **self._binary_params())
        if metric in self.MULTICLASS_METRICS:
            return MulticlassClassificationEvaluator(
                metricName=metric, **self._reg_or_multiclass_params())

        raise ValueError(f'Non existing metric was passed: {metric}')
