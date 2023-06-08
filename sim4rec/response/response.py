import math
from typing import List
from collections.abc import Iterable

import pyspark.sql.types as st
import pyspark.sql.functions as sf
from pyspark.ml import Transformer, Estimator
from pyspark.ml.param.shared import HasInputCols, HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

from pyspark.sql import DataFrame
from pyspark import keyword_only

from sim4rec.params import (
    HasWeights, HasSeedSequence,
    HasConstantValue, HasClipNegative,
    HasMean, HasStandardDeviation
)


class ActionModelEstimator(Estimator,
                           HasOutputCol,
                           DefaultParamsReadable,
                           DefaultParamsWritable):
    @keyword_only
    def __init__(
        self,
        outputCol : str = None
    ):
        """
        Base class for response estimator

        :param outputCol: Name of the response score column, defaults
            to None
        """

        super().__init__()
        self.setParams(**self._input_kwargs)

    @keyword_only
    def setParams(
        self,
        outputCol : str = None
    ):
        return self._set(**self._input_kwargs)


class ActionModelTransformer(Transformer,
                             HasOutputCol,
                             DefaultParamsReadable,
                             DefaultParamsWritable):
    @keyword_only
    def __init__(
        self,
        outputCol : str = None
    ):
        """
        Base class for response transformer. transform() will be
        used to calculate score based on inputCols, and write it
        to outputCol column

        :param outputCol: Name of the response score column, defaults
            to None
        """

        super().__init__()
        self.setParams(**self._input_kwargs)

    @keyword_only
    def setParams(
        self,
        outputCol : str = None
    ):
        return self._set(**self._input_kwargs)


class BernoulliResponse(ActionModelTransformer,
                        HasInputCol,
                        HasSeedSequence):
    def __init__(
        self,
        inputCol : str = None,
        outputCol : str = None,
        seed : int = None
    ):
        """
        Samples responses from probability column

        :param inputCol: Probability column name. Probability should
            be in range [0; 1]
        :param outputCol: Output column name
        :param seed: Random state seed, defaults to None
        """

        super().__init__(outputCol=outputCol)

        self._set(inputCol=inputCol)
        self.initSeedSequence(seed)

    def _transform(
        self,
        df : DataFrame
    ):
        inputCol = self.getInputCol()
        outputCol = self.getOutputCol()
        seed = self.getNextSeed()

        return df.withColumn(
            outputCol,
            sf.when(sf.rand(seed=seed) <= sf.col(inputCol), 1).otherwise(0)
        )


class NoiseResponse(ActionModelTransformer,
                    HasMean,
                    HasStandardDeviation,
                    HasClipNegative,
                    HasSeedSequence):
    def __init__(
        self,
        mu : float = None,
        sigma : float = None,
        outputCol : str = None,
        clipNegative : bool = True,
        seed : int = None
    ):
        """
        Creates random response sampled from normal distribution

        :param mu: Mean parameter of normal distribution
        :param sigma: Standard deviation parameter of normal distribution
        :param outputCol: Output column name
        :param clip_negative: Whether to make response non-negative,
            defaults to True
        :param seed: Random state seed, defaults to None
        """

        super().__init__(outputCol=outputCol)

        self._set(mean=mu, std=sigma, clipNegative=clipNegative)
        self.initSeedSequence(seed)

    def _transform(
        self,
        df : DataFrame
    ):
        mu = self.getMean()
        sigma = self.getStandardDeviation()
        clip_negative = self.getClipNegative()
        outputCol = self.getOutputCol()
        seed = self.getNextSeed()

        expr = sf.randn(seed=seed) * sigma + mu
        if clip_negative:
            expr = sf.greatest(expr, sf.lit(0))

        return df.withColumn(outputCol, expr)


class ConstantResponse(ActionModelTransformer,
                       HasConstantValue):
    def __init__(
        self,
        value : float = 0.0,
        outputCol : str = None
    ):
        """
        Always returns constant valued response

        :param value: Response value
        :param outputCol: Output column name
        """

        super().__init__(outputCol=outputCol)
        
        self._set(constantValue=value)

    def _transform(
        self,
        df : DataFrame
    ):
        value = self.getConstantValue()
        outputColumn = self.getOutputCol()

        return df.withColumn(outputColumn, sf.lit(value))


class CosineSimilatiry(ActionModelTransformer,
                       HasInputCols):
    def __init__(
        self,
        inputCols : List[str] = None,
        outputCol : str = None
    ):
        """
        Calculates the cosine similarity between two vectors.
        The result is in [0; 1] range

        :param inputCols: Two column names with dense vectors
        :param outputCol: Output column name
        """

        if inputCols is not None and len(inputCols) != 2:
            raise ValueError('There must be two array columns '\
                             'to calculate cosine similarity')

        super().__init__(outputCol=outputCol)
        self._set(inputCols=inputCols)

    def _transform(
        self,
        df : DataFrame
    ):
        inputCols = self.getInputCols()
        outputCol = self.getOutputCol()

        def cosine_similarity(first, second):
            num = first.dot(second)
            den = first.norm(2) * second.norm(2)

            if den == 0:
                return float(0)

            cosine = max(min(num / den, 1.0), -1.0)
            return float(1 - math.acos(cosine) / math.pi)

        cos_udf = sf.udf(cosine_similarity, st.DoubleType())

        return df.withColumn(
            outputCol,
            cos_udf(sf.col(inputCols[0]), sf.col(inputCols[1]))
        )


class ParametricResponseFunction(ActionModelTransformer,
                                 HasInputCols,
                                 HasWeights):
    def __init__(
        self,
        inputCols : List[str] = None,
        outputCol : str = None,
        weights : Iterable = None
    ):
        """
        Calculates response based on the weighted sum of input responses

        :param inputCols: Input responses column names
        :param outputCol: Output column name
        :param weights: Input responses weights
        """

        super().__init__(outputCol=outputCol)
        self._set(inputCols=inputCols, weights=weights)

    def _transform(
        self,
        df : DataFrame
    ):
        inputCols = self.getInputCols()
        outputCol = self.getOutputCol()
        weights = self.getWeights()

        return df.withColumn(
            outputCol,
            sum([
                sf.col(c) * weights[i]
                    for i, c in enumerate(inputCols)
                ])
        )
