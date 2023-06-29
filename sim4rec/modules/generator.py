import random
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import torch

import pyspark.sql.types as st
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from sdv.tabular import CopulaGAN, CTGAN, GaussianCopula, TVAE

from replay.session_handler import State
from sim4rec.params import (
    HasLabel, HasDevice, HasDataSize,
    HasParallelizationLevel, HasSeedSequence, HasWeights
)
from sim4rec.utils import (
    NotFittedError, EmptyDataFrameError, save, load
)


class GeneratorBase(ABC, HasLabel, HasDataSize, HasSeedSequence):
    """
    Base class for data generators
    """
    def __init__(
        self,
        label : str,
        seed : int = None
    ):
        """
        :param label: Generator string label
        :param seed: Fixes seed sequence to use during multiple
            generator calls, defaults to None
        """
        super().__init__()
        self.setLabel(label)
        self.setDataSize(0)
        self.initSeedSequence(seed)

        self._fit_called = False
        self._df = None

    def fit(
        self,
        df : DataFrame
    ):
        """
        Fits generator on passed dataframe

        :param df: Source dataframe to fit on
        """

        raise NotImplementedError()

    @abstractmethod
    def generate(
        self,
        num_samples : int
    ):
        """
        Generates num_samples from fitted model or saved dataframe

        :param num_samples: Number of samples to generate
        """
        raise NotImplementedError()

    def sample(
        self,
        sample_frac : float
    ) -> DataFrame:
        """
        Samples a fraction of rows from a dataframe, generated with
        generate() call

        :param sample_frac: Fraction of rows
        :returns: Sampled dataframe
        """

        if self._df is None:
            raise EmptyDataFrameError(
                'Dataframe is empty. Maybe the generate() was never called?'
            )

        seed = self.getNextSeed()

        return self._df.sample(sample_frac, seed=seed)


class RealDataGenerator(GeneratorBase):
    """
    Real data generator, which can sample from existing dataframe
    """
    _source_df : DataFrame

    def fit(
        self,
        df : DataFrame
    ) -> None:
        """
        :param df: Dataframe for generation and sampling
        """
        if self._fit_called:
            self._source_df.unpersist()

        self._source_df = df.cache()
        self._fit_called = True

    def generate(
        self,
        num_samples : int
    ) -> DataFrame:
        """
        Generates a number of samples from fitted dataframe
        and keeps it for sampling

        :param num_samples: Number of samples to generate
        :returns: Generated dataframe
        """

        if not self._fit_called:
            raise NotFittedError()

        source_size = self._source_df.count()

        if num_samples > source_size:
            raise ValueError('Not enough samples in fitted dataframe')

        seed = self.getNextSeed()

        if self._df is not None:
            self._df.unpersist()

        self._df = self._source_df.orderBy(sf.rand(seed=seed))\
                                  .limit(num_samples)\
                                  .cache()
        self.setDataSize(self._df.count())

        return self._df


def set_sdv_seed(seed : int = None):
    """
    Fixes seed for SDV
    """
    # this is the only way to fix seed in SDV library
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(torch.seed() if seed is None else seed)


# pylint: disable=too-many-ancestors
class SDVDataGenerator(GeneratorBase, HasParallelizationLevel, HasDevice):
    """
    Synthetic data generator with a bunch of models from SDV library
    """
    _model : Optional[Union[CopulaGAN, CTGAN, GaussianCopula, TVAE]] = None

    SEED_COLUMN_NAME = '__seed'

    _sdv_model_dict = {
        'copulagan' : CopulaGAN,
        'ctgan' : CTGAN,
        'gaussiancopula' : GaussianCopula,
        'tvae' : TVAE
    }

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        label : str,
        id_column_name : str,
        model_name : str = 'gaussiancopula',
        parallelization_level : int = 1,
        device_name : str = 'cpu',
        seed : int = None
    ):
        """
        :param label: Generator string label
        :param id_column_name: Column name for identifier
        :param model_name: Name of a SDV model. Possible values are:
            ['copulagan', 'ctgan', 'gaussiancopula', 'tvae'],
            defaults to 'gaussiancopula'
        :param parallelization_level: Parallelization level, defaults to 1
        :param device_name: PyTorch device name, defaults to 'cpu'
        :param seed: Fixes seed sequence to use during multiple
            generator calls, defaults to None
        """

        super().__init__(label=label, seed=seed)

        self.setParallelizationLevel(parallelization_level)
        self._id_col_name = id_column_name
        self._model_name = model_name
        self.setDevice(device_name)
        self._schema = None

    def fit(
        self,
        df : DataFrame
    ) -> None:
        """
        Fits a generation model with a passed dataframe. The one should
        pass only feature columns

        :param df: Dataframe to fit on
        """

        model_params = {'cuda' : self.getDevice()}\
            if self._model_name != 'gaussiancopula' else {}

        self._model = self._sdv_model_dict[self._model_name](**model_params)

        if self._id_col_name in df.columns:
            df = df.drop(self._id_col_name)

        self._schema = st.StructType(
            [
                st.StructField(self._id_col_name, st.StringType())
            ] + df.schema.fields
        )

        self._model.fit(df.toPandas())
        self._fit_called = True

    def setDevice(
        self,
        value : str
    ) -> None:
        """
        Changes the current device. Note, that for gaussiancopula
        model, only cpu is supported

        :param device_name: PyTorch device name
        """

        super().setDevice(value)

        if self._model_name != 'gaussiancopula' and self._fit_called:
            self._model._model.set_device(torch.device(value))

    def generate(
        self,
        num_samples : int
    ) -> DataFrame:
        """
        Generates a number of samples from fitted dataframe
        and keeps it for sampling

        :param num_samples: Number of samples to generate
        :returns: Generated dataframe
        """

        if not self._fit_called:
            raise NotFittedError('Fit was never called')

        if num_samples < 0:
            raise ValueError('num_samples must be non-negative value')

        if self._df is not None:
            self._df.unpersist()

        label = self.getLabel()
        pl = self.getParallelizationLevel()
        seed = self.getNextSeed()

        result_df = State().session.range(
            start=0, end=num_samples, numPartitions=pl
        )
        result_df = result_df\
            .withColumnRenamed('id', self._id_col_name)\
            .withColumn(
                self._id_col_name,
                sf.concat(sf.lit(f'{label}_'), sf.col(self._id_col_name))
            )\
            .withColumn(self.SEED_COLUMN_NAME, sf.spark_partition_id() + sf.lit(seed))

        model = self._model
        seed_col = self.SEED_COLUMN_NAME

        def generate_pdf(iterator):
            for pdf in iterator:
                seed = hash(pdf[seed_col][0]) & 0xffffffff
                set_sdv_seed(seed)

                sampled_df = model.sample(len(pdf), output_file_path='disable')
                yield pd.concat([pdf, sampled_df], axis=1)

        set_sdv_seed()

        result_df = result_df.mapInPandas(generate_pdf, self._schema)

        self._df = result_df.cache()
        self.setDataSize(self._df.count())

        return self._df

    def save_model(
        self,
        filename : str
    ):
        """
        Saves generator model to file. Note, that it saves only
        fitted model, but not the generated dataframe

        :param filename: Path to the file
        """

        if not self._fit_called:
            raise NotFittedError('Fit was never called')

        save_device = self.getDevice()
        self.setDevice('cpu')

        generator_data = (
            self.getLabel(),
            self._id_col_name,
            self._model_name,
            self.getParallelizationLevel(),
            save_device,
            self.getInitSeed(),
            self._model,
            self._schema
        )

        save(generator_data, filename)

    @staticmethod
    def load(filename : str):
        """
        Loads the generator model from the file

        :param filename: Path to the file
        :return: Generator instance with restored model
        """

        label, id_col_name, model_name, p_level,\
            device_name, init_seed, model, schema = load(filename)

        generator = SDVDataGenerator(
            label=label,
            id_column_name=id_col_name,
            model_name=model_name,
            parallelization_level=p_level,
            device_name=device_name,
            seed=init_seed
        )

        generator._model = model
        generator._fit_called = True
        generator._schema = schema

        try:
            generator.setDevice(device_name)
        except RuntimeError:
            print(f'Cannot load model to device {device_name}. Setting cpu instead')
            generator.setDevice('cpu')

        return generator


# pylint: disable=too-many-ancestors
class CompositeGenerator(GeneratorBase, HasWeights):
    """
    Wrapper for sampling from multiple generators. Use weights
    parameter to control the sampling fraction for each of the
    generator
    """
    def __init__(
        self,
        generators : List[GeneratorBase],
        label : str,
        weights : Iterable = None,
    ):
        """
        :param generators: List of generators
        :param label: Generator string label
        :param weights: Weights for each of the generator. Weights
            must be normalized (sums to 1), defaults to None
        """

        super().__init__(label=label)

        self._generators = generators
        data_sizes = [g.getDataSize() for g in self._generators]
        data_sizes_sum = sum(data_sizes)
        self.setDataSize(data_sizes_sum)

        if weights is None and data_sizes_sum != 0:
            if data_sizes_sum > 0:
                weights = [s / data_sizes_sum for s in data_sizes]
            else:
                n = len(self._generators)
                weights = [1 / n] * n

        self.setWeights(weights)

    def generate(
        self,
        num_samples: int
    ) -> None:
        """
        For each generator calls generate() with number of samples,
        proportional to weights to generate num_samples in total. You
        can call this method to not perform generate() separately on
        each generator

        :param num_samples: Total number of samples to generate
        """

        weights = self.getWeights()
        num_required_samples = [round(num_samples * w) for w in weights]

        for g, n in zip(self._generators, num_required_samples):
            _ = g.generate(n)

        self.setDataSize(sum([g.getDataSize() for g in self._generators]))

    def sample(
        self,
        sample_frac : float
    ) -> DataFrame:
        """
        Samples a fraction of rows from generators according to the weights.

        :param sample_frac: Fraction of rows
        :returns: Sampled dataframe
        """

        weights = self.getWeights()

        data_sizes = [g.getDataSize() for g in self._generators]
        data_sizes_sum = sum(data_sizes)

        num_required_samples = [int(data_sizes_sum * sample_frac * w) for w in weights]

        for i in range(len(data_sizes)):
            if num_required_samples[i] > data_sizes[i]:
                raise ValueError(
                    f'Not enough samples in generator {self._generators[i].getLabel()}'
                )

        generator_fracs = []
        for n, s in zip(num_required_samples, data_sizes):
            generator_fracs.append(0.0 if s == 0 else n / s)

        result_df = self._generators[0].sample(generator_fracs[0])

        for g, f in zip(self._generators[1:], generator_fracs[1:]):
            result_df = result_df.unionByName(g.sample(f))

        return result_df
