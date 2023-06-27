from typing import List

import pandas as pd
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pyspark.sql.types as st
from pyspark.sql import DataFrame
from pyspark.ml import Transformer, Estimator
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.param.shared import HasInputCols, HasOutputCols

from sim4rec.params import HasDevice, HasSeed


class Encoder(torch.nn.Module):
    """
    Encoder layer
    """
    def __init__(
        self,
        input_dim : int,
        hidden_dim : int,
        latent_dim : int
    ):
        super().__init__()

        input_dims = [input_dim, hidden_dim, latent_dim]
        self._layers = torch.nn.ModuleList([
            torch.nn.Linear(_in, _out)
            for _in, _out in zip(input_dims[:-1], input_dims[1:])
        ])

    def forward(self, X):
        """
        Performs forward pass through layer
        """
        X = F.normalize(X, p=2)
        for layer in self._layers[:-1]:
            X = layer(X)
            X = F.leaky_relu(X)

        X = self._layers[-1](X)

        return X


class Decoder(torch.nn.Module):
    """
    Decoder layer
    """
    def __init__(
        self,
        input_dim : int,
        hidden_dim : int,
        latent_dim : int
    ):
        super().__init__()

        input_dims = [latent_dim, hidden_dim, input_dim]
        self._layers = torch.nn.ModuleList([
            torch.nn.Linear(_in, _out)
            for _in, _out in zip(input_dims[:-1], input_dims[1:])
        ])

    def forward(self, X):
        """
        Performs forward pass through layer
        """
        for layer in self._layers[:-1]:
            X = layer(X)
            X = F.leaky_relu(X)

        X = self._layers[-1](X)

        return X


# pylint: disable=too-many-ancestors
class EncoderEstimator(Estimator,
                       HasInputCols,
                       HasOutputCols,
                       HasDevice,
                       HasSeed,
                       DefaultParamsReadable,
                       DefaultParamsWritable):
    """
    Estimator for encoder part of the autoencoder pipeline. Trains
    the encoder to process incoming data into latent representation
    """
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        inputCols : List[str],
        outputCols : List[str],
        hidden_dim : int,
        lr : float,
        batch_size : int,
        num_loader_workers : int,
        max_iter : int = 100,
        device_name : str = 'cpu',
        seed : int = None
    ):
        """
        :param inputCols: Column names to process
        :param outputCols: List of output column names per latent coordinate.
            The length of outputCols will determine the embedding dimension size
        :param hidden_dim: Size of hidden layers
        :param lr: Learning rate
        :param batch_size: Batch size during training process
        :param num_loader_workers: Number of cpus to use for data loader
        :param max_iter: Maximum number of iterations, defaults to 100
        :param device_name: PyTorch device name, defaults to 'cpu'
        """

        super().__init__()

        self._set(inputCols=inputCols, outputCols=outputCols)
        self.setDevice(device_name)
        self.setSeed(seed)

        self._input_dim = len(inputCols)
        self._hidden_dim = hidden_dim
        self._latent_dim = len(outputCols)

        self._lr = lr
        self._batch_size = batch_size
        self._num_loader_workers = num_loader_workers
        self._max_iter = max_iter

    # pylint: disable=too-many-locals, not-callable
    def _fit(
        self,
        dataset : DataFrame
    ):
        inputCols = self.getInputCols()
        outputCols = self.getOutputCols()
        device_name = self.getDevice()
        seed = self.getSeed()
        device = torch.device(self.getDevice())

        X = dataset.select(*inputCols).toPandas().values

        torch.manual_seed(torch.seed() if seed is None else seed)

        train_loader = DataLoader(X, batch_size=self._batch_size,
                                  shuffle=True, num_workers=self._num_loader_workers)

        encoder = Encoder(
            input_dim=self._input_dim,
            hidden_dim=self._hidden_dim,
            latent_dim=self._latent_dim
        )
        decoder = Decoder(
            input_dim=self._input_dim,
            hidden_dim=self._hidden_dim,
            latent_dim=self._latent_dim
        )

        model = torch.nn.Sequential(encoder, decoder).to(torch.device(self.getDevice()))

        optimizer = opt.Adam(model.parameters(), lr=self._lr)
        crit = torch.nn.MSELoss()

        for _ in range(self._max_iter):
            loss = 0
            for X_batch in train_loader:
                X_batch = X_batch.float().to(device)

                optimizer.zero_grad()

                pred = model(X_batch)
                train_loss = crit(pred, X_batch)

                train_loss.backward()
                optimizer.step()

                loss += train_loss.item()

        torch.manual_seed(torch.seed())

        return EncoderTransformer(
            inputCols=inputCols,
            outputCols=outputCols,
            encoder=encoder,
            device_name=device_name
        )


class EncoderTransformer(Transformer,
                         HasInputCols,
                         HasOutputCols,
                         HasDevice,
                         DefaultParamsReadable,
                         DefaultParamsWritable):
    """
    Encoder transformer that transforms incoming columns into latent
    representation. Output data will be appended to dataframe and
    named according to outputCols parameter
    """
    def __init__(
        self,
        inputCols : List[str],
        outputCols : List[str],
        encoder : Encoder,
        device_name : str = 'cpu'
    ):
        """
        :param inputCols: Column names to process
        :param outputCols: List of output column names per latent coordinate.
            The length of outputCols must be equal to embedding dimension of
            a trained encoder
        :param encoder: Trained encoder
        :param device_name: PyTorch device name, defaults to 'cpu'
        """

        super().__init__()

        self._set(inputCols=inputCols, outputCols=outputCols)
        self._encoder = encoder
        self.setDevice(device_name)

    def setDevice(self, value):
        super().setDevice(value)

        self._encoder.to(torch.device(value))

    # pylint: disable=not-callable
    def _transform(
        self,
        dataset : DataFrame
    ):
        inputCols = self.getInputCols()
        outputCols = self.getOutputCols()
        device = torch.device(self.getDevice())

        encoder = self._encoder

        @torch.no_grad()
        def encode(iterator):
            for pdf in iterator:
                X = torch.tensor(pdf.loc[:, inputCols].values).float().to(device)
                yield pd.DataFrame(
                    data=encoder(X).cpu().numpy(),
                    columns=outputCols
                )

        schema = st.StructType(
            [st.StructField(c, st.FloatType()) for c in outputCols]
        )

        return dataset.mapInPandas(encode, schema)
