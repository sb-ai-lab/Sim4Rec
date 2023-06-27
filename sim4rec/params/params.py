import sys
import numpy as np
from pyspark.ml.param.shared import Params, Param, TypeConverters


class HasUserKeyColumn(Params):
    """
    Controls user identifier column name
    """

    userKeyColumn = Param(
        Params._dummy(),
        "userKeyColumn",
        "User identifier column name",
        typeConverter=TypeConverters.toString
    )

    def setUserKeyColumn(self, value):
        """
        Sets user indentifier column name

        :param value: new column name
        """
        return self._set(userKeyColumn=value)

    def getUserKeyColumn(self):
        """
        Returns item indentifier column name
        """
        return self.getOrDefault(self.userKeyColumn)


class HasItemKeyColumn(Params):
    """
    Controls item identifier column name
    """

    itemKeyColumn = Param(
        Params._dummy(),
        "itemKeyColumn",
        "Item identifier column name",
        typeConverter=TypeConverters.toString
    )

    def setItemKeyColumn(self, value):
        """
        Sets item indentifier column name

        :param value: new column name
        """
        return self._set(itemKeyColumn=value)

    def getItemKeyColumn(self):
        """
        Returns item indentifier column name
        """
        return self.getOrDefault(self.itemKeyColumn)


class HasSeed(Params):
    """
    Controls random state seed
    """

    seed = Param(
        Params._dummy(),
        "seed",
        "Random state seed",
        typeConverter=TypeConverters.toInt
    )

    def setSeed(self, value):
        """
        Changes random state seed

        :param value: new random state seed
        """
        return self._set(seed=value)

    def getSeed(self):
        """
        Returns state seed
        """
        return self.getOrDefault(self.seed)


class HasSeedSequence(Params):
    """
    Controls random state seed of sequence
    """
    _rng : np.random.Generator

    current_seed = Param(
        Params._dummy(),
        "current_seed",
        "Random state seed sequence",
        typeConverter=TypeConverters.toInt
    )

    init_seed = Param(
        Params._dummy(),
        "init_seed",
        "Sequence initial seed",
        typeConverter=TypeConverters.toInt
    )

    def initSeedSequence(self, value):
        """
        Sets initial random state seed of sequence

        :param value: new initial random state seed of sequence
        """
        self._rng = np.random.default_rng(value)
        return self._set(
            init_seed=value if value is not None else -1,
            current_seed=self._rng.integers(0, sys.maxsize)
        )

    def getInitSeed(self):
        """
        Returns initial random state seed of sequence
        """
        value = self.getOrDefault(self.init_seed)
        return None if value == -1 else value

    def getNextSeed(self):
        """
        Returns current random state seed of sequence
        """
        seed = self.getOrDefault(self.current_seed)
        self._set(current_seed=self._rng.integers(0, sys.maxsize))
        return seed


class HasWeights(Params):
    """
    Controls weights for models ensemble
    """

    weights = Param(
        Params._dummy(),
        "weights",
        "Weights for models ensemble",
        typeConverter=TypeConverters.toListFloat
    )

    def setWeights(self, value):
        """
        Changes weights for models ensemble

        :param value: new weights
        """
        return self._set(weights=value)

    def getWeights(self):
        """
        Returns weigths for models ensemble
        """
        return self.getOrDefault(self.weights)


class HasMean(Params):
    """
    Controls mean parameter of normal distribution
    """

    mean = Param(
        Params._dummy(),
        "mean",
        "Mean parameter of normal distribution",
        typeConverter=TypeConverters.toFloat
    )

    def setMean(self, value):
        """
        Changes mean parameter of normal distribution

        :param value: new value of mean parameter
        """
        return self._set(mean=value)

    def getMean(self):
        """
        Returns mean parameter
        """
        return self.getOrDefault(self.mean)


class HasStandardDeviation(Params):
    """
    Controls Standard Deviation parameter of normal distribution
    """

    std = Param(
        Params._dummy(),
        "std",
        "Standard Deviation parameter of normal distribution",
        typeConverter=TypeConverters.toFloat
    )

    def setStandardDeviation(self, value):
        """
        Changes Standard Deviation parameter of normal distribution

        :param value: new value of std parameter
        """

        return self._set(std=value)

    def getStandardDeviation(self):
        """
        Returns value of std parameter
        """
        return self.getOrDefault(self.std)


class HasClipNegative(Params):
    """
    Controls flag that controls clipping of negative values
    """

    clipNegative = Param(
        Params._dummy(),
        "clipNegative",
        "Boolean flag to clip negative values",
        typeConverter=TypeConverters.toBoolean
    )

    def setClipNegative(self, value):
        """
        Changes flag that controls clipping of negative values

        :param value: New value of flag
        """
        return self._set(clipNegative=value)

    def getClipNegative(self):
        """
        Returns flag that controls clipping of negative values
        """
        return self.getOrDefault(self.clipNegative)


class HasConstantValue(Params):
    """
    Controls constant value parameter
    """

    constantValue = Param(
        Params._dummy(),
        "constantValue",
        "Constant value parameter",
        typeConverter=TypeConverters.toFloat
    )

    def setConstantValue(self, value):
        """
        Sets constant value parameter

        :param value: Value
        """
        return self._set(constantValue=value)

    def getConstantValue(self):
        """
        Returns constant value
        """
        return self.getOrDefault(self.constantValue)


class HasLabel(Params):
    """
    Controls string label
    """
    label = Param(
        Params._dummy(),
        "label",
        "String label",
        typeConverter=TypeConverters.toString
    )

    def setLabel(self, value):
        """
        Sets string label

        :param value: Label
        """
        return self._set(label=value)

    def getLabel(self):
        """
        Returns current string label
        """
        return self.getOrDefault(self.label)


class HasDevice(Params):
    """
    Controls device
    """
    device = Param(
        Params._dummy(),
        "device",
        "Name of a device to use",
        typeConverter=TypeConverters.toString
    )

    def setDevice(self, value):
        """
        Sets device

        :param value: Name of device to use
        """
        return self._set(device=value)

    def getDevice(self):
        """
        Returns current device
        """
        return self.getOrDefault(self.device)


class HasDataSize(Params):
    """
    Controls data size
    """
    data_size = Param(
        Params._dummy(),
        "data_size",
        "Size of a DataFrame",
        typeConverter=TypeConverters.toInt
    )

    def setDataSize(self, value):
        """
        Sets data size to a certain value

        :param value: Size of a DataFrame
        """
        return self._set(data_size=value)

    def getDataSize(self):
        """
        Returns current size of a DataFrame
        """
        return self.getOrDefault(self.data_size)


class HasParallelizationLevel(Params):
    """
    Controls parallelization level
    """
    parallelizationLevel = Param(
        Params._dummy(),
        "parallelizationLevel",
        "Level of parallelization",
        typeConverter=TypeConverters.toInt
    )

    def setParallelizationLevel(self, value):
        """
        Sets level of parallelization

        :param value: Level of parallelization
        """
        return self._set(parallelizationLevel=value)

    def getParallelizationLevel(self):
        """
        Returns current level of parallelization
        """
        return self.getOrDefault(self.parallelizationLevel)
