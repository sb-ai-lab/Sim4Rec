import sys
import numpy as np
from pyspark.ml.param.shared import Params, Param, TypeConverters

class HasUserKeyColumn(Params):
    
    userKeyColumn = Param(
        Params._dummy(),
        "userKeyColumn",
        "User identifier column name",
        typeConverter=TypeConverters.toString
    )

    def __init__(self):
        super().__init__()

    def setUserKeyColumn(self, value):
        return self._set(userKeyColumn=value)

    def getUserKeyColumn(self):
        return self.getOrDefault(self.userKeyColumn)


class HasItemKeyColumn(Params):
    
    itemKeyColumn = Param(
        Params._dummy(),
        "itemKeyColumn",
        "Item identifier column name",
        typeConverter=TypeConverters.toString
    )

    def __init__(self):
        super().__init__()

    def setItemKeyColumn(self, value):
        return self._set(itemKeyColumn=value)

    def getItemKeyColumn(self):
        return self.getOrDefault(self.itemKeyColumn)


class HasSeed(Params):

    seed = Param(
        Params._dummy(),
        "seed",
        "Random state seed",
        typeConverter=TypeConverters.toInt
    )

    def __init__(self):
        super().__init__()

    def setSeed(self, value):
        return self._set(seed=value)

    def getSeed(self):
        return self.getOrDefault(self.seed)


class HasSeedSequence(Params):

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

    def __init__(self):
        super().__init__()

    def initSeedSequence(self, value):
        self._rng = np.random.default_rng(value)
        return self._set(
            init_seed=value if value is not None else -1,
            current_seed=self._rng.integers(0, sys.maxsize)
        )

    def getInitSeed(self):
        value = self.getOrDefault(self.init_seed)
        return None if value == -1 else value

    def getNextSeed(self):
        seed = self.getOrDefault(self.current_seed)
        self._set(current_seed=self._rng.integers(0, sys.maxsize))
        return seed


class HasWeights(Params):

    weights = Param(
        Params._dummy(),
        "weights",
        "Weights for models ensemble",
        typeConverter=TypeConverters.toListFloat
    )

    def __init__(self):
        super().__init__()

    def setWeights(self, value):
        return self._set(weights=value)

    def getWeights(self):
        return self.getOrDefault(self.weights)


class HasMean(Params):

    mean = Param(
        Params._dummy(),
        "mean",
        "Mean parameter of normal distribution",
        typeConverter=TypeConverters.toFloat
    )

    def __init__(self):
        super().__init__()

    def setMean(self, value):
        return self._set(mean=value)

    def getMean(self):
        return self.getOrDefault(self.mean)


class HasStandardDeviation(Params):

    std = Param(
        Params._dummy(),
        "std",
        "Standard Deviation parameter of normal distribution",
        typeConverter=TypeConverters.toFloat
    )

    def __init__(self):
        super().__init__()

    def setStandardDeviation(self, value):
        return self._set(std=value)

    def getStandardDeviation(self):
        return self.getOrDefault(self.std)


class HasClipNegative(Params):

    clipNegative = Param(
        Params._dummy(),
        "clipNegative",
        "Boolean flag to clip negative values",
        typeConverter=TypeConverters.toBoolean
    )

    def __init__(self):
        super().__init__()

    def setClipNegative(self, value):
        return self._set(clipNegative=value)

    def getClipNegative(self):
        return self.getOrDefault(self.clipNegative)


class HasConstantValue(Params):
    constantValue = Param(
        Params._dummy(),
        "constantValue",
        "Constant value parameter",
        typeConverter=TypeConverters.toFloat
    )

    def __init__(self):
        super().__init__()

    def setConstantValue(self, value):
        return self._set(constantValue=value)

    def getConstantValue(self):
        return self.getOrDefault(self.constantValue)


class HasLabel(Params):
    label = Param(
        Params._dummy(),
        "label",
        "String label",
        typeConverter=TypeConverters.toString
    )

    def __init__(self):
        super().__init__()

    def setLabel(self, value):
        return self._set(label=value)

    def getLabel(self):
        return self.getOrDefault(self.label)


class HasDevice(Params):
    device = Param(
        Params._dummy(),
        "device",
        "Name of a device to use",
        typeConverter=TypeConverters.toString
    )

    def __init__(self):
        super().__init__()

    def setDevice(self, value):
        return self._set(device=value)

    def getDevice(self):
        return self.getOrDefault(self.device)


class HasDataSize(Params):
    data_size = Param(
        Params._dummy(),
        "data_size",
        "Size of a DataFrame",
        typeConverter=TypeConverters.toInt
    )

    def __init__(self):
        super().__init__()

    def setDataSize(self, value):
        return self._set(data_size=value)

    def getDataSize(self):
        return self.getOrDefault(self.data_size)


class HasParallelizationLevel(Params):
    parallelizationLevel = Param(
        Params._dummy(),
        "parallelizationLevel",
        "Level of parallelization",
        typeConverter=TypeConverters.toInt
    )

    def __init__(self):
        super().__init__()

    def setParallelizationLevel(self, value):
        return self._set(parallelizationLevel=value)

    def getParallelizationLevel(self):
        return self.getOrDefault(self.parallelizationLevel)
