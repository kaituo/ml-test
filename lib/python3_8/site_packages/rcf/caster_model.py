# Java imports
from typing import List, Optional, Tuple, Any

import jpype
import jpype.types as jtypes
from jpype import JArray
from typing import List, Optional
from com.amazon.randomcutforest.parkservices import RCFCaster
from com.amazon.randomcutforest.config import Precision
from com.amazon.randomcutforest.parkservices import ForecastDescriptor
from com.amazon.randomcutforest.config import TransformMethod
import jpype

# -------------------------------------------------------------------
# Cache the Java array types
# -------------------------------------------------------------------
DoubleArray = JArray(jtypes.JDouble)  # double[]
Double2DArray = JArray(DoubleArray)  # double[][]
LongArray = JArray(jtypes.JLong)  # long[]

# cache the JVM array classes
DoubleArray   = JArray(jtypes.JDouble)   # double[]
Double2DArray = JArray(DoubleArray)      # double[][]

def _to_double2d(data: Any) -> Double2DArray:
    """
    Convert 1-D or 2-D Python / NumPy / Pandas data → Java double[][].

    * 1-D input  => shape (N, 1)
    * 2-D input  => shape stays (N, D)
    """
    # ---------------------------------------------------------------
    # 1) normalise to a *list of lists* of Python floats
    # ---------------------------------------------------------------
    try:  # unwrap a Pandas Series
        import pandas as pd
        if isinstance(data, pd.Series):
            data = data.values
    except ImportError:
        pass

    import numpy as np
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            data = data[:, None]         # (N,) -> (N,1)
        data = data.tolist()             # ndarray -> nested list

    # plain Python list/tuple cases
    if isinstance(data, (list, tuple)):
        if not data:
            raise ValueError("`data` is empty")
        # 1-D list → wrap each scalar in its own list
        if not isinstance(data[0], (list, tuple)):
            data = [[float(v)] for v in data]
        else:
            # ensure inner values are plain floats
            data = [[float(v) for v in row] for row in data]
    else:
        raise TypeError("Unsupported data type for _to_double2d")

    # ---------------------------------------------------------------
    # 2) build the actual primitive double[][]
    # ---------------------------------------------------------------
    return Double2DArray([DoubleArray(row) for row in data])

def _to_long_array(ts, n_rows):
    """Convert timestamps or create increasing sequences like 0,1,2..."""
    if ts is None:
        ts = list(range(n_rows))
    return LongArray(ts)

class CasterModel:
    """
    Random Cut Forest Python Binding around the AWS Random Cut Forest Official Java version:
    https://github.com/aws/random-cut-forest-by-aws
    """

    def __init__(self, base_dimension=1 , shingle_size=20, num_trees: int = 50, output_after: int=64, anomaly_rate=0.005,
                 z_factor=2.5, score_differencing=0.5, ignore_delta_threshold=0, ignore_delta_threshold_ratio=0,
                 sample_size=256, strategy='EXPECTED_INVERSE_DEPTH', imputationMethod="",
                 fixedValue=None, forecastHorizon=15, rcfCalibration=False):
        if fixedValue is None:
            fixedValue = []

            # Convert Python string to the Java enum
        rcf_dimensions = base_dimension * shingle_size

        forestBuilder = (RCFCaster
        .builder()
        .dimensions(rcf_dimensions)
        .sampleSize(sample_size)
        .numberOfTrees(num_trees)
        .timeDecay(0.0001)
        .initialAcceptFraction(output_after*1.0/sample_size)
        .parallelExecutionEnabled(True)
        .compact(True)
        .precision(Precision.FLOAT_32)
        .boundingBoxCacheFraction(1)
        .shingleSize(shingle_size)
        .anomalyRate(anomaly_rate)
        .outputAfter(output_after)
        .internalShinglingEnabled(True)
        .transformMethod(TransformMethod.NORMALIZE)
        .forecastHorizon(forecastHorizon)
        .useRCFCallibration(rcfCalibration)
        #.randomSeed(0)
        )

        self.forest = forestBuilder.build()

    def process(self, point: List[float]) -> ForecastDescriptor:
        """
        Compute an anomaly score for the given point.

        Parameters
        ----------
        point: List[float]
            A data point with shingle size

        Returns
        -------
        float
            The anomaly score for the given point

        """
        return self.forest.process(point, 0)

    def process(self, point: List[float], time: int) -> ForecastDescriptor:
        """
        Compute an anomaly score for the given point.

        Parameters
        ----------
        point: List[float]
            A data point with shingle size

        Returns
        -------
        float
            The anomaly score for the given point

        """
        return self.forest.process(point, time)

    def processSequentially(self, point: List[List[float]]) -> ForecastDescriptor:
        """
        Compute an anomaly score for the given point.

        Parameters
        ----------
        point: List[float]
            A data point with shingle size

        Returns
        -------
        float
            The anomaly score for the given point

        """
        zeros = [0] * len(point)
        always_true = lambda _: True  # or `def always_true(_): return True`
        return self.forest.processSequentially(point, zeros, always_true)



    # -------------------------------------------------------------------
    # Your wrapper method (put inside CasterModel)
    # -------------------------------------------------------------------
    def processSequentially(self,
                            points: List[List[float]],
                            timestamps=None):
        """
        Python-friendly wrapper for
          List<AnomalyDescriptor> processSequentially(double[][], long[], Function)
        """

        j_points = _to_double2d(points)
        j_times = _to_long_array(timestamps, len(points))
        always_true = lambda _: True  # or `def always_true(_): return True`

        # call the exact 3-arg overload
        return list(self.forest.processSequentially(j_points, j_times, always_true))
