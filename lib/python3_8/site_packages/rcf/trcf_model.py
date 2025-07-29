# Java imports
from typing import List, Optional, Tuple, Any

import numpy as np
import logging
from com.amazon.randomcutforest.parkservices import ThresholdedRandomCutForest
from com.amazon.randomcutforest.config import Precision
from com.amazon.randomcutforest.parkservices import AnomalyDescriptor
from com.amazon.randomcutforest.config import TransformMethod
from com.amazon.randomcutforest.parkservices.config import ScoringStrategy
import jpype
from com.amazon.randomcutforest.config import ForestMode, ImputationMethod

class TRandomCutForestModel:
    """
    Random Cut Forest Python Binding around the AWS Random Cut Forest Official Java version:
    https://github.com/aws/random-cut-forest-by-aws
    """

    def __init__(self, rcf_dimensions, shingle_size, num_trees: int = 30, output_after: int=256, anomaly_rate=0.005,
                 z_factor=2.5, score_differencing=0.5, ignore_delta_threshold=0, ignore_delta_threshold_ratio=0,
                 sample_size=256, strategy='EXPECTED_INVERSE_DEPTH', imputationMethod="",
                 fixedValue=None):
        if fixedValue is None:
            fixedValue = []

            # Convert Python string to the Java enum
        imputationMethod_enum = self._parse_imputation_method(imputationMethod)

        forestBuilder = (ThresholdedRandomCutForest
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
        #.learnIgnoreNearExpected(True) #3.8
        .alertOnce(True) #3.8
        .autoAdjust(True) #3.8
        .ignoreNearExpectedFromAbove([ignore_delta_threshold])
        #.normalizeTime(True)
        #.randomSeed(0)
        )
        if imputationMethod_enum is not None:
            forestBuilder = (forestBuilder.forestMode(ForestMode.STREAMING_IMPUTE)
                             .imputationMethod(imputationMethod_enum))
        if fixedValue:
            forestBuilder = forestBuilder.fillValues(fixedValue)

        self.forest = forestBuilder.build()
        if strategy == 'DISTANCE':
            self.forest.setScoringStrategy(ScoringStrategy.DISTANCE)
        self.forest.setZfactor(z_factor)
        # minimum difference between actual and expected value
        # 3.7
        #self.forest.setIgnoreNearExpectedFromAbove([ignore_delta_threshold])
        self.forest.setIgnoreNearExpectedFromAboveByRatio([ignore_delta_threshold_ratio])
        self.forest.setIgnoreNearExpectedFromBelowByRatio([ignore_delta_threshold_ratio])
        self.forest.setScoreDifferencing(score_differencing)
        #self.forest.getThresholder().setUpperZfactor(10)
        #self.forest.setLowerThreshold(1.1);

    def _parse_imputation_method(self, imputation_method_str: str):
        """
        Convert Python string into the appropriate Java enum constant.
        Default to ImputationMethod.NONE if string is unrecognized.
        """
        if imputation_method_str == "FIXED_VALUES":
            return ImputationMethod.FIXED_VALUES
        elif imputation_method_str == "ZERO":
            return ImputationMethod.ZERO
        elif imputation_method_str == "NONE":
            return None
        else:
            # Fallback if an unrecognized string is given
            return None

    def process(self, point: List[float]) -> AnomalyDescriptor:
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

    def process(self, point: List[float], time: int) -> AnomalyDescriptor:
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
