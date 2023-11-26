"""feature_metrics.py

Abstract class for feature metric implementations
"""
from abc import ABC, abstractmethod
import pandas as pd


class FeatureMetric(ABC): 
    """Abstract Base Class for feature metrics"""
    @abstractmethod  
    def fit(self, reference_data: pd.DataFrame, **kwargs):
        """learn parameters of data distribution"""
        pass
    @abstractmethod
    def evaluate(self, eval_data, **kwargs):
        """evaluate supplied data against learned distribution"""
        pass
    @abstractmethod
    def save(self, file_path: str):
        """save this object"""
        pass

    @classmethod
    @abstractmethod
    def load(cls, filepath: str):
        """load object from file, return result"""
        pass
