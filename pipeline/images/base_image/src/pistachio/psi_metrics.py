"""psi_metrics.py

An implementation of FeatureMetrics that computes population stability index
"""

from pistachio.feature_metrics import FeatureMetric

import pickle
import pandas as pd
from typing import List, Dict
import numpy as np


class PSImetrics(FeatureMetric):
    """class for computing Poppulation Stability Index
    string columns are supported as categorical columns, but not continuous (qcut errors on them)"""
    def __init__(self, max_unique_values=20, n_bins=20):
        self._categorical_cols = []
        self._continuous_cols = []
        self._max_unique_values = max_unique_values
        self._n_bins = n_bins
        self._psi_zero_bin_delta = 0.01 
        self._data_metrics = {}
    
    @property 
    def categorical_cols(self):
        """getter for categorical cols"""
        return self._categorical_cols

    @categorical_cols.setter 
    def categorical_cols(self, cols: List[str]):
        """setter for categorical cols"""
        self._categorical_cols = cols
        self._continuous_cols = list(set(self._continuous_cols) - set(cols))

    @property 
    def continuous_cols(self):
        """getter for continuous cols"""
        return self._continuous_cols

    @continuous_cols.setter 
    def continuous_cols(self, cols: List[str]):
        """setter for continuous cols"""
        self._continuous_cols = cols
        self._categorical_cols = list(set(self._categorical_cols) - set(cols))

    def infer_columns(self, data: pd.DataFrame, ignore_cols: List[str]=None):
        """attempt to determine categorical/continuous columns"""

        if ignore_cols:
            data=data.drop(columns=ignore_cols)

        # floats are continuous
        self._continuous_cols.extend(data.select_types(include=['float64']).columns)
        # categories are categorical
        self._categorical_cols.extend(data.select_types(include=['category']).columns)

        # integers strings dates  are categorical if there are few distinct values
        for cc in data.select_types(include=['int64','object','datetime64']).columns:
            if data[cc].nunique > self._max_unique_values:
                self._continuous_cols.append(cc)
            else:
                self._categorical_cols.append(cc)
        
        print("inferred column types:")
        print(f"categorical columns: {self._categorical_cols}")
        print(f"continuous columns: {self._continuous_cols}")

    def _fit_continuous_column(self, col: pd.Series):
        # map continuous to categorical
        # missing = col.isna().sum()
        catted, bins = pd.qcut(col,self._n_bins, retbins=True, labels=False, duplicates='drop')
        # this is good for floats, but may need to adapt for other dtype
        if pd.api.types.is_datetime64_any_dtype(col):
            bins = np.array([pd.Timestamp.min] + list(bins) + [pd.Timestamp.max])
        else:        
            bins = np.array([-np.inf] + list(bins) + [np.inf])
        
        cat_result = self._fit_categorical_column(pd.Series(catted))
        cat_result['bin_thresholds'] = bins
        return cat_result
        
    def _fit_categorical_column(self, col: pd.Series):
        """get properties of categorical column"""

        total = len(col)
        # the_by = by if by else col
        proportions = {val: float(count/total) for val, count in zip(*np.unique(col[~col.isna()], return_counts=True))}
        missing = col.isna().sum()
        # result = {'name': col.name}
        # for k,v in zip(proportions.index, proportions.values):
        #     result[k] = v 
        result = {} 
        result['proportions'] = proportions
        result['_NA'] = missing/total 
        result['_TOTAL_EVENTS'] = total
        return result
    
    def _evaluate_categorical_column(self, col: pd.Series, from_continuous=False):
        """evaluate a categorical column"""
        eval_total = len(col)
        eval_proportions = {val: count/eval_total for val, count in zip(*np.unique(col[~col.isna()], return_counts=True))}
        eval_missing = col.isna().sum()

        reference_data = self._data_metrics['categorical_columns'].get(col.name) if not from_continuous\
              else self._data_metrics['continuous_columns'].get(col.name)
        reference_proportions = reference_data['proportions']
        reference_total = reference_data['_TOTAL_EVENTS']
        reference_missing = reference_data['_NA']

        if not reference_data:
            raise ValueError(f'Error - could not retrieve reference statistics for categorical column {col.name}')
        
        # adjust for missing/zero content bins
        zero_count_reference = list(set(eval_proportions.keys()) - set(reference_proportions.keys()))
        zero_count_eval = list(set(reference_proportions.keys()) - set(eval_proportions.keys()))
        if zero_count_eval:
            print(f'The following categories had zero count in evaluation data but were observed in reference data: {zero_count_eval}')
        if zero_count_reference: 
            print(f'The following categories had zero count in reference data but were observed in evaluation data: {zero_count_reference}')

        # compute PSI
        psi = 0
        for k in list(set(list(eval_proportions.keys()) + list(reference_data['proportions'].keys()))):
            ref_prop = reference_proportions.get(k, self._psi_zero_bin_delta/reference_total)
            eval_prop = eval_proportions.get(k, self._psi_zero_bin_delta/eval_total)

            psi += (ref_prop - eval_prop)*np.log(ref_prop/eval_prop)
        
        # flag if missing data not expected
        if (reference_missing == 0) and (eval_missing > 0):
            print(f'Column had no missing records in reference data, but {eval_missing} in evaluation data')
        
        # account for missing, if applicable
        if (reference_missing > 0) or (eval_missing > 0):
            reference_miss_prop = np.max([reference_missing,self._psi_zero_bin_delta])/reference_total
            eval_miss_prop = np.max([eval_missing,self._psi_zero_bin_delta])/eval_total
            # print(f'reference miss prop: {reference_miss_prop}, eval miss prop: {eval_miss_prop}')
            psi += (reference_miss_prop - eval_miss_prop)*np.log(reference_miss_prop/eval_miss_prop)

        # convert to json serialisable types
        if pd.api.types.is_datetime64_dtype(col):
            reference_proportions = {str(k): v for k,v in reference_proportions.items()}
            eval_proportions = {str(k): v for k,v in eval_proportions.items()}
        elif pd.api.types.is_integer_dtype(col):
            reference_proportions = {int(k): v for k,v in reference_proportions.items()}
            eval_proportions = {int(k): v for k,v in eval_proportions.items()}

        details = {
            'reference_total': int(reference_total),
            'reference_proportions':  reference_proportions,
            'reference_missing': reference_missing,
            'eval_total': eval_total,
            'eval_proportions': eval_proportions,
            'eval_missing': int(eval_missing),
            'PSI': psi
        }
        return psi, details
        
    def _evaluate_continuous_column(self, col: pd.Series):
        """evaluate a conntinuous column"""
        reference_data = self._data_metrics['continuous_columns'].get(col.name)
        bin_thresholds = reference_data['bin_thresholds']

        catted = pd.cut(col, bin_thresholds, labels=False)
        # this is an array not a series, will need to fix
        psi, details = self._evaluate_categorical_column(catted, from_continuous=True)
        
        if pd.api.types.is_datetime64_dtype(col):
            bin_thresholds = [str(x) for x in bin_thresholds] 
        else:
            bin_thresholds = bin_thresholds.tolist()
        details['bin_thresholds'] = bin_thresholds
        return psi, details

        
    def fit(self, reference_data: pd.DataFrame):
        """learn parameters of data distribution"""
        
        # only infer if no categorical/continuous columns are specified, and flag is True
        if not (self._categorical_cols or self._continuous_cols):
            raise ValueError('no categorical or continuous columns specified, either set categorical_cols or continuous_cols, or call infer_columns()')
        
        ignored_cols = list(set(reference_data.columns) - set(self._categorical_cols + self._continuous_cols))
        undefined_cols = list(set(self._categorical_cols + self._continuous_cols) - set(reference_data.columns))
        if undefined_cols:
            raise ValueError(f'the following columns are not present in input data: {undefined_cols}')

        print("analysing the following categorical columns:")
        print('\n'.join(self._categorical_cols))
        print("analysing the following continuous columns:")
        print('\n'.join(self._continuous_cols))
        print("the following columns in the data will be ignored:")
        print('\n'.join(ignored_cols))

        categorical_columns = {}
        for cc in self._categorical_cols:
            categorical_columns[cc] = self._fit_categorical_column(reference_data[cc])
        continuous_columns = {}
        for cc in self._continuous_cols:
            continuous_columns[cc] = self._fit_continuous_column(reference_data[cc])
        self._data_metrics = {
            'categorical_columns': categorical_columns,
            'continuous_columns': continuous_columns
            }
        print("fitting done!")

    def evaluate(self, eval_data, **kwargs):
        """evaluate supplied data against learned distribution"""

        ignored_cols = list(set(eval_data.columns) - set(self._categorical_cols + self._continuous_cols))
        print(f'the following columns will not be evalutated: {ignored_cols}')
        # loop over columns
        # categorical_columns
        results = { }
        psi_values = []
        for col in self._categorical_cols:
            if col not in eval_data.columns:
                raise ValueError(f'Error column {col} is missing from data to be evaluated')
            psi_val, details = self._evaluate_categorical_column(eval_data[col])
            psi_values.append((col, psi_val, 'categorical', eval_data.dtypes.get(col)))
            results[col] = details
            print(f'evaluated column {col}, PSI = {psi_val}')
        for col in self._continuous_cols:
            if col not in eval_data.columns:
                raise ValueError(f'Error column {col} is missing from data to be evaluated')
            psi_val, details = self._evaluate_continuous_column(eval_data[col])
            psi_values.append((col, psi_val, 'continuous', eval_data.dtypes.get(col)))
            details['datatype'] = str(eval_data.dtypes.get(col))
            results[col] = details
            print(f'evaluated column {col}, PSI = {psi_val}')
        return psi_values, results

    def save(self, filepath: str):
        """save this object"""
        with open(filepath, 'wb') as outfile:
            # pickle.dump(self.__dict__, outfile, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self, outfile, pickle.HIGHEST_PROTOCOL)
        print(f"wrote data to {filepath}")       

    @classmethod
    def load(cls, filepath: str):
        """load object from file, return result"""
        with open(filepath, 'rb') as infile:
            obj = pickle.load(infile)
        print(f"read data from {filepath}")    
        return obj

