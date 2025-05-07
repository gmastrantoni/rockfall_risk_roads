"""
Hazard Classification Module

This module provides functions for classifying hazard parameters
and calculating hazard scores for rockfall risk assessment.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Tuple, Dict, Optional, Any, Callable

class HazardClassification:
    """
    Class for classifying hazard parameters and calculating hazard scores.
    
    This class handles the classification of continuous hazard parameters
    into discrete hazard classes and calculates weighted hazard scores.
    """
    
    def __init__(
        self,
        parameter_weights: Optional[Dict[str, float]] = None,
        class_thresholds: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None
    ):
        """
        Initialize the HazardClassification.

        Parameters
        ----------
        parameter_weights : Optional[Dict[str, float]], optional
            Dictionary of parameter names to weights, by default None
        class_thresholds : Optional[Dict[str, Dict[str, Tuple[float, float]]]], optional
            Dictionary of parameter names to class thresholds, by default None
        """
        # Set default parameter weights if not provided
        if parameter_weights is None:
            self.parameter_weights = {
                'susceptibility': 0.4,
                'velocity': 0.3,
                'energy': 0.3
            }
        else:
            self.parameter_weights = parameter_weights
        
        # Set default class thresholds if not provided
        if class_thresholds is None:
            self.class_thresholds = {
                'susceptibility': {
                    'Very Low': (0.0, 0.2),
                    'Low': (0.2, 0.4),
                    'Moderate': (0.4, 0.6),
                    'High': (0.6, 0.8),
                    'Very High': (0.8, 1.0)
                },
                'velocity': {
                    'Very Low': (0.0, 4.0),
                    'Low': (4.0, 8.0),
                    'Moderate': (8.0, 12.0),
                    'High': (12.0, 16.0),
                    'Very High': (16.0, float('inf'))
                },
                'energy': {
                    'Very Low': (0.0, 100.0),
                    'Low': (100.0, 200.0),
                    'Moderate': (200.0, 300.0),
                    'High': (300.0, 400.0),
                    'Very High': (400.0, float('inf'))
                }
            }
        else:
            self.class_thresholds = class_thresholds
        
        # Define class to value mapping
        self.class_to_value = {
            'Very Low': 1,
            'Low': 2,
            'Moderate': 3,
            'High': 4,
            'Very High': 5
        }
    
    def classify_parameter(self, parameter_name: str, value: float) -> str:
        """
        Classify a parameter value into a hazard class.

        Parameters
        ----------
        parameter_name : str
            Name of the parameter
        value : float
            Parameter value

        Returns
        -------
        str
            Hazard class
        """
        # Get thresholds for this parameter
        thresholds = self.class_thresholds.get(parameter_name)
        
        if thresholds is None:
            raise ValueError(f"Unknown parameter: {parameter_name}")
        
        # Classify value based on thresholds
        for class_name, (min_val, max_val) in thresholds.items():
            if min_val <= value < max_val:
                return class_name
        
        # Default to None if value exceeds all thresholds
        return None
    
    def calculate_hazard_score(
        self,
        parameters: Dict[str, float]
    ) -> Tuple[float, str]:
        """
        Calculate weighted hazard score from parameter values.

        Parameters
        ----------
        parameters : Dict[str, float]
            Dictionary of parameter names to values

        Returns
        -------
        Tuple[float, str]
            Tuple containing hazard score and hazard class
        """
        # Classify each parameter
        classes = {}
        for param_name, value in parameters.items():
            classes[param_name] = self.classify_parameter(param_name, value)
        
        # Convert classes to values
        values = {}
        for param_name, class_name in classes.items():
            values[param_name] = self.class_to_value.get(class_name, 0)  # Default to 0 if not found
        
        # Calculate weighted score
        weighted_score = 0.0
        total_weight = 0.0
        
        for param_name, value in values.items():
            weight = self.parameter_weights.get(param_name, 0.0)
            weighted_score += weight * value
            total_weight += weight
        
        if total_weight > 0:
            hazard_score = weighted_score / total_weight
        else:
            hazard_score = None  # Default to None if no weights
        
        # Classify final hazard score
        hazard_class = self.classify_hazard_score(hazard_score)
        
        return hazard_score, hazard_class
    
    def classify_hazard_score(self, score: float) -> str:
        """
        Classify a hazard score into a hazard class.

        Parameters
        ----------
        score : float
            Hazard score

        Returns
        -------
        str
            Hazard class
        """
        if score >= 4.5:
            return 'Very High'
        elif score >= 3.5:
            return 'High'
        elif score >= 2.5:
            return 'Moderate'
        elif score >= 1.5:
            return 'Low'
        elif score >=0:
            return 'Very Low'
        else:
            return 'NULL'
    
    def classify_dataframe(self, df: pd.DataFrame, suffix: str = '_max') -> pd.DataFrame:
        """
        Classify hazard parameters in a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing hazard parameters
        suffix : str, optional
            Suffix for parameter columns, by default '_max'

        Returns
        -------
        pd.DataFrame
            DataFrame with added hazard classifications
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Classify each parameter
        for param_name in self.parameter_weights.keys():
            col_name = f"{param_name}{suffix}"
            
            if col_name in result.columns:
                result[f"{param_name}_class"] = result[col_name].apply(
                    lambda x: self.classify_parameter(param_name, x)
                )
                
                result[f"{param_name}_value"] = result[f"{param_name}_class"].map(self.class_to_value)
        
        # Calculate hazard scores
        def calculate_score_row(row):
            # Extract parameter values from the row
            parameters = {}
            for param_name in self.parameter_weights.keys():
                col_name = f"{param_name}{suffix}"
                if col_name in row:
                    parameters[param_name] = row[col_name]
            
            # Calculate hazard score and class
            if parameters:
                hazard_score, hazard_class = self.calculate_hazard_score(parameters)
                return pd.Series([hazard_score, hazard_class])
            else:
                return pd.Series([np.nan, 'Unknown'])
        
        # Apply score calculation to each row
        result[['hazard_score', 'hazard_class']] = result.apply(calculate_score_row, axis=1)
        
        return result
