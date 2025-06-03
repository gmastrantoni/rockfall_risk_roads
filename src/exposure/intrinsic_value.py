"""
Intrinsic Value Module

This module provides functions for calculating the intrinsic value of road segments
based on road characteristics for exposure assessment in rockfall risk analysis.
"""
import geopandas as gpd
import pandas as pd
import numpy as np
import logging
from typing import Union, List, Tuple, Dict, Optional, Any

# Set up logger
logger = logging.getLogger(__name__)


class IntrinsicValueCalculator:
    """
    Class for calculating the intrinsic value of road segments.
    
    This class handles the assessment of intrinsic road value based on
    road characteristics such as type, function, condition, and toll status.
    """
    
    def __init__(
        self,
        type_weight: float = 0.4,
        function_weight: float = 0.3,
        condition_weight: float = 0.2,
        toll_weight: float = 0.1,
        type_column: str = 'tr_str_ty',
        function_column: str = 'tr_str_cf',
        condition_column: str = 'tr_str_sta',
        toll_column: str = 'tr_str_ped'
    ):
        """
        Initialize the IntrinsicValueCalculator.

        Parameters
        ----------
        type_weight : float, optional
            Weight for road type, by default 0.4
        function_weight : float, optional
            Weight for road function, by default 0.3
        condition_weight : float, optional
            Weight for road condition, by default 0.2
        toll_weight : float, optional
            Weight for toll status, by default 0.1
        type_column : str, optional
            Column name for road type, by default 'tr_str_ty'
        function_column : str, optional
            Column name for road function, by default 'tr_str_cf'
        condition_column : str, optional
            Column name for road condition, by default 'tr_str_sta'
        toll_column : str, optional
            Column name for toll status, by default 'tr_str_ped'
        """
        # Store weights
        self.weights = {
            'type': type_weight,
            'function': function_weight,
            'condition': condition_weight,
            'toll': toll_weight
        }
        
        # Store column names
        self.columns = {
            'type': type_column,
            'function': function_column,
            'condition': condition_column,
            'toll': toll_column
        }
        
        # Define scoring systems for each attribute
        
        # Road type scoring (tr_str_ty)
        # Higher scores for more important road types
        self.type_scores = {
            '01': 5,  # indifferentiated road
            '02': 2,  # tratto pedonale
            '03': 4,  # di raccordo intermodale
            '04': 3,  # Rampa / svincolo
            '05': 1,  
            '93': 3   # Other/Unknown (default to medium)
        }
        
        # Road function scoring (tr_str_cf)
        # Higher scores for more important road types
        self.function_scores = {
            '01': 5,  # autostrada
            '02': 4,  # strada extraurbana principale
            '03': 3,  # strada extraurbana secondaria
            '04': 2,  # strada urbana di scorrimento
            '05': 1,  # strada urbana di quartiere
            '95': 2,   # Other/Unknown (default to 2)
            '93': 2   # Other/Unknown (default to 2)
        }
        
        # Road condition scoring (tr_str_sta)
        self.condition_scores = {
            '01': 5,  # in esercizio
            '02': 3,  # in costruzione
            '03': 1,  # in disuso
            '93': 3   # Unknown (default to 3)
        }
        
        # Toll status scoring (tr_str_ped)
        # Toll roads typically have higher importance
        self.toll_scores = {
            '01': 5,  # Toll road
            '02': 3,  # Non-toll road
            '93': 3   # Unknown (default to medium)
        }
    
    def calculate_intrinsic_value(self, road_segment) -> float:
        """
        Calculate the intrinsic value of a road segment.

        Parameters
        ----------
        road_segment : pd.Series
            Series containing road attributes

        Returns
        -------
        float
            Intrinsic value score (1-5 scale)
        """
        # Extract attribute values from road segment
        road_type = str(road_segment.get(self.columns['type'], '93'))
        road_function = str(road_segment.get(self.columns['function'], '93'))
        road_condition = str(road_segment.get(self.columns['condition'], '93'))
        road_toll = str(road_segment.get(self.columns['toll'], '93'))
        
        # Calculate individual scores with fallbacks to default values
        type_score = self.type_scores.get(road_type, self.type_scores['93'])
        function_score = self.function_scores.get(road_function, self.function_scores['93'])
        condition_score = self.condition_scores.get(road_condition, self.condition_scores['93'])
        toll_score = self.toll_scores.get(road_toll, self.toll_scores['93'])
        
        # Calculate weighted intrinsic value score
        intrinsic_value = (
            self.weights['type'] * type_score +
            self.weights['function'] * function_score +
            self.weights['condition'] * condition_score +
            self.weights['toll'] * toll_score
        )
        
        # Normalize to 1-5 scale
        return min(5, max(1, intrinsic_value))
    
    def calculate_for_dataframe(
        self,
        road_segments: gpd.GeoDataFrame,
        output_column: str = 'intrinsic_value_score'
    ) -> gpd.GeoDataFrame:
        """
        Calculate intrinsic value for all road segments in a DataFrame.

        Parameters
        ----------
        road_segments : gpd.GeoDataFrame
            GeoDataFrame containing road segments
        output_column : str, optional
            Column name for output score, by default 'intrinsic_value_score'

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with added intrinsic value scores
        """
        # Create a copy to avoid modifying the original
        result = road_segments.copy()
        
        # Apply calculation to each row
        result[output_column] = result.apply(self.calculate_intrinsic_value, axis=1)
        
        return result
