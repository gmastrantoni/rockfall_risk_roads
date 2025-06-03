"""
Exposure Assessment Module

This module provides functions for calculating the exposure component of rockfall risk
assessment based on both intrinsic road value and network relevance.

The exposure assessment quantifies the importance and value of road infrastructure
through both inherent road characteristics and network topological importance.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import logging
from typing import Union, List, Tuple, Dict, Optional, Any
from .intrinsic_value import IntrinsicValueCalculator

# Set up logger
logger = logging.getLogger(__name__)


class ExposureAssessment:
    """
    Class for calculating the exposure component of rockfall risk assessment.
    
    This class combines intrinsic road value and network relevance to determine
    the overall exposure score for road segments.
    """
    
    def __init__(
        self,
        intrinsic_weight: float = 0.7,
        network_weight: float = 0.3,
        intrinsic_value_calculator: Optional[IntrinsicValueCalculator] = None
    ):
        """
        Initialize the ExposureAssessment.

        Parameters
        ----------
        intrinsic_weight : float, optional
            Weight for intrinsic road value, by default 0.7
        network_weight : float, optional
            Weight for network relevance, by default 0.3
        intrinsic_value_calculator : Optional[IntrinsicValueCalculator], optional
            Calculator for intrinsic value, by default None (creates a new one)
        """
        self.intrinsic_weight = intrinsic_weight
        self.network_weight = network_weight
        
        if intrinsic_value_calculator is None:
            self.intrinsic_calculator = IntrinsicValueCalculator()
        else:
            self.intrinsic_calculator = intrinsic_value_calculator
    
    def calculate_exposure_score(
        self,
        intrinsic_value: float,
        network_relevance: float
    ) -> float:
        """
        Calculate exposure score from intrinsic value and network relevance.

        Parameters
        ----------
        intrinsic_value : float
            Intrinsic road value score (1-5 scale)
        network_relevance : float
            Network relevance score (1-5 scale)

        Returns
        -------
        float
            Exposure score (1-5 scale)
        """
        # Calculate weighted exposure score
        exposure_score = (
            self.intrinsic_weight * intrinsic_value +
            self.network_weight * network_relevance
        )
        
        # Normalize to 1-5 scale
        return min(5, max(1, exposure_score))
    
    def classify_exposure(self, score: float) -> str:
        """
        Classify an exposure score into a categorical class.

        Parameters
        ----------
        score : float
            Exposure score (1-5 scale)

        Returns
        -------
        str
            Exposure class
        """
        if score >= 4.5:
            return 'Very High'
        elif score >= 3.5:
            return 'High'
        elif score >= 2.5:
            return 'Moderate'
        elif score >= 1.5:
            return 'Low'
        else:
            return 'Very Low'
    
    def assess_exposure(
        self,
        road_segments: gpd.GeoDataFrame,
        network_relevance_column: str = 'network_relevance_score',
        intrinsic_value_column: str = 'intrinsic_value_score',
        output_score_column: str = 'exposure_score',
        output_class_column: str = 'exposure_class'
    ) -> gpd.GeoDataFrame:
        """
        Perform complete exposure assessment for road segments.
        
        This function calculates both intrinsic value (if not already present)
        and combines it with network relevance to determine overall exposure.

        Parameters
        ----------
        road_segments : gpd.GeoDataFrame
            GeoDataFrame containing road segments
        network_relevance_column : str, optional
            Column with network relevance scores, by default 'network_relevance_score'
        intrinsic_value_column : str, optional
            Column with intrinsic value scores, by default 'intrinsic_value_score'
        output_score_column : str, optional
            Column for output exposure score, by default 'exposure_score'
        output_class_column : str, optional
            Column for output exposure class, by default 'exposure_class'

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with added exposure scores and classes
        """
        logger.info(f"Starting exposure assessment for {len(road_segments)} road segments")
        
        # Create a copy to avoid modifying the original
        result = road_segments.copy()
        
        # Calculate intrinsic value if not already present
        if intrinsic_value_column not in result.columns:
            logger.info("Calculating intrinsic value for road segments")
            result = self.intrinsic_calculator.calculate_for_dataframe(
                result,
                output_column=intrinsic_value_column
            )
        
        # Handle missing network relevance scores
        if network_relevance_column not in result.columns:
            logger.warning(f"Network relevance column '{network_relevance_column}' not found, using default values")
            result[network_relevance_column] = 3.0  # Default to medium
        
        # Calculate exposure score
        logger.info("Calculating exposure scores")
        result[output_score_column] = result.apply(
            lambda row: self.calculate_exposure_score(
                row[intrinsic_value_column],
                row[network_relevance_column]
            ),
            axis=1
        )
        
        # Classify exposure
        logger.info("Classifying exposure scores")
        result[output_class_column] = result[output_score_column].apply(self.classify_exposure)
        
        # Log summary statistics
        class_distribution = result[output_class_column].value_counts()
        logger.info("Exposure class distribution:")
        for cls, count in class_distribution.items():
            logger.info(f"  {cls}: {count} segments ({count/len(result)*100:.1f}%)")
        
        logger.info("Exposure assessment completed successfully")
        return result


def calculate_exposure_for_network(
    road_segments: gpd.GeoDataFrame,
    network_analysis_results: Optional[gpd.GeoDataFrame] = None,
    intrinsic_weight: float = 0.7,
    network_weight: float = 0.3,
    id_column: str = 'segment_id',
    simplify_endpoints: bool = True,
    endpoint_tolerance: float = 1.0,
    sample_fraction: float = 1.0,
    max_edges: Optional[int] = None
) -> gpd.GeoDataFrame:
    """
    Calculate exposure for a road network by performing both intrinsic value
    assessment and network relevance analysis.

    Parameters
    ----------
    road_segments : gpd.GeoDataFrame
        GeoDataFrame containing road segments
    network_analysis_results : Optional[gpd.GeoDataFrame], optional
        Pre-calculated network analysis results, by default None
    intrinsic_weight : float, optional
        Weight for intrinsic road value, by default 0.7
    network_weight : float, optional
        Weight for network relevance, by default 0.3
    id_column : str, optional
        Column containing segment identifiers, by default 'segment_id'
    simplify_endpoints : bool, optional
        Whether to simplify endpoints in network analysis, by default True
    endpoint_tolerance : float, optional
        Tolerance for endpoint simplification, by default 1.0
    sample_fraction : float, optional
        Fraction of edges to sample for network analysis, by default 1.0
    max_edges : Optional[int], optional
        Maximum number of edges to analyze, by default None

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with exposure assessment results
    """
    logger.info(f"Starting exposure calculation for {len(road_segments)} road segments")
    
    # Check if we need to perform network analysis
    if network_analysis_results is None:
        # Import here to avoid circular imports
        from ..exposure.network_relevance import analyze_network_relevance
        
        logger.info("No pre-calculated network analysis results, performing network analysis")
        network_analysis_results = analyze_network_relevance(
            road_segments,
            id_column=id_column,
            simplify_endpoints=simplify_endpoints,
            endpoint_tolerance=endpoint_tolerance,
            sample_fraction=sample_fraction,
            max_edges=max_edges
        )
    else:
        logger.info("Using pre-calculated network analysis results")
    
    # Initialize exposure assessment
    exposure_assessment = ExposureAssessment(
        intrinsic_weight=intrinsic_weight,
        network_weight=network_weight
    )
    
    # Perform exposure assessment
    result = exposure_assessment.assess_exposure(network_analysis_results)
    
    logger.info("Exposure calculation completed successfully")
    return result