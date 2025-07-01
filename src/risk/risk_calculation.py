"""
Risk Calculation Module

This module provides functions for calculating final risk scores by combining
hazard, vulnerability, and exposure components following the risk equation:
Risk = Hazard x Vulnerability x Exposure.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import logging
from typing import Union, List, Tuple, Dict, Optional, Any

# Set up logger
logger = logging.getLogger(__name__)


class RiskCalculator:
    """
    Class for calculating final risk scores by combining hazard, vulnerability, and exposure.
    
    This class implements the risk equation: Risk = Hazard x Vulnerability x Exposure.
    
    Hazard: 1-5 (int/float)
    Vulnerability: 0 or 1 (binary)
    Exposure: 0-1 (float)
    """
    
    def __init__(
        self,
        hazard_weight: float = 1.0,
        vulnerability_weight: float = 1.0,
        exposure_weight: float = 1.0,
        special_classes: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the RiskCalculator.

        Parameters
        ----------
        hazard_weight : float, optional
            Weight for hazard component, by default 1.0
        vulnerability_weight : float, optional
            Weight for vulnerability component, by default 1.0
        exposure_weight : float, optional
            Weight for exposure component, by default 1.0
        special_classes : Optional[Dict[str, float]], optional
            Special risk classes and their fixed scores, by default None
        """
        self.hazard_weight = hazard_weight
        self.vulnerability_weight = vulnerability_weight
        self.exposure_weight = exposure_weight
        
        # Default special class handling
        if special_classes is None:
            self.special_classes = {
                'Not at Risk': -1.0,            # Segments marked as not at risk
                'Area of Attention': 0.0        # Segments in area of attention
            }
        else:
            self.special_classes = special_classes
    
    def calculate_risk(
        self,
        hazard: float,
        vulnerability: float,
        exposure: float,
        special_class: Optional[str] = None
    ) -> float:
        """
        Calculate risk from hazard, vulnerability, and exposure.

        Parameters
        ----------
        hazard : float
            Hazard score (1-5 scale)
        vulnerability : float
            Vulnerability score (0 or 1)
        exposure : float
            Exposure score (0-1 scale)
        special_class : Optional[str], optional
            Special risk class, by default None

        Returns
        -------
        float
            Raw risk score
        """
        # Handle special classes if provided
        if special_class is not None and special_class in self.special_classes:
            return self.special_classes[special_class]
        
        # Calculate weighted risk score
        risk_score = (
            self.hazard_weight * hazard *
            self.vulnerability_weight * vulnerability *
            self.exposure_weight * exposure
        )
        if vulnerability == 0:
            risk_score = -1.0  # Not at Risk if vulnerability is 0

        return risk_score

    def normalize_risk_score(
        self,
        risk_score: float,
        min_score: float = 0.0,
        max_score: float = 5.0  # max hazard (5) * max vulnerability (1) * max exposure (1)
    ) -> float:
        """
        Normalize a risk score to a 1-5 scale.

        Parameters
        ----------
        risk_score : float
            Raw risk score
        min_score : float, optional
            Minimum possible risk score, by default 0.0
        max_score : float, optional
            Maximum possible risk score, by default 5.0 (5*1*1)

        Returns
        -------
        float
            Normalized risk score (1-5 scale)
        """
        # Handle special values
        if risk_score < 0:
            return -1.0  # Not at Risk
        if risk_score == 0:
            return 0.0   # Area of Attention
        
        # Normalize to 1-5 scale
        normalized = round(1 + 4 * (risk_score - min_score) / (max_score - min_score), 1)
        
        # Clip to 1-5 range
        return min(5, max(1, normalized))

    def classify_risk(self, score: float) -> str:
        """
        Classify a risk score into a risk class.

        Parameters
        ----------
        score : float
            Normalized risk score (1-5 scale or special values)

        Returns
        -------
        str
            Risk class
        """
        # Round score to 1 decimal place for classification
        # score = round(score, 1)
        # Handle special values
        if score < 0:
            return 'Not at Risk'
        if score == 0:
            return 'Area of Attention'
        
        # Classify based on 1-5 scale
        if score >= 3.5:
            return 'Very High Risk'
        elif score >= 2.5:
            return 'High Risk'
        elif score >= 1.5:
            return 'Moderate Risk'
        else:
            return 'Low Risk'


def calculate_risk_for_segments(
    road_segments: gpd.GeoDataFrame,
    hazard_column: str = 'hazard_score',
    vulnerability_column: str = 'vulnerability_score',
    exposure_column: str = 'exposure_score',
    risk_class_column: str = 'risk_class',
    output_raw_column: str = 'risk_score',
    output_normalized_column: str = 'risk_score_normalized',
    output_class_column: str = 'risk_class_final'
) -> gpd.GeoDataFrame:
    """
    Calculate risk scores for road segments by combining hazard, vulnerability, and exposure.

    Parameters
    ----------
    road_segments : gpd.GeoDataFrame
        GeoDataFrame containing road segments with hazard, vulnerability, and exposure scores
    hazard_column : str, optional
        Column containing hazard scores (1-5), by default 'hazard_score'
    vulnerability_column : str, optional
        Column containing vulnerability scores (0 or 1), by default 'vulnerability_score'
    exposure_column : str, optional
        Column containing exposure scores (0-1), by default 'exposure_score'
    risk_class_column : str, optional
        Column containing special risk classes, by default 'risk_class'
    output_raw_column : str, optional
        Column for output raw risk scores, by default 'risk_score'
    output_normalized_column : str, optional
        Column for output normalized risk scores, by default 'risk_score_normalized'
    output_class_column : str, optional
        Column for output risk classes, by default 'risk_class_final'

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with added risk scores and classes
    """
    logger.info(f"Starting risk calculation for {len(road_segments)} road segments")
    
    # Create a copy to avoid modifying the original
    result = road_segments.copy()
    
    # Check if all required columns exist
    missing_columns = []
    for column in [hazard_column, vulnerability_column, exposure_column]:
        if column not in result.columns:
            missing_columns.append(column)
    
    if missing_columns:
        logger.warning(f"Missing column(s): {missing_columns}. Some scores will use default values.")
    
    # Initialize risk calculator
    calculator = RiskCalculator()
    
    # Prepare input data with defaults for missing values
    if hazard_column not in result.columns:
        logger.warning(f"Hazard column '{hazard_column}' not found, using default value of 3.0")
        result[hazard_column] = 3.0  # mid value for 1-5
    if vulnerability_column not in result.columns:
        logger.warning(f"Vulnerability column '{vulnerability_column}' not found, using default value of 1 (vulnerable)")
        result[vulnerability_column] = 1  # default to vulnerable
    if exposure_column not in result.columns:
        logger.warning(f"Exposure column '{exposure_column}' not found, using default value of 0.5")
        result[exposure_column] = 0.5  # mid value for 0-1
    
    # Calculate raw risk scores
    logger.info("Calculating raw risk scores")

    def calc_risk(row):
        # Check for special class
        special_class = None
        if risk_class_column in row and row[risk_class_column] in calculator.special_classes:
            special_class = row[risk_class_column]
        
        # Get component scores
        hazard = row[hazard_column]
        vulnerability = row[vulnerability_column]
        exposure = row[exposure_column]
        
        # If vulnerability is 0, force risk = -1 (Not at Risk)
        if vulnerability == 0:
            return -1.0
        # Calculate risk
        return calculator.calculate_risk(hazard, vulnerability, exposure, special_class)
    
    result[output_raw_column] = result.apply(calc_risk, axis=1)
    
    # Normalize risk scores
    logger.info("Normalizing risk scores")
    max_possible = 5.0  # max hazard (5) * max vulnerability (1) * max exposure (1)
    def normalize_risk_override(x, row):
        if row[vulnerability_column] == 0:
            return -1.0
        return calculator.normalize_risk_score(x, max_score=max_possible)
    result[output_normalized_column] = result.apply(lambda row: normalize_risk_override(row[output_raw_column], row), axis=1)
    
    # Classify risk
    logger.info("Classifying risk scores")
    def classify_risk_override(x, row):
        if row[vulnerability_column] == 0:
            return 'Not at Risk'
        return calculator.classify_risk(x)
    result[output_class_column] = result.apply(lambda row: classify_risk_override(row[output_normalized_column], row), axis=1)
    
    # Log summary statistics
    valid_mask = (result[output_normalized_column] > 0)  # Exclude special classes
    if valid_mask.any():
        valid_scores = result.loc[valid_mask, output_normalized_column]
        score_stats = valid_scores.describe()
        logger.info(f"Risk score statistics (excluding special classes): min={score_stats['min']:.2f}, max={score_stats['max']:.2f}, mean={score_stats['mean']:.2f}")
    
    class_distribution = result[output_class_column].value_counts()
    logger.info("Risk class distribution:")
    for cls, count in class_distribution.items():
        logger.info(f"  {cls}: {count} segments ({count/len(result)*100:.1f}%)")
    
    logger.info("Risk calculation completed successfully")
    return result


def generate_risk_summary(
    risk_assessment: gpd.GeoDataFrame,
    segment_length_column: str = 'segment_length',
    risk_class_column: str = 'risk_class_final',
    output_summary: bool = True
) -> pd.DataFrame:
    """
    Generate summary statistics from risk assessment.

    Parameters
    ----------
    risk_assessment : gpd.GeoDataFrame
        GeoDataFrame containing risk assessment results
    segment_length_column : str, optional
        Column containing segment length, by default 'segment_length'
    risk_class_column : str, optional
        Column containing risk class, by default 'risk_class_final'
    output_summary : bool, optional
        Whether to print summary to log, by default True

    Returns
    -------
    pd.DataFrame
        DataFrame containing summary statistics
    """
    logger.info("Generating risk assessment summary statistics")
    
    # Check if length column exists, create if not
    if segment_length_column not in risk_assessment.columns:
        logger.warning(f"Segment length column '{segment_length_column}' not found, calculating from geometry")
        risk_assessment = risk_assessment.copy()
        risk_assessment[segment_length_column] = risk_assessment.geometry.length
    
    # Generate summary by risk class
    summary = risk_assessment.groupby(risk_class_column).agg({
        risk_class_column: 'count',
        segment_length_column: 'sum'
    }).rename(columns={
        risk_class_column: 'count',
        segment_length_column: 'total_length_m'
    })
    
    # Calculate percentage by length
    total_length = risk_assessment[segment_length_column].sum()
    summary['percentage'] = (summary['total_length_m'] / total_length) * 100
    
    if output_summary:
        logger.info("\nRisk Assessment Summary:")
        logger.info(f"Total segments: {len(risk_assessment)}")
        logger.info(f"Total length: {total_length:.1f} meters\n")
        logger.info("Distribution by risk class:")
        
        # Sort classes in a logical order
        class_order = [
            'Very High Risk', 'High Risk', 'Moderate Risk', 'Low Risk', 'Very Low Risk',
            'Area of Attention', 'Not at Risk'
        ]
        
        # Filter and sort classes that are present in the summary
        present_classes = [cls for cls in class_order if cls in summary.index]
        
        for cls in present_classes:
            if cls in summary.index:
                count = summary.loc[cls, 'count']
                length = summary.loc[cls, 'total_length_m']
                percentage = summary.loc[cls, 'percentage']
                logger.info(f"  {cls}: {count} segments ({length:.1f} m, {percentage:.1f}%)")
    
    return summary


def generate_risk_network_matrix(
    risk_assessment: gpd.GeoDataFrame,
    risk_class_column: str = 'risk_class_final',
    network_relevance_column: str = 'network_relevance_class',
    segment_length_column: str = 'segment_length'
) -> pd.DataFrame:
    """
    Generate a matrix of risk class vs network relevance.
    
    This function creates a cross-tabulation to show the distribution of
    road segments across risk classes and network relevance categories.

    Parameters
    ----------
    risk_assessment : gpd.GeoDataFrame
        GeoDataFrame containing risk assessment results
    risk_class_column : str, optional
        Column containing risk class, by default 'risk_class_final'
    network_relevance_column : str, optional
        Column containing network relevance class, by default 'network_relevance_class'
    segment_length_column : str, optional
        Column containing segment length, by default 'segment_length'

    Returns
    -------
    pd.DataFrame
        Cross-tabulation of risk class vs network relevance
    """
    logger.info("Generating risk vs network relevance matrix")
    
    # Check if required columns exist
    missing_columns = []
    for column in [risk_class_column, network_relevance_column]:
        if column not in risk_assessment.columns:
            missing_columns.append(column)
    
    if missing_columns:
        logger.warning(f"Missing column(s): {missing_columns}. Cannot generate complete matrix.")
        return pd.DataFrame()
    
    # Check if length column exists, create if not
    if segment_length_column not in risk_assessment.columns:
        logger.warning(f"Segment length column '{segment_length_column}' not found, calculating from geometry")
        risk_assessment = risk_assessment.copy()
        risk_assessment[segment_length_column] = risk_assessment.geometry.length
    
    # Generate cross-tabulation
    matrix = pd.crosstab(
        risk_assessment[network_relevance_column],
        risk_assessment[risk_class_column],
        values=risk_assessment[segment_length_column],
        aggfunc='sum'
    ).fillna(0)
    
    # Calculate row and column totals
    matrix['Total'] = matrix.sum(axis=1)
    matrix.loc['Total'] = matrix.sum(axis=0)
    
    logger.info("Risk vs Network Relevance Matrix generated successfully")
    
    return matrix


def identify_priority_segments(
    risk_assessment: gpd.GeoDataFrame,
    risk_column: str = 'risk_score_normalized',
    network_column: str = 'relevance_score',
    risk_threshold: float = 3.5,
    network_threshold: float = 3.5,
    output_column: str = 'priority_segment'
) -> gpd.GeoDataFrame:
    """
    Identify priority segments for intervention based on risk and network relevance.
    
    This function flags segments that have both high risk and high network relevance,
    indicating they should be prioritized for mitigation measures.

    Parameters
    ----------
    risk_assessment : gpd.GeoDataFrame
        GeoDataFrame containing risk assessment results
    risk_column : str, optional
        Column containing normalized risk scores, by default 'risk_score_normalized'
    network_column : str, optional
        Column containing network relevance scores, by default 'network_relevance_score'
    risk_threshold : float, optional
        Threshold for high risk, by default 3.5
    network_threshold : float, optional
        Threshold for high network relevance, by default 3.5
    output_column : str, optional
        Column for priority flag, by default 'priority_segment'

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with added priority segment flag
    """
    logger.info("Identifying priority segments for intervention")
    
    # Create a copy to avoid modifying the original
    result = risk_assessment.copy()
    
    # Check if required columns exist
    missing_columns = []
    for column in [risk_column, network_column]:
        if column not in result.columns:
            missing_columns.append(column)
    
    if missing_columns:
        logger.warning(f"Missing column(s): {missing_columns}. Cannot identify priority segments.")
        result[output_column] = False
        return result
    
    # Identify priority segments
    valid_mask = (result[risk_column] > 0)  # Exclude special classes like 'Not at Risk'
    result[output_column] = False
    
    result.loc[
        valid_mask & 
        (result[risk_column] >= risk_threshold) & 
        (result[network_column] >= network_threshold),
        output_column
    ] = True
    
    # Count priority segments
    priority_count = result[output_column].sum()
    logger.info(f"Identified {priority_count} priority segments for intervention")
    logger.info(f"Priority segments represent {priority_count/len(result)*100:.1f}% of all segments")
    
    return result