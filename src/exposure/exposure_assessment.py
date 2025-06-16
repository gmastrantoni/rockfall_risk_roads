"""
Exposure Assessment Module (Redesigned)

This module integrates the intrinsic value module with network relevance analysis
for exposure assessment of road segments.
"""

import geopandas as gpd
from src.road.network_analysis import RoadNetworkGraph
from src.exposure.network_relevance import NetworkRelevanceAnalyzer, calculate_network_exposure
from src.exposure.intrinsic_value import IntrinsicValueCalculator
from src.utils.geo_utils import multilinestring_to_linestring

def perform_exposure_assessment(
    segmented_roads: gpd.GeoDataFrame,
    segment_id_col: str = 'segment_id',
    relevance_weight: float = 0.6,
    intrinsic_weight: float = 0.4
) -> gpd.GeoDataFrame:
    """
    Perform integrated exposure assessment using intrinsic value and network relevance.

    Parameters
    ----------
    segmented_roads_path : str, optional
        Path to the segmented roads GeoPackage, by default 'data/intermediate/road_segments.gpkg'
    segment_id_col : str, optional
        Column name for segment ID, by default 'segment_id'
    relevance_weight : float, optional
        Weight for network relevance, by default 0.6
    intrinsic_weight : float, optional
        Weight for intrinsic value, by default 0.4

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with integrated exposure assessment
    """
    # Load road segments
    roads = segmented_roads.copy()

    # Transform MultiLineString to LineString if needed
    if (roads.geometry.geom_type == 'MultiLineString').any():
        road_segments = multilinestring_to_linestring(roads)
    else:
        road_segments = roads

    # Step 1: Calculate intrinsic values
    intrinsic_calc = IntrinsicValueCalculator()
    road_segments = intrinsic_calc.calculate_for_dataframe(road_segments)

    # Step 2: Create network graph
    road_network = RoadNetworkGraph()
    graph = road_network.create_graph_from_segments(road_segments, segment_id_col=segment_id_col)

    # Step 3: Analyze network relevance
    relevance_analyzer = NetworkRelevanceAnalyzer(graph, road_network)
    relevance_analyzer.calculate_baseline_metrics()
    df_impacts = relevance_analyzer.analyze_all_segments()
    df_classified = relevance_analyzer.classify_segments_by_relevance(df_impacts)

    # Step 4: Merge results back to GeoDataFrame
    gdf_with_relevance = relevance_analyzer.export_relevance_to_geodataframe(
        road_segments, df_classified, segment_id_col=segment_id_col
    )

    # Step 5: Calculate integrated exposure
    gdf_final = calculate_network_exposure(
        gdf_with_relevance,
        relevance_weight=relevance_weight,
        intrinsic_weight=intrinsic_weight
    )
    return gdf_final