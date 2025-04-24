"""
Network Analysis Module

This module provides functions for analyzing road network topology and
calculating network relevance for exposure assessment.
"""

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Dict, Optional, Any
from shapely.geometry import LineString, Point


def create_road_network_graph(
    road_segments: gpd.GeoDataFrame,
    id_column: str = 'segment_id',
    directed: bool = True
) -> nx.Graph:
    """
    Convert road segments into a network graph structure.

    Parameters
    ----------
    road_segments : gpd.GeoDataFrame
        GeoDataFrame containing road segments
    id_column : str, optional
        Column containing unique identifiers, by default 'segment_id'
    directed : bool, optional
        Whether to create a directed graph, by default True

    Returns
    -------
    nx.Graph
        NetworkX graph representing the road network
    """
    # Choose graph type based on directed flag
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    # Function to get the start and end points of a LineString
    def get_endpoints(geom):
        if not isinstance(geom, LineString):
            return None, None
        
        coords = list(geom.coords)
        if len(coords) < 2:
            return None, None
        
        return Point(coords[0]), Point(coords[-1])
    
    # Create nodes and edges from road segments
    for idx, road in road_segments.iterrows():
        # Skip if no geometry
        if road.geometry is None or road.geometry.is_empty:
            continue
        
        # Get start and end points
        start_point, end_point = get_endpoints(road.geometry)
        if start_point is None or end_point is None:
            continue
        
        # Create nodes for start and end points if they don't exist
        start_node_id = f"node_{start_point.x:.6f}_{start_point.y:.6f}"
        end_node_id = f"node_{end_point.x:.6f}_{end_point.y:.6f}"
        
        # Add nodes with coordinates
        G.add_node(start_node_id, pos=(start_point.x, start_point.y))
        G.add_node(end_node_id, pos=(end_point.x, end_point.y))
        
        # Add edge with road segment properties
        segment_id = road[id_column]
        segment_length = road.geometry.length
        
        G.add_edge(
            start_node_id,
            end_node_id,
            id=segment_id,
            length=segment_length,
            weight=segment_length,  # Use length as weight for shortest path calculations
            attrs=road.to_dict()
        )
    
    return G


def calculate_baseline_betweenness(
    G: nx.Graph,
    weight: str = 'weight',
    normalized: bool = True
) -> Dict[Tuple[str, str], float]:
    """
    Calculate betweenness centrality for all edges in the network.

    Parameters
    ----------
    G : nx.Graph
        NetworkX graph representing the road network
    weight : str, optional
        Edge attribute to use as weight, by default 'weight'
    normalized : bool, optional
        Whether to normalize betweenness values, by default True

    Returns
    -------
    Dict[Tuple[str, str], float]
        Dictionary mapping edges to betweenness values
    """
    # Calculate edge betweenness centrality
    edge_betweenness = nx.edge_betweenness_centrality(
        G,
        weight=weight,
        normalized=normalized
    )
    
    return edge_betweenness


def calculate_network_disruption(
    G: nx.Graph,
    baseline_betweenness: Dict[Tuple[str, str], float],
    sample_fraction: float = 1.0,
    max_edges: Optional[int] = None,
    weight: str = 'weight'
) -> Dict[str, float]:
    """
    Calculate network disruption impact for each edge.

    Parameters
    ----------
    G : nx.Graph
        NetworkX graph representing the road network
    baseline_betweenness : Dict[Tuple[str, str], float]
        Dictionary mapping edges to baseline betweenness values
    sample_fraction : float, optional
        Fraction of edges to sample (0.0-1.0), by default 1.0
    max_edges : Optional[int], optional
        Maximum number of edges to analyze, by default None
    weight : str, optional
        Edge attribute to use as weight, by default 'weight'

    Returns
    -------
    Dict[str, float]
        Dictionary mapping segment IDs to disruption scores
    """
    disruption_scores = {}
    
    # Get a list of all edges
    all_edges = list(G.edges())
    
    # Sample edges if requested
    if sample_fraction < 1.0 or (max_edges is not None and max_edges < len(all_edges)):
        if sample_fraction < 1.0:
            num_edges = int(len(all_edges) * sample_fraction)
        if max_edges is not None:
            num_edges = min(num_edges, max_edges)
        
        # Randomly sample edges
        import random
        sampled_edges = random.sample(all_edges, num_edges)
    else:
        sampled_edges = all_edges
    
    # Analyze each edge
    for edge in sampled_edges:
        source, target = edge
        
        # Get the segment ID
        edge_id = G[source][target]['id']
        
        # Create a copy of the graph and remove the edge
        G_temp = G.copy()
        G_temp.remove_edge(source, target)
        
        # Check if removing the edge disconnects the graph
        # For directed graphs, we check the weakly connected components
        if directed and hasattr(nx, 'is_weakly_connected'):
            is_connected = nx.is_weakly_connected(G_temp)
        else:
            is_connected = nx.is_connected(G_temp)
        
        # If removing the edge disconnects the graph, assign maximum disruption
        if not is_connected:
            disruption_scores[edge_id] = 1.0
            continue
        
        try:
            # Calculate new betweenness values
            new_betweenness = nx.edge_betweenness_centrality(
                G_temp,
                weight=weight,
                normalized=True
            )
            
            # Calculate total variation in betweenness
            total_variation = sum(
                abs(new_betweenness.get((u, v), 0) - baseline_betweenness.get((u, v), 0))
                for u, v in G_temp.edges()
            )
            
            disruption_scores[edge_id] = total_variation
        except Exception as e:
            # If betweenness calculation fails, assign a default value
            disruption_scores[edge_id] = 0.5
            print(f"Error calculating betweenness for edge {edge_id}: {e}")
    
    return disruption_scores


def classify_network_relevance(
    disruption_scores: Dict[str, float],
    min_score: Optional[float] = None,
    max_score: Optional[float] = None
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Classify road segments based on network disruption impact.

    Parameters
    ----------
    disruption_scores : Dict[str, float]
        Dictionary mapping segment IDs to disruption scores
    min_score : Optional[float], optional
        Minimum score for normalization, by default None
    max_score : Optional[float], optional
        Maximum score for normalization, by default None

    Returns
    -------
    Tuple[Dict[str, float], Dict[str, str]]
        Tuple containing:
        - normalized_scores: Dictionary mapping segment IDs to normalized scores (1-5)
        - relevance_classes: Dictionary mapping segment IDs to relevance classes
    """
    # Determine min and max scores if not provided
    if min_score is None:
        min_score = min(disruption_scores.values())
    if max_score is None:
        max_score = max(disruption_scores.values())
    
    # Normalize scores to 1-5 scale
    normalized_scores = {}
    for edge_id, score in disruption_scores.items():
        # Min-max normalization to 1-5 scale
        if max_score > min_score:  # Avoid division by zero
            normalized = 1 + 4 * (score - min_score) / (max_score - min_score)
        else:
            normalized = 3  # Default to medium if all scores are equal
        normalized_scores[edge_id] = normalized
    
    # Classify into categories
    relevance_classes = {}
    for edge_id, score in normalized_scores.items():
        if score >= 4.5:
            relevance_classes[edge_id] = 'Very High'
        elif score >= 3.5:
            relevance_classes[edge_id] = 'High'
        elif score >= 2.5:
            relevance_classes[edge_id] = 'Moderate'
        elif score >= 1.5:
            relevance_classes[edge_id] = 'Low'
        else:
            relevance_classes[edge_id] = 'Very Low'
    
    return normalized_scores, relevance_classes


def transfer_relevance_to_segments(
    road_segments: gpd.GeoDataFrame,
    relevance_scores: Dict[str, float],
    id_column: str = 'segment_id',
    score_column: str = 'network_relevance_score',
    class_column: str = 'network_relevance_class'
) -> gpd.GeoDataFrame:
    """
    Transfer network relevance scores to road segments.

    Parameters
    ----------
    road_segments : gpd.GeoDataFrame
        GeoDataFrame containing road segments
    relevance_scores : Dict[str, float]
        Dictionary mapping segment IDs to relevance scores
    id_column : str, optional
        Column containing segment identifiers, by default 'segment_id'
    score_column : str, optional
        Column name for relevance scores, by default 'network_relevance_score'
    class_column : str, optional
        Column name for relevance classes, by default 'network_relevance_class'

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with added relevance scores and classes
    """
    # Create a copy of the input GeoDataFrame
    result = road_segments.copy()
    
    # Add relevance scores to the GeoDataFrame
    result[score_column] = result[id_column].map(
        lambda x: relevance_scores.get(x, 3.0)  # Default to medium if not found
    )
    
    # Classify relevance scores
    def classify_score(score):
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
    
    result[class_column] = result[score_column].apply(classify_score)
    
    return result


def analyze_network_relevance(
    road_segments: gpd.GeoDataFrame,
    id_column: str = 'segment_id',
    sample_fraction: float = 1.0,
    max_edges: Optional[int] = None
) -> gpd.GeoDataFrame:
    """
    Perform complete network relevance analysis for road segments.

    Parameters
    ----------
    road_segments : gpd.GeoDataFrame
        GeoDataFrame containing road segments
    id_column : str, optional
        Column containing segment identifiers, by default 'segment_id'
    sample_fraction : float, optional
        Fraction of edges to sample, by default 1.0
    max_edges : Optional[int], optional
        Maximum number of edges to analyze, by default None

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with added network relevance information
    """
    # Create network graph
    G = create_road_network_graph(road_segments, id_column=id_column)
    
    # Calculate baseline betweenness
    baseline_betweenness = calculate_baseline_betweenness(G)
    
    # Calculate network disruption
    disruption_scores = calculate_network_disruption(
        G,
        baseline_betweenness,
        sample_fraction=sample_fraction,
        max_edges=max_edges
    )
    
    # Classify network relevance
    normalized_scores, relevance_classes = classify_network_relevance(disruption_scores)
    
    # Transfer relevance to segments
    result = transfer_relevance_to_segments(
        road_segments,
        normalized_scores,
        id_column=id_column
    )
    
    return result
