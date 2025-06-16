"""
network_analysis.py

Network Analysis Module for Rockfall Risk Assessment
This module provides functions for analyzing road network graphs, including
creation of network graphs from road segments and basic network metrics.
"""

import networkx as nx
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString
from typing import Dict, List, Tuple, Optional, Union
import warnings


class RoadNetworkGraph:
    """
    A class to create and manage road network graphs from geospatial data.
    """
    
    def __init__(self, tolerance: float = 0.001):
        """
        Initialize the RoadNetworkGraph.
        
        Parameters:
        -----------
        tolerance : float
            Tolerance for considering two points as the same node (in map units)
        """
        self.graph = nx.Graph()
        self.tolerance = tolerance
        self.nodes_dict = {}
        self.edge_to_segment = {}
        
    def create_graph_from_segments(self, road_segments: gpd.GeoDataFrame,
                                   segment_id_col: str = 'segment_id',
                                   weight_col: Optional[str] = None) -> nx.Graph:
        """
        Create a network graph from road segments GeoDataFrame.
        
        Parameters:
        -----------
        road_segments : gpd.GeoDataFrame
            GeoDataFrame containing road segments with LineString geometries
        segment_id_col : str
            Column name containing unique segment identifiers
        weight_col : str, optional
            Column name to use as edge weight (e.g., 'length', 'travel_time')
            
        Returns:
        --------
        nx.Graph
            NetworkX graph representing the road network
        """
        # Clear existing graph
        self.graph.clear()
        self.nodes_dict.clear()
        self.edge_to_segment.clear()
        
        # Extract nodes and edges from road segments
        node_id = 0
        
        for idx, segment in road_segments.iterrows():
            geom = segment.geometry
            
            if not isinstance(geom, LineString):
                warnings.warn(f"Segment {segment[segment_id_col]} is not a LineString, skipping.")
                continue
            
            # Get start and end points
            start_point = Point(geom.coords[0])
            end_point = Point(geom.coords[-1])
            
            # Find or create nodes
            start_node = self._get_or_create_node(start_point, node_id)
            if start_node == node_id:
                node_id += 1
                
            end_node = self._get_or_create_node(end_point, node_id)
            if end_node == node_id:
                node_id += 1
            
            # Add edge with attributes
            edge_attrs = {
                'segment_id': segment[segment_id_col],
                'geometry': geom,
                'length': geom.length
            }
            
            # Add weight if specified
            if weight_col and weight_col in segment:
                edge_attrs['weight'] = segment[weight_col]
            else:
                edge_attrs['weight'] = geom.length
            
            # Add additional attributes from the segment
            for col in road_segments.columns:
                if col not in ['geometry', segment_id_col] and col not in edge_attrs:
                    edge_attrs[col] = segment[col]
            
            self.graph.add_edge(start_node, end_node, **edge_attrs)
            self.edge_to_segment[(start_node, end_node)] = segment[segment_id_col]
            
        return self.graph
    
    def _get_or_create_node(self, point: Point, next_id: int) -> int:
        """
        Get existing node ID or create new node for a point.
        
        Parameters:
        -----------
        point : Point
            Shapely Point object
        next_id : int
            Next available node ID
            
        Returns:
        --------
        int
            Node ID
        """
        # Check if node already exists within tolerance
        for node_id, node_point in self.nodes_dict.items():
            if point.distance(node_point) < self.tolerance:
                return node_id
        
        # Create new node
        self.nodes_dict[next_id] = point
        self.graph.add_node(next_id, pos=(point.x, point.y), geometry=point)
        return next_id
    
    def get_network_components(self) -> List[List[int]]:
        """
        Get connected components of the network.
        
        Returns:
        --------
        List[List[int]]
            List of connected components (each component is a list of node IDs)
        """
        return list(nx.connected_components(self.graph))
    
    def is_connected(self) -> bool:
        """
        Check if the graph is fully connected.
        
        Returns:
        --------
        bool
            True if graph is connected, False otherwise
        """
        return nx.is_connected(self.graph)
    
    def get_largest_component(self) -> nx.Graph:
        """
        Get the largest connected component of the graph.
        
        Returns:
        --------
        nx.Graph
            Subgraph containing the largest connected component
        """
        if self.is_connected():
            return self.graph
        
        largest_cc = max(nx.connected_components(self.graph), key=len)
        return self.graph.subgraph(largest_cc).copy()
    
    def calculate_basic_metrics(self) -> Dict[str, Union[float, int]]:
        """
        Calculate basic network metrics.
        
        Returns:
        --------
        Dict[str, Union[float, int]]
            Dictionary containing basic network metrics
        """
        metrics = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'is_connected': self.is_connected(),
            'num_components': nx.number_connected_components(self.graph),
            'density': nx.density(self.graph),
            'average_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0
        }
        
        if self.is_connected():
            metrics['diameter'] = nx.diameter(self.graph)
            metrics['radius'] = nx.radius(self.graph)
            metrics['average_shortest_path_length'] = nx.average_shortest_path_length(self.graph, weight='weight')
        
        return metrics
    
    def get_edge_by_segment_id(self, segment_id: str) -> Optional[Tuple[int, int]]:
        """
        Get edge (node pair) corresponding to a segment ID.
        
        Parameters:
        -----------
        segment_id : str
            Road segment identifier
            
        Returns:
        --------
        Optional[Tuple[int, int]]
            Tuple of (start_node, end_node) or None if not found
        """
        for edge, seg_id in self.edge_to_segment.items():
            if seg_id == segment_id:
                return edge
        return None


def calculate_betweenness_centrality(graph: nx.Graph, 
                                   normalized: bool = True,
                                   weight: Optional[str] = 'weight',
                                   k: Optional[int] = None) -> Dict[Tuple[int, int], float]:
    """
    Calculate edge betweenness centrality for the road network.
    
    Parameters:
    -----------
    graph : nx.Graph
        NetworkX graph representing the road network
    normalized : bool
        Whether to normalize betweenness values
    weight : str, optional
        Edge attribute to use as weight
    k : int, optional
        Use k node samples to estimate betweenness (for large graphs)
        
    Returns:
    --------
    Dict[Tuple[int, int], float]
        Dictionary mapping edges to betweenness centrality values
    """
    return nx.edge_betweenness_centrality(graph, normalized=normalized, 
                                         weight=weight, k=k)


def analyze_network_connectivity(graph: nx.Graph) -> pd.DataFrame:
    """
    Analyze network connectivity patterns.
    
    Parameters:
    -----------
    graph : nx.Graph
        NetworkX graph representing the road network
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with connectivity analysis results
    """
    # Get degree centrality
    degree_cent = nx.degree_centrality(graph)
    
    # Get clustering coefficient
    clustering = nx.clustering(graph)
    
    # Create DataFrame
    df_nodes = pd.DataFrame({
        'node_id': list(degree_cent.keys()),
        'degree_centrality': list(degree_cent.values()),
        'degree': [graph.degree(n) for n in degree_cent.keys()],
        'clustering_coefficient': [clustering.get(n, 0) for n in degree_cent.keys()]
    })
    
    # Add node positions if available
    pos_dict = nx.get_node_attributes(graph, 'pos')
    if pos_dict:
        df_nodes['x'] = [pos_dict.get(n, (None, None))[0] for n in df_nodes['node_id']]
        df_nodes['y'] = [pos_dict.get(n, (None, None))[1] for n in df_nodes['node_id']]
    
    return df_nodes.sort_values('degree_centrality', ascending=False)


def identify_bridges(graph: nx.Graph) -> List[Tuple[int, int]]:
    """
    Identify bridge edges (edges whose removal disconnects the graph).
    
    Parameters:
    -----------
    graph : nx.Graph
        NetworkX graph representing the road network
        
    Returns:
    --------
    List[Tuple[int, int]]
        List of bridge edges
    """
    return list(nx.bridges(graph))


def calculate_edge_connectivity(graph: nx.Graph) -> int:
    """
    Calculate minimum number of edges that need to be removed to disconnect the graph.
    
    Parameters:
    -----------
    graph : nx.Graph
        NetworkX graph representing the road network
        
    Returns:
    --------
    int
        Edge connectivity of the graph
    """
    if not nx.is_connected(graph):
        return 0
    return nx.edge_connectivity(graph)