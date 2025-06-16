"""
network_relevance.py

Network Relevance Module for Rockfall Risk Assessment
This module provides functions for analyzing road network topology and calculating 
network relevance for exposure assessment. The network relevance analysis identifies 
critical road segments whose loss would significantly impact the overall transportation 
network functionality.
"""

import networkx as nx
import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
# from collections import defaultdict
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
# from functools import partial


class NetworkRelevanceAnalyzer:
    """
    Analyzes network relevance of road segments for exposure assessment.
    """
    
    def __init__(self, graph: nx.Graph, road_network: 'RoadNetworkGraph' = None):
        """
        Initialize the NetworkRelevanceAnalyzer.
        
        Parameters:
        -----------
        graph : nx.Graph
            NetworkX graph representing the road network
        road_network : RoadNetworkGraph, optional
            RoadNetworkGraph object for segment-edge mapping
        """
        self.graph = graph.copy()
        self.road_network = road_network
        self.baseline_metrics = {}
        self.segment_impacts = {}
        
    def calculate_baseline_metrics(self) -> Dict[str, float]:
        """
        Calculate baseline network metrics before any segment removal.
        
        Returns:
        --------
        Dict[str, float]
            Dictionary of baseline metrics
        """
        self.baseline_metrics = {
            'connected': nx.is_connected(self.graph),
            'num_components': nx.number_connected_components(self.graph),
            'largest_component_size': len(max(nx.connected_components(self.graph), key=len)),
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges()
        }
        
        # Calculate average shortest path length for largest component
        if self.baseline_metrics['connected']:
            self.baseline_metrics['avg_shortest_path'] = nx.average_shortest_path_length(
                self.graph, weight='weight'
            )
        else:
            largest_cc = max(nx.connected_components(self.graph), key=len)
            subgraph = self.graph.subgraph(largest_cc)
            self.baseline_metrics['avg_shortest_path'] = nx.average_shortest_path_length(
                subgraph, weight='weight'
            )
        
        # Calculate betweenness centrality
        self.baseline_metrics['betweenness'] = nx.edge_betweenness_centrality(
            self.graph, normalized=True, weight='weight'
        )
        
        return self.baseline_metrics
    
    def evaluate_segment_removal_impact(self, segment_id: str = None, 
                                      edge: Tuple[int, int] = None,
                                      parallel: bool = False,
                                      n_jobs: int = -1) -> Dict[str, float]:
        """
        Evaluate the impact of removing a specific road segment.
        
        Parameters:
        -----------
        segment_id : str, optional
            Road segment identifier
        edge : Tuple[int, int], optional
            Edge to remove (if segment_id not provided)
        parallel : bool
            Whether to use parallel processing for multiple segments
        n_jobs : int
            Number of parallel jobs (-1 for all CPUs)
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of impact metrics
        """
        if segment_id and self.road_network:
            edge = self.road_network.get_edge_by_segment_id(segment_id)
            if not edge:
                warnings.warn(f"Segment {segment_id} not found in network")
                return {}
        
        if not edge:
            raise ValueError("Either segment_id or edge must be provided")
        
        # Create temporary graph with edge removed
        temp_graph = self.graph.copy()
        if temp_graph.has_edge(*edge):
            temp_graph.remove_edge(*edge)
        else:
            # Try reversed edge
            edge = (edge[1], edge[0])
            if temp_graph.has_edge(*edge):
                temp_graph.remove_edge(*edge)
            else:
                warnings.warn(f"Edge {edge} not found in graph")
                return {}
        
        # Calculate impact metrics
        impact = {
            'edge': edge,
            'segment_id_impact': segment_id if segment_id else f"{edge[0]}-{edge[1]}"
        }
        
        # Connectivity impact
        new_num_components = nx.number_connected_components(temp_graph)
        impact['components_increase'] = new_num_components - self.baseline_metrics['num_components']
        impact['disconnects_graph'] = new_num_components > self.baseline_metrics['num_components']
        
        # Size of largest component after removal
        if temp_graph.number_of_nodes() > 0:
            largest_cc_size = len(max(nx.connected_components(temp_graph), key=len))
            impact['largest_component_reduction'] = (
                self.baseline_metrics['largest_component_size'] - largest_cc_size
            ) / self.baseline_metrics['largest_component_size']
        else:
            impact['largest_component_reduction'] = 1.0
        
        # Calculate isolated nodes
        isolated_nodes = [n for n in temp_graph.nodes() if temp_graph.degree(n) == 0]
        impact['isolated_nodes'] = len(isolated_nodes)
        impact['isolated_nodes_ratio'] = len(isolated_nodes) / self.baseline_metrics['total_nodes']
        
        # Calculate reachability impact
        impact['reachability_impact'] = self._calculate_reachability_impact(temp_graph)
        
        # Get baseline betweenness for this edge
        if edge in self.baseline_metrics['betweenness']:
            impact['betweenness_centrality'] = self.baseline_metrics['betweenness'][edge]
        else:
            # Try reversed edge
            impact['betweenness_centrality'] = self.baseline_metrics['betweenness'].get(
                (edge[1], edge[0]), 0
            )
        
        # Calculate detour factor (increase in shortest paths)
        impact['detour_factor'] = self._calculate_detour_factor(edge, temp_graph)
        
        return impact
    
    def _calculate_reachability_impact(self, modified_graph: nx.Graph) -> float:
        """
        Calculate the impact on network reachability.
        
        Parameters:
        -----------
        modified_graph : nx.Graph
            Graph with edge removed
            
        Returns:
        --------
        float
            Reachability impact score (0-1)
        """
        # Count reachable node pairs in original graph
        original_reachable = 0
        for component in nx.connected_components(self.graph):
            n = len(component)
            original_reachable += n * (n - 1) / 2
        
        # Count reachable node pairs in modified graph
        modified_reachable = 0
        for component in nx.connected_components(modified_graph):
            n = len(component)
            modified_reachable += n * (n - 1) / 2
        
        if original_reachable > 0:
            return 1 - (modified_reachable / original_reachable)
        return 0
    
    def _calculate_detour_factor(self, edge: Tuple[int, int], 
                                modified_graph: nx.Graph) -> float:
        """
        Calculate the average detour factor for paths that used the removed edge.
        
        Parameters:
        -----------
        edge : Tuple[int, int]
            Removed edge
        modified_graph : nx.Graph
            Graph with edge removed
            
        Returns:
        --------
        float
            Average detour factor (>1 means longer paths)
        """
        u, v = edge
        
        # Check if nodes are still connected after edge removal
        if not (modified_graph.has_node(u) and modified_graph.has_node(v)):
            return float('inf')
        
        if nx.has_path(modified_graph, u, v):
            try:
                # Calculate new shortest path length
                new_length = nx.shortest_path_length(modified_graph, u, v, weight='weight')
                # Get original edge weight
                original_length = self.graph[u][v].get('weight', 1)
                return new_length / original_length
            except:
                return float('inf')
        else:
            return float('inf')
    
    def analyze_all_segments(self, parallel: bool = True, 
                           n_jobs: int = -1) -> pd.DataFrame:
        """
        Analyze the impact of removing each segment in the network.
        
        Parameters:
        -----------
        parallel : bool
            Whether to use parallel processing
        n_jobs : int
            Number of parallel jobs (-1 for all CPUs)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with impact analysis for all segments
        """
        # Ensure baseline metrics are calculated
        if not self.baseline_metrics:
            self.calculate_baseline_metrics()
        
        edges = list(self.graph.edges())
        results = []
        
        if parallel and len(edges) > 10:  # Only parallelize for larger networks
            # Parallel processing
            if n_jobs == -1:
                n_jobs = None  # Use all available CPUs
                
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # Submit all tasks
                future_to_edge = {
                    executor.submit(self._evaluate_edge_wrapper, edge): edge 
                    for edge in edges
                }
                
                # Collect results
                for future in as_completed(future_to_edge):
                    edge = future_to_edge[future]
                    try:
                        impact = future.result()
                        results.append(impact)
                    except Exception as exc:
                        warnings.warn(f"Edge {edge} generated exception: {exc}")
        else:
            # Sequential processing
            for edge in edges:
                impact = self.evaluate_segment_removal_impact(edge=edge)
                results.append(impact)
        
        # Convert to DataFrame
        df_impacts = pd.DataFrame(results)
        
        # Store results
        self.segment_impacts = df_impacts
        
        return df_impacts
    
    def _evaluate_edge_wrapper(self, edge: Tuple[int, int]) -> Dict[str, float]:
        """
        Wrapper function for parallel processing of edge evaluation.
        """
        return self.evaluate_segment_removal_impact(edge=edge)
    
    def classify_segments_by_relevance(self, df_impacts: pd.DataFrame = None,
                                     classification_method: str = 'composite') -> pd.DataFrame:
        """
        Classify road segments based on their network relevance.
        
        Parameters:
        -----------
        df_impacts : pd.DataFrame, optional
            Impact analysis results (if not provided, uses stored results)
        classification_method : str
            Method for classification: 'composite', 'betweenness', 'disruption'
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with segments classified by relevance
        """
        if df_impacts is None:
            df_impacts = self.segment_impacts
            
        if df_impacts.empty:
            warnings.warn("No impact analysis results available")
            return pd.DataFrame()
        
        df_classified = df_impacts.copy()
        
        if classification_method == 'composite':
            # Composite score based on multiple factors
            df_classified['relevance_score'] = (
                0.3 * df_classified['betweenness_centrality'] +
                0.2 * df_classified['components_increase'].clip(0, 1) +
                0.2 * df_classified['largest_component_reduction'] +
                0.15 * df_classified['isolated_nodes_ratio'] +
                0.15 * df_classified['reachability_impact']
            )
            
            # Handle infinite detour factors
            detour_normalized = df_classified['detour_factor'].copy()
            detour_normalized[detour_normalized == float('inf')] = detour_normalized[
                detour_normalized != float('inf')
            ].max() * 2 if len(detour_normalized[detour_normalized != float('inf')]) > 0 else 10
            detour_normalized = (detour_normalized - 1) / (detour_normalized.max() - 1)
            
            df_classified['relevance_score'] += 0.1 * detour_normalized
            
        elif classification_method == 'betweenness':
            df_classified['relevance_score'] = df_classified['betweenness_centrality']
            
        elif classification_method == 'disruption':
            df_classified['relevance_score'] = (
                df_classified['components_increase'] * 0.4 +
                df_classified['largest_component_reduction'] * 0.3 +
                df_classified['reachability_impact'] * 0.3
            )
        
        # Normalize relevance score to 0-1
        if df_classified['relevance_score'].max() > 0:
            df_classified['relevance_score'] = (
                df_classified['relevance_score'] / df_classified['relevance_score'].max()
            )
        
        # Classify into categories
        df_classified['relevance_class'] = pd.cut(
            df_classified['relevance_score'],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Critical']
        )
        
        # Sort by relevance score
        df_classified = df_classified.sort_values('relevance_score', ascending=False)
        
        return df_classified
    
    def export_relevance_to_geodataframe(self, road_segments: gpd.GeoDataFrame,
                                        df_relevance: pd.DataFrame,
                                        segment_id_col: str = 'segment_id') -> gpd.GeoDataFrame:
        """
        Export relevance analysis results back to GeoDataFrame.
        
        Parameters:
        -----------
        road_segments : gpd.GeoDataFrame
            Original road segments GeoDataFrame
        df_relevance : pd.DataFrame
            Relevance analysis results
        segment_id_col : str
            Column name for segment identifier
            
        Returns:
        --------
        gpd.GeoDataFrame
            Road segments with relevance attributes added
        """
        # Create a copy of the original GeoDataFrame
        gdf_result = road_segments.copy()
        
        # Merge relevance results
        relevance_cols = ['relevance_score', 'relevance_class', 'betweenness_centrality',
                         'disconnects_graph', 'components_increase', 
                         'largest_component_reduction', 'reachability_impact']
        
        # Create mapping from edge to segment_id if needed
        if segment_id_col not in df_relevance.columns and self.road_network:
            segment_mapping = {}
            for edge, seg_id in self.road_network.edge_to_segment.items():
                segment_mapping[f"{edge[0]}-{edge[1]}"] = seg_id
                segment_mapping[f"{edge[1]}-{edge[0]}"] = seg_id
            
            df_relevance['segment_id_mapped'] = df_relevance['segment_id_impact'].map(segment_mapping)
            merge_col = 'segment_id_mapped'
        else:
            merge_col = segment_id_col
        
        # Merge with original geodataframe
        gdf_result = gdf_result.merge(
            df_relevance[[merge_col] + relevance_cols],
            left_on=segment_id_col,
            right_on=merge_col,
            how='left'
        )
        
        # Fill NaN values for segments not analyzed (e.g., isolated segments)
        gdf_result['relevance_score'] = gdf_result['relevance_score'].fillna(0)
        gdf_result['relevance_class'] = gdf_result['relevance_class'].fillna('Very Low')
        
        return gdf_result


def calculate_network_exposure(gdf_segments: gpd.GeoDataFrame,
                             relevance_weight: float = 0.6,
                             intrinsic_weight: float = 0.4,
                             intrinsic_value_col: str = 'intrinsic_value_score') -> gpd.GeoDataFrame:
    """
    Calculate exposure factor based on network relevance and intrinsic value.
    
    Parameters:
    -----------
    gdf_segments : gpd.GeoDataFrame
        Road segments with relevance analysis and intrinsic value scores
    relevance_weight : float
        Weight for network relevance in exposure calculation
    intrinsic_weight : float
        Weight for intrinsic value in exposure calculation
    intrinsic_value_col : str
        Column name containing intrinsic value scores
        
    Returns:
    --------
    gpd.GeoDataFrame
        Road segments with exposure factor added
    """
    gdf_result = gdf_segments.copy()
    
    # Check if intrinsic value column exists
    if intrinsic_value_col not in gdf_result.columns:
        warnings.warn(f"Column '{intrinsic_value_col}' not found. Using segment length instead.")
        # Fallback to length-based approach
        gdf_result['intrinsic_normalized'] = (
            gdf_result.geometry.length / gdf_result.geometry.length.max()
        )
    else:
        # Normalize intrinsic value scores (1-5 scale) to 0-1
        gdf_result['intrinsic_normalized'] = (
            (gdf_result[intrinsic_value_col] - 1) / 4
        )
    
    # Calculate exposure factor
    gdf_result['exposure_factor'] = (
        relevance_weight * gdf_result['relevance_score'] +
        intrinsic_weight * gdf_result['intrinsic_normalized']
    )
    
    # Classify exposure
    gdf_result['exposure_class'] = pd.cut(
        gdf_result['exposure_factor'],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    return gdf_result