"""
network_relevance_optimized.py

Optimized Network Relevance Module for Rockfall Risk Assessment
This module provides optimized functions for analyzing road network topology and calculating 
network relevance for exposure assessment with improved performance for large networks.

RANDOM SAMPLING
"""

import networkx as nx
import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import logging
import time
from tqdm import tqdm
import random
from collections import defaultdict

logger = logging.getLogger(__name__)


class OptimizedNetworkRelevanceAnalyzer:
    """
    Optimized analyzer for network relevance of road segments for exposure assessment.
    Designed to handle large networks efficiently using sampling and parallel processing.
    """
    
    def __init__(self, graph: nx.Graph, road_network: 'RoadNetworkGraph' = None,
                 max_network_size: int = 5000, sample_fraction: float = 0.1,
                 enable_parallel: bool = True, max_workers: int = None):
        """
        Initialize the OptimizedNetworkRelevanceAnalyzer.
        
        Parameters:
        -----------
        graph : nx.Graph
            NetworkX graph representing the road network
        road_network : RoadNetworkGraph, optional
            RoadNetworkGraph object for segment-edge mapping
        max_network_size : int
            Maximum network size before using sampling strategies
        sample_fraction : float
            Fraction of edges to sample for large networks (0.1 = 10%)
        enable_parallel : bool
            Whether to use parallel processing
        max_workers : int, optional
            Maximum number of parallel workers (None = auto-detect)
        """
        self.graph = graph.copy()
        self.road_network = road_network
        self.max_network_size = max_network_size
        self.sample_fraction = sample_fraction
        self.enable_parallel = enable_parallel
        # Ensure max_workers is int or None
        if max_workers is not None:
            try:
                self.max_workers = int(max_workers)
            except Exception:
                logger.warning(f"max_workers value '{max_workers}' could not be converted to int. Using None.")
                self.max_workers = None
        else:
            self.max_workers = None
        
        self.baseline_metrics = {}
        self.segment_impacts = {}
        
        # Network size info
        self.num_nodes = self.graph.number_of_nodes()
        self.num_edges = self.graph.number_of_edges()
        self.is_large_network = self.num_edges > max_network_size
        
        logger.info(f"Network size: {self.num_nodes} nodes, {self.num_edges} edges")
        if self.is_large_network:
            logger.info(f"Large network detected. Will use sampling strategy with {sample_fraction*100:.1f}% sample rate")
        
    def _get_largest_component(self) -> nx.Graph:
        # """Get the largest connected component of the graph."""
        if nx.is_connected(self.graph):
            return self.graph
        
        components = list(nx.connected_components(self.graph))
        largest_component = max(components, key=len)
        subgraph = self.graph.subgraph(largest_component).copy()
        
        logger.info(f"Using largest connected component: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
        return subgraph
    
    def calculate_baseline_metrics(self, timeout: float = 300.0) -> Dict[str, Union[float, Dict]]:
        """
        Calculate baseline network metrics with optimization for large networks.
        
        Parameters:
        -----------
        timeout : float
            Timeout in seconds for expensive calculations
            
        Returns:
        --------
        Dict[str, Union[float, Dict]]
            Dictionary of baseline metrics
        """
        logger.info("Calculating baseline network metrics")
        start_time = time.time()
        
        # Get largest connected component for calculations
        main_graph = self._get_largest_component()
        
        self.baseline_metrics = {
            'connected': nx.is_connected(self.graph),
            'num_components': nx.number_connected_components(self.graph),
            'largest_component_size': main_graph.number_of_nodes(),
            'total_nodes': self.num_nodes,
            'total_edges': self.num_edges,
            'density': nx.density(main_graph),
            'average_degree': sum(dict(main_graph.degree()).values()) / main_graph.number_of_nodes()
        }
        
        # Calculate average shortest path length with timeout
        try:
            if self.is_large_network:
                logger.info("Large network: using approximation for shortest path calculations")
                # Sample a subset of nodes for shortest path calculation
                sample_size = min(500, main_graph.number_of_nodes() // 10)
                sample_nodes = random.sample(list(main_graph.nodes()), sample_size)
                
                # Calculate average shortest path on sample
                path_lengths = []
                with tqdm(total=len(sample_nodes), desc="Sampling shortest paths") as pbar:
                    for node in sample_nodes:
                        try:
                            lengths = nx.single_source_shortest_path_length(
                                main_graph, node, cutoff=10
                            )
                            path_lengths.extend(lengths.values())
                            pbar.update(1)
                        except Exception as e:
                            logger.warning(f"Error calculating paths from node {node}: {e}")
                            continue
                
                if path_lengths:
                    self.baseline_metrics['avg_shortest_path'] = np.mean(path_lengths)
                else:
                    self.baseline_metrics['avg_shortest_path'] = float('inf')
            else:
                # For smaller networks, calculate exact value
                logger.info("Small network: calculating exact shortest path length")
                self.baseline_metrics['avg_shortest_path'] = nx.average_shortest_path_length(
                    main_graph, weight='weight'
                )
        except Exception as e:
            logger.warning(f"Could not calculate average shortest path: {e}")
            self.baseline_metrics['avg_shortest_path'] = float('inf')
        
        # Calculate betweenness centrality with optimization
        try:
            if self.is_large_network:
                logger.info("Large network: using sampling for betweenness centrality")
                # Use sampling for betweenness centrality
                k = min(500, main_graph.number_of_nodes() // 5)  # Sample size
                self.baseline_metrics['betweenness'] = nx.edge_betweenness_centrality(
                    main_graph, normalized=True, weight='weight', k=k
                )
                logger.info(f"Calculated betweenness centrality using {k} node samples")
            else:
                logger.info("Small network: calculating exact betweenness centrality")
                self.baseline_metrics['betweenness'] = nx.edge_betweenness_centrality(
                    main_graph, normalized=True, weight='weight'
                )
        except Exception as e:
            logger.error(f"Error calculating betweenness centrality: {e}")
            # Fallback to empty dict
            self.baseline_metrics['betweenness'] = {}
        
        elapsed_time = time.time() - start_time
        logger.info(f"Baseline metrics calculated in {elapsed_time:.2f} seconds")
        
        return self.baseline_metrics
    
    def _evaluate_single_edge_impact(self, edge: Tuple[int, int]) -> Dict[str, Union[float, str, Tuple]]:
        """
        Evaluate the impact of removing a single edge (optimized version).
        
        Parameters:
        -----------
        edge : Tuple[int, int]
            Edge to evaluate
            
        Returns:
        --------
        Dict[str, Union[float, str, Tuple]]
            Dictionary of impact metrics
        """
        # Create temporary graph with edge removed
        temp_graph = self.graph.copy()
        
        # Try to remove edge (handle both directions)
        edge_removed = False
        original_edge = edge
        
        if temp_graph.has_edge(*edge):
            temp_graph.remove_edge(*edge)
            edge_removed = True
        elif temp_graph.has_edge(edge[1], edge[0]):
            edge = (edge[1], edge[0])
            temp_graph.remove_edge(*edge)
            edge_removed = True
        
        if not edge_removed:
            return {
                'edge': original_edge,
                'segment_id_impact': f"{original_edge[0]}-{original_edge[1]}",
                'error': 'Edge not found in graph'
            }
        
        # Calculate impact metrics
        impact = {
            'edge': edge,
            'segment_id_impact': f"{edge[0]}-{edge[1]}"
        }
        
        # Basic connectivity metrics
        new_num_components = nx.number_connected_components(temp_graph)
        impact['components_increase'] = new_num_components - self.baseline_metrics['num_components']
        impact['disconnects_graph'] = new_num_components > self.baseline_metrics['num_components']
        
        # Size of largest component after removal
        if temp_graph.number_of_nodes() > 0:
            try:
                largest_cc_size = len(max(nx.connected_components(temp_graph), key=len))
                impact['largest_component_reduction'] = (
                    self.baseline_metrics['largest_component_size'] - largest_cc_size
                ) / max(self.baseline_metrics['largest_component_size'], 1)
            except:
                impact['largest_component_reduction'] = 0.0
        else:
            impact['largest_component_reduction'] = 1.0
        
        # Calculate isolated nodes
        isolated_nodes = [n for n in temp_graph.nodes() if temp_graph.degree(n) == 0]
        impact['isolated_nodes'] = len(isolated_nodes)
        impact['isolated_nodes_ratio'] = len(isolated_nodes) / max(self.baseline_metrics['total_nodes'], 1)
        
        # Get baseline betweenness for this edge
        if edge in self.baseline_metrics.get('betweenness', {}):
            impact['betweenness_centrality'] = self.baseline_metrics['betweenness'][edge]
        elif (edge[1], edge[0]) in self.baseline_metrics.get('betweenness', {}):
            impact['betweenness_centrality'] = self.baseline_metrics['betweenness'][(edge[1], edge[0])]
        else:
            impact['betweenness_centrality'] = 0.0
        
        # Simplified reachability impact (avoid expensive calculations)
        if impact['disconnects_graph']:
            impact['reachability_impact'] = 1.0  # Maximum impact if it disconnects
        else:
            impact['reachability_impact'] = impact['betweenness_centrality']  # Use betweenness as proxy
        
        # Simplified detour factor
        u, v = edge
        if temp_graph.has_node(u) and temp_graph.has_node(v):
            if nx.has_path(temp_graph, u, v):
                impact['detour_factor'] = 2.0  # Simplified: assume moderate detour
            else:
                impact['detour_factor'] = float('inf')  # No alternative path
        else:
            impact['detour_factor'] = float('inf')
        
        return impact
    
    def analyze_all_segments(self, progress_callback=None) -> pd.DataFrame:
        """
        Analyze the impact of removing each segment in the network (optimized).
        
        Parameters:
        -----------
        progress_callback : callable, optional
            Callback function for progress updates
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with impact analysis for all segments
        """
        # Ensure baseline metrics are calculated
        if not self.baseline_metrics:
            self.calculate_baseline_metrics()
        
        edges = list(self.graph.edges())
        
        # Use sampling for large networks
        if self.is_large_network:
            sample_size = max(100, int(len(edges) * self.sample_fraction))
            edges = random.sample(edges, min(sample_size, len(edges)))
            logger.info(f"Sampling {len(edges)} edges from {self.graph.number_of_edges()} total edges")
        
        results = []
        
        if self.enable_parallel and len(edges) > 50:
            logger.info(f"Using parallel processing with {self.max_workers or 'auto'} workers")
            
            # Use ThreadPoolExecutor for I/O bound operations
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_edge = {
                    executor.submit(self._evaluate_single_edge_impact, edge): edge 
                    for edge in edges
                }
                
                # Collect results with progress bar
                with tqdm(total=len(future_to_edge), desc="Analyzing network segments") as pbar:
                    for future in as_completed(future_to_edge):
                        edge = future_to_edge[future]
                        try:
                            impact = future.result(timeout=30)  # 30 second timeout per edge
                            if 'error' not in impact:
                                results.append(impact)
                        except Exception as exc:
                            logger.warning(f"Edge {edge} generated exception: {exc}")
                        finally:
                            pbar.update(1)
                            if progress_callback:
                                progress_callback(len(results), len(edges))
        
        else:
            # Sequential processing with progress bar
            logger.info("Using sequential processing")
            with tqdm(total=len(edges), desc="Analyzing network segments") as pbar:
                for edge in edges:
                    try:
                        impact = self._evaluate_single_edge_impact(edge)
                        if 'error' not in impact:
                            results.append(impact)
                    except Exception as e:
                        logger.warning(f"Error analyzing edge {edge}: {e}")
                    finally:
                        pbar.update(1)
                        if progress_callback:
                            progress_callback(len(results), len(edges))
        
        if not results:
            logger.warning("No valid results from network analysis")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df_impacts = pd.DataFrame(results)
        
        # If we used sampling, extrapolate results to full network
        if self.is_large_network and len(edges) < self.graph.number_of_edges():
            logger.info("Extrapolating sampled results to full network")
            df_impacts = self._extrapolate_results(df_impacts)
        
        # Store results
        self.segment_impacts = df_impacts
        
        logger.info(f"Network analysis completed: {len(df_impacts)} segments analyzed")
        return df_impacts
    
    def _extrapolate_results(self, sampled_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extrapolate sampled results to represent the full network.
        
        Parameters:
        -----------
        sampled_df : pd.DataFrame
            DataFrame with sampled analysis results
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with extrapolated results
        """
        if sampled_df.empty:
            return sampled_df
        
        # Create synthetic entries for non-sampled edges
        all_edges = list(self.graph.edges())
        sampled_edges = set()
        
        for _, row in sampled_df.iterrows():
            edge_str = row['segment_id_impact']
            sampled_edges.add(edge_str)
        
        # Find edges not in sample
        unsampled_edges = []
        for edge in all_edges:
            edge_str = f"{edge[0]}-{edge[1]}"
            edge_str_rev = f"{edge[1]}-{edge[0]}"
            if edge_str not in sampled_edges and edge_str_rev not in sampled_edges:
                unsampled_edges.append(edge)
        
        if not unsampled_edges:
            return sampled_df
        
        logger.info(f"Extrapolating results for {len(unsampled_edges)} unsampled edges")
        
        # Calculate statistical distributions from sampled data
        numeric_cols = ['betweenness_centrality', 'largest_component_reduction', 
                       'isolated_nodes_ratio', 'reachability_impact']
        
        # Create synthetic entries based on sampled statistics
        synthetic_rows = []
        for edge in unsampled_edges:
            synthetic_row = {
                'edge': edge,
                'segment_id_impact': f"{edge[0]}-{edge[1]}",
                'components_increase': 0,  # Most edges don't disconnect
                'disconnects_graph': False,
                'isolated_nodes': 0,
                'detour_factor': 1.5  # Default moderate detour
            }
            
            # Use statistical sampling for continuous variables
            for col in numeric_cols:
                if col in sampled_df.columns and not sampled_df[col].isna().all():
                    # Use mean with some random variation
                    mean_val = sampled_df[col].mean()
                    std_val = sampled_df[col].std()
                    synthetic_val = np.random.normal(mean_val, std_val * 0.1)  # Small variation
                    synthetic_row[col] = max(0, synthetic_val)  # Ensure non-negative
                else:
                    synthetic_row[col] = 0.0
            
            synthetic_rows.append(synthetic_row)
        
        # Combine sampled and synthetic results
        synthetic_df = pd.DataFrame(synthetic_rows)
        combined_df = pd.concat([sampled_df, synthetic_df], ignore_index=True)
        
        return combined_df
    
    def classify_segments_by_relevance(self, df_impacts: pd.DataFrame = None,
                                     classification_method: str = 'composite') -> pd.DataFrame:
        """
        Classify road segments based on their network relevance (optimized version).
        
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
            logger.warning("No impact analysis results available for classification")
            return pd.DataFrame()
        
        logger.info(f"Classifying {len(df_impacts)} segments by network relevance")
        df_classified = df_impacts.copy()
        
        if classification_method == 'composite':
            # Composite score based on multiple factors
            # Handle missing columns gracefully
            weights = {
                'betweenness_centrality': 0.3,
                'components_increase': 0.2,
                'largest_component_reduction': 0.2,
                'isolated_nodes_ratio': 0.15,
                'reachability_impact': 0.15
            }
            
            df_classified['relevance_score'] = 0.0
            
            for factor, weight in weights.items():
                if factor in df_classified.columns:
                    # Normalize the factor to 0-1 range
                    col_data = df_classified[factor].fillna(0)
                    if col_data.max() > col_data.min():
                        normalized = (col_data - col_data.min()) / (col_data.max() - col_data.min())
                    else:
                        normalized = col_data * 0  # All zeros if no variation
                    
                    df_classified['relevance_score'] += weight * normalized
                else:
                    logger.warning(f"Factor '{factor}' not found in impact data")
            
            # Handle detour factor separately
            if 'detour_factor' in df_classified.columns:
                detour_normalized = df_classified['detour_factor'].copy()
                detour_normalized[detour_normalized == float('inf')] = detour_normalized[
                    detour_normalized != float('inf')
                ].max() * 2 if len(detour_normalized[detour_normalized != float('inf')]) > 0 else 10
                
                if detour_normalized.max() > 1:
                    detour_normalized = (detour_normalized - 1) / (detour_normalized.max() - 1)
                    df_classified['relevance_score'] += 0.1 * detour_normalized
                    
        elif classification_method == 'betweenness':
            df_classified['relevance_score'] = df_classified['betweenness_centrality'].fillna(0)
            
        elif classification_method == 'disruption':
            # Focus on network disruption metrics
            disruption_score = (
                df_classified['components_increase'].fillna(0) * 0.4 +
                df_classified['largest_component_reduction'].fillna(0) * 0.3 +
                df_classified['reachability_impact'].fillna(0) * 0.3
            )
            df_classified['relevance_score'] = disruption_score
        
        # Normalize relevance score to 0-1
        if df_classified['relevance_score'].max() > 0:
            df_classified['relevance_score'] = (
                df_classified['relevance_score'] / df_classified['relevance_score'].max()
            )
        
        # Classify into categories
        df_classified['relevance_class'] = pd.cut(
            df_classified['relevance_score'],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Critical'],
            include_lowest=True
        )
        
        # Handle any NaN values in classification
        df_classified['relevance_class'] = df_classified['relevance_class'].fillna('Very Low')
        
        # Sort by relevance score
        df_classified = df_classified.sort_values('relevance_score', ascending=False)
        
        logger.info("Network relevance classification completed")
        return df_classified
    
    def export_relevance_to_geodataframe(self, road_segments: gpd.GeoDataFrame,
                                        df_relevance: pd.DataFrame,
                                        segment_id_col: str = 'segment_id') -> gpd.GeoDataFrame:
        """
        Export relevance analysis results back to GeoDataFrame (optimized version).
        
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
        logger.info("Exporting relevance results to GeoDataFrame")
        
        # Create a copy of the original GeoDataFrame
        gdf_result = road_segments.copy()
        
        # Relevance columns to add
        relevance_cols = ['relevance_score', 'relevance_class', 'betweenness_centrality',
                         'disconnects_graph', 'components_increase', 
                         'largest_component_reduction', 'reachability_impact']
        
        # Only include columns that exist in the relevance dataframe
        available_cols = [col for col in relevance_cols if col in df_relevance.columns]
        
        # Create mapping from edge to segment_id if needed
        if segment_id_col not in df_relevance.columns and self.road_network:
            logger.info("Creating segment ID mapping from edge information")
            segment_mapping = {}
            for edge, seg_id in self.road_network.edge_to_segment.items():
                segment_mapping[f"{edge[0]}-{edge[1]}"] = seg_id
                segment_mapping[f"{edge[1]}-{edge[0]}"] = seg_id
            
            df_relevance['segment_id_mapped'] = df_relevance['segment_id_impact'].map(segment_mapping)
            merge_col = 'segment_id_mapped'
        else:
            merge_col = 'segment_id_impact'  # Use the impact column directly
        
        # Create a mapping dictionary for faster lookup
        relevance_dict = {}
        for _, row in df_relevance.iterrows():
            key = row[merge_col] if merge_col in row else None
            if key:
                relevance_dict[key] = {col: row[col] for col in available_cols if col in row}
        
        # Apply relevance data to road segments
        for col in available_cols:
            gdf_result[col] = 0.0  # Initialize with default values
        
        # Map relevance data to segments
        for idx, segment in gdf_result.iterrows():
            segment_id = segment[segment_id_col]
            
            # Try different segment ID formats
            possible_keys = [
                str(segment_id),
                f"{segment_id}",
                f"segment_{segment_id}"
            ]
            
            relevance_data = None
            for key in possible_keys:
                if key in relevance_dict:
                    relevance_data = relevance_dict[key]
                    break
            
            if relevance_data:
                for col in available_cols:
                    if col in relevance_data:
                        gdf_result.loc[idx, col] = relevance_data[col]
        
        # Fill NaN values for segments not analyzed
        for col in available_cols:
            if col == 'relevance_score':
                gdf_result[col] = gdf_result[col].fillna(0.0)
            elif col == 'relevance_class':
                gdf_result[col] = gdf_result[col].fillna('Very Low')
            else:
                gdf_result[col] = gdf_result[col].fillna(0.0)
        
        logger.info(f"Relevance data exported to {len(gdf_result)} road segments")
        return gdf_result


def calculate_network_exposure_optimized(gdf_segments: gpd.GeoDataFrame,
                                        relevance_weight: float = 0.6,
                                        intrinsic_weight: float = 0.4,
                                        intrinsic_value_col: str = 'intrinsic_value_score',
                                        progress_callback=None) -> gpd.GeoDataFrame:
    """
    Calculate exposure factor based on network relevance and intrinsic value (optimized version).
    
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
    progress_callback : callable, optional
        Callback function for progress updates
        
    Returns:
    --------
    gpd.GeoDataFrame
        Road segments with exposure factor added
    """
    logger.info("Calculating optimized network exposure")
    gdf_result = gdf_segments.copy()
    
    # Check if intrinsic value column exists
    if intrinsic_value_col not in gdf_result.columns:
        logger.warning(f"Column '{intrinsic_value_col}' not found. Using segment length as fallback.")
        # Fallback to length-based approach
        if 'geometry' in gdf_result.columns:
            lengths = gdf_result.geometry.length
            gdf_result['intrinsic_normalized'] = lengths / lengths.max() if lengths.max() > 0 else 0
        else:
            gdf_result['intrinsic_normalized'] = 0.5  # Default medium value
    else:
        # Normalize intrinsic value scores (assuming 1-5 scale) to 0-1
        intrinsic_values = gdf_result[intrinsic_value_col].fillna(3.0)  # Default to medium
        gdf_result['intrinsic_normalized'] = (intrinsic_values - 1) / 4
        gdf_result['intrinsic_normalized'] = gdf_result['intrinsic_normalized'].clip(0, 1)
    
    # Calculate exposure factor
    relevance_scores = gdf_result['relevance_score'].fillna(0.0)
    gdf_result['exposure_factor'] = (
        relevance_weight * relevance_scores +
        intrinsic_weight * gdf_result['intrinsic_normalized']
    )
    
    # Classify exposure
    gdf_result['exposure_class'] = pd.cut(
        gdf_result['exposure_factor'],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
        include_lowest=True
    )
    
    # Handle any NaN values in classification
    gdf_result['exposure_class'] = gdf_result['exposure_class'].fillna('Very Low')
    
    # if progress_callback:
    #     progress_callback(len(gdf_result), len(gdf_result))
    
    logger.info("Optimized network exposure calculation completed")
    return gdf_result