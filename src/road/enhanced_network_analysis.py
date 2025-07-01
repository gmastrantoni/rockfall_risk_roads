"""
Enhanced Network Analysis Module with Progress Tracking and Performance Optimization

This module provides optimized functions for analyzing road network graphs, including
efficient creation of network graphs from road segments with comprehensive progress tracking.
"""

import networkx as nx
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString
from typing import Dict, List, Tuple, Optional, Union
import warnings
import logging
import time
from collections import defaultdict

# Import progress tracking utilities
from ..utils.progress_tracker import ProgressTracker, create_progress_tracker

# Try to import rtree for spatial indexing (optional optimization)
try:
    from rtree import index
    RTREE_AVAILABLE = True
except ImportError:
    RTREE_AVAILABLE = False


class EnhancedRoadNetworkGraph:
    """
    An enhanced class to create and manage road network graphs from geospatial data
    with optimized performance and comprehensive progress tracking.
    """
    
    def __init__(
        self, 
        tolerance: float = 0.001,
        enable_progress_tracking: bool = True,
        use_spatial_index: bool = True,
        batch_size: int = 1000,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Enhanced RoadNetworkGraph.
        
        Parameters:
        -----------
        tolerance : float
            Tolerance for considering two points as the same node (in map units)
        enable_progress_tracking : bool
            Whether to enable progress tracking
        use_spatial_index : bool
            Whether to use spatial indexing for faster node lookup
        batch_size : int
            Batch size for processing segments
        logger : logging.Logger, optional
            Logger instance
        """
        self.graph = nx.Graph()
        self.tolerance = tolerance
        self.enable_progress_tracking = enable_progress_tracking
        self.use_spatial_index = use_spatial_index and RTREE_AVAILABLE
        self.batch_size = batch_size
        self.logger = logger or logging.getLogger(__name__)
        
        # Node management
        self.nodes_dict = {}
        self.edge_to_segment = {}
        self.node_counter = 0
        
        # Spatial index for fast node lookup
        if self.use_spatial_index:
            try:
                self.spatial_index = index.Index()
                self.spatial_index_available = True
            except Exception as e:
                self.logger.warning(f"Could not create spatial index: {e}")
                self.spatial_index_available = False
                self.use_spatial_index = False
        else:
            self.spatial_index_available = False
            
        if not RTREE_AVAILABLE and use_spatial_index:
            self.logger.warning("rtree not available - install with 'pip install rtree' for faster processing")
    
    def create_graph_from_segments(
        self, 
        road_segments: gpd.GeoDataFrame,
        segment_id_col: str = 'segment_id',
        weight_col: Optional[str] = None,
        include_attributes: bool = True
    ) -> nx.Graph:
        """
        Create a network graph from road segments GeoDataFrame with optimization and progress tracking.
        
        Parameters:
        -----------
        road_segments : gpd.GeoDataFrame
            GeoDataFrame containing road segments with LineString geometries
        segment_id_col : str
            Column name containing unique segment identifiers
        weight_col : str, optional
            Column name to use as edge weight (e.g., 'length', 'travel_time')
        include_attributes : bool
            Whether to include all segment attributes as edge attributes
            
        Returns:
        --------
        nx.Graph
            NetworkX graph representing the road network
        """
        start_time = time.time()
        self.logger.info(f"Starting optimized graph creation from {len(road_segments)} road segments")
        
        # Clear existing graph
        self._reset_graph()
        
        # Create progress tracker if enabled
        if self.enable_progress_tracking:
            tracker = create_progress_tracker(
                total_items=4,  # Major steps
                description="Creating network graph",
                logger=self.logger
            )
            tracker.start()
        
        try:
            # Step 1: Validate and prepare data
            if self.enable_progress_tracking:
                tracker.update(1, "Validating and preparing segment data")
            
            valid_segments = self._validate_segments(road_segments, segment_id_col)
            self.logger.info(f"Processing {len(valid_segments)} valid segments")
            
            # Step 2: Extract nodes efficiently
            if self.enable_progress_tracking:
                tracker.update(1, "Extracting nodes from segments")
            
            self.logger.info("Extracting nodes from segments")
            nodes_data = self._extract_nodes_optimized(valid_segments, segment_id_col)
            
            # Step 3: Create nodes in graph
            if self.enable_progress_tracking:
                tracker.update(1, "Creating graph nodes")
            
            self._create_graph_nodes(nodes_data)
            
            # Step 4: Create edges in graph
            if self.enable_progress_tracking:
                tracker.update(1, "Creating graph edges")
            
            self._create_graph_edges(
                valid_segments, 
                segment_id_col, 
                weight_col, 
                include_attributes
            )
            
            if self.enable_progress_tracking:
                tracker.finish()
            
            # Log final statistics
            total_time = time.time() - start_time
            self.logger.info(f"Graph creation completed in {total_time:.2f}s")
            self.logger.info(f"Created graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            
            return self.graph
            
        except Exception as e:
            if self.enable_progress_tracking and 'tracker' in locals():
                tracker.finish()
            self.logger.error(f"Error in graph creation: {e}")
            raise
    
    def _reset_graph(self):
        """Reset all graph data structures."""
        self.graph.clear()
        self.nodes_dict.clear()
        self.edge_to_segment.clear()
        self.node_counter = 0
        
        if self.spatial_index_available:
            # Create new spatial index
            self.spatial_index = index.Index()
    
    def _validate_segments(self, road_segments: gpd.GeoDataFrame, segment_id_col: str) -> gpd.GeoDataFrame:
        """Validate and filter segments for graph creation."""
        # Check for required columns
        if segment_id_col not in road_segments.columns:
            raise ValueError(f"Segment ID column '{segment_id_col}' not found")
        
        # Filter valid LineString geometries
        valid_mask = road_segments.geometry.apply(
            lambda geom: isinstance(geom, LineString) and not geom.is_empty
        )
        
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            self.logger.warning(f"Skipping {invalid_count} invalid geometries")
        
        valid_segments = road_segments[valid_mask].copy()
        
        # Check for duplicate segment IDs
        if valid_segments[segment_id_col].duplicated().any():
            self.logger.warning("Duplicate segment IDs found, keeping first occurrence")
            valid_segments = valid_segments.drop_duplicates(subset=[segment_id_col])
        
        return valid_segments
    
    def _extract_nodes_optimized(
        self, 
        road_segments: gpd.GeoDataFrame, 
        segment_id_col: str
    ) -> Dict[int, Dict]:
        """Extract nodes from segments with optimized lookup."""
        nodes_data = {}
        segment_to_nodes = {}
        
        # Use coordinate-based lookup for better performance
        coord_to_node = {}  # Map coordinates to node IDs
        
        # Create progress tracker for segments
        if self.enable_progress_tracking:
            segment_tracker = create_progress_tracker(
                total_items=len(road_segments),
                description="Extracting segment endpoints",
                logger=self.logger
            )
            segment_tracker.start()
        
        try:
            for idx, (_, segment) in enumerate(road_segments.iterrows()):
                geom = segment.geometry
                segment_id = segment[segment_id_col]
                
                # Extract start and end points
                start_point = Point(geom.coords[0])
                end_point = Point(geom.coords[-1])
                
                # Get or create node IDs using optimized lookup
                start_node_id = self._get_or_create_node_optimized(start_point, nodes_data, coord_to_node)
                end_node_id = self._get_or_create_node_optimized(end_point, nodes_data, coord_to_node)
                
                # Store segment to nodes mapping
                segment_to_nodes[segment_id] = (start_node_id, end_node_id)
                
                if self.enable_progress_tracking and (idx + 1) % self.batch_size == 0:
                    segment_tracker.update(
                        self.batch_size, 
                        f"Processed {idx + 1}/{len(road_segments)} segments"
                    )
            
            # Final update
            if self.enable_progress_tracking:
                remaining = len(road_segments) % self.batch_size
                if remaining > 0:
                    segment_tracker.update(remaining)
                segment_tracker.finish()
            
            self.logger.info(f"Extracted {len(nodes_data)} unique nodes from {len(road_segments)} segments")
            
            # Store segment to nodes mapping for edge creation
            self.segment_to_nodes = segment_to_nodes
            
            return nodes_data
            
        except Exception as e:
            if self.enable_progress_tracking and 'segment_tracker' in locals():
                segment_tracker.finish()
            raise
    
    def _get_or_create_node_optimized(self, point: Point, nodes_data: Dict, coord_to_node: Dict) -> int:
        """Optimized node lookup using coordinate-based approach or spatial indexing."""
        if self.spatial_index_available:
            return self._get_or_create_node_spatial(point, nodes_data)
        else:
            return self._get_or_create_node_coordinate(point, nodes_data, coord_to_node)
    
    def _get_or_create_node_spatial(self, point: Point, nodes_data: Dict) -> int:
        """Use spatial index for fast node lookup."""
        # Query spatial index for nearby nodes
        candidates = list(self.spatial_index.intersection(
            (point.x - self.tolerance, point.y - self.tolerance,
             point.x + self.tolerance, point.y + self.tolerance)
        ))
        
        # Check actual distance for candidates
        for node_id in candidates:
            if node_id in self.nodes_dict:
                if point.distance(self.nodes_dict[node_id]) < self.tolerance:
                    return node_id
        
        # Create new node
        node_id = self.node_counter
        self.node_counter += 1
        
        self.nodes_dict[node_id] = point
        nodes_data[node_id] = {
            'id': node_id,
            'point': point,
            'coords': (point.x, point.y)
        }
        
        # Add to spatial index
        self.spatial_index.insert(node_id, (point.x, point.y, point.x, point.y))
        
        return node_id
    
    def _get_or_create_node_coordinate(self, point: Point, nodes_data: Dict, coord_to_node: Dict) -> int:
        """Use coordinate-based lookup for node creation (faster than distance calculation)."""
        # Round coordinates to tolerance precision for faster lookup
        precision = max(1, int(-np.log10(self.tolerance)))
        rounded_x = round(point.x, precision)
        rounded_y = round(point.y, precision)
        coord_key = (rounded_x, rounded_y)
        
        # Check if we've seen this coordinate before
        if coord_key in coord_to_node:
            return coord_to_node[coord_key]
        
        # Check nearby coordinates within tolerance
        for dx in [-self.tolerance, 0, self.tolerance]:
            for dy in [-self.tolerance, 0, self.tolerance]:
                nearby_key = (round(rounded_x + dx, precision), round(rounded_y + dy, precision))
                if nearby_key in coord_to_node:
                    existing_node_id = coord_to_node[nearby_key]
                    existing_point = nodes_data[existing_node_id]['point']
                    if point.distance(existing_point) < self.tolerance:
                        # Add this coordinate to the same node
                        coord_to_node[coord_key] = existing_node_id
                        return existing_node_id
        
        # Create new node
        node_id = self.node_counter
        self.node_counter += 1
        
        nodes_data[node_id] = {
            'id': node_id,
            'point': point,
            'coords': (point.x, point.y)
        }
        
        coord_to_node[coord_key] = node_id
        
        return node_id
    
    def _create_graph_nodes(self, nodes_data: Dict):
        """Create nodes in the NetworkX graph."""
        self.logger.info(f"Creating {len(nodes_data)} nodes in graph")
        
        for node_id, node_data in nodes_data.items():
            point = node_data['point']
            self.graph.add_node(
                node_id, 
                pos=(point.x, point.y), 
                geometry=point
            )
            self.nodes_dict[node_id] = point
    
    def _create_graph_edges(
        self, 
        road_segments: gpd.GeoDataFrame,
        segment_id_col: str,
        weight_col: Optional[str],
        include_attributes: bool
    ):
        """Create edges in the NetworkX graph with progress tracking."""
        self.logger.info(f"Creating {len(road_segments)} edges in graph")
        
        # Create progress tracker for edge creation
        if self.enable_progress_tracking:
            edge_tracker = create_progress_tracker(
                total_items=len(road_segments),
                description="Creating graph edges",
                logger=self.logger
            )
            edge_tracker.start()
        
        try:
            for idx, (_, segment) in enumerate(road_segments.iterrows()):
                segment_id = segment[segment_id_col]
                geom = segment.geometry
                
                # Get node IDs for this segment
                if segment_id in self.segment_to_nodes:
                    start_node, end_node = self.segment_to_nodes[segment_id]
                else:
                    self.logger.warning(f"No nodes found for segment {segment_id}")
                    continue
                
                # Create edge attributes
                edge_attrs = {
                    'segment_id': segment_id,
                    'length': geom.length
                }
                
                # Add weight
                if weight_col and weight_col in segment and pd.notna(segment[weight_col]):
                    edge_attrs['weight'] = float(segment[weight_col])
                else:
                    edge_attrs['weight'] = geom.length
                
                # Add geometry and other attributes selectively
                if include_attributes:
                    # Only add geometry if specifically needed (memory intensive)
                    edge_attrs['geometry'] = geom
                    
                    # Add important segment attributes
                    for col in ['hazard_score', 'vulnerability', 'exposure_factor', 'risk_class']:
                        if col in road_segments.columns and pd.notna(segment[col]):
                            edge_attrs[col] = segment[col]
                
                # Add edge to graph
                self.graph.add_edge(start_node, end_node, **edge_attrs)
                self.edge_to_segment[(start_node, end_node)] = segment_id
                
                # Progress update
                if self.enable_progress_tracking and (idx + 1) % self.batch_size == 0:
                    edge_tracker.update(
                        self.batch_size,
                        f"Created {idx + 1}/{len(road_segments)} edges"
                    )
            
            # Final progress update
            if self.enable_progress_tracking:
                remaining = len(road_segments) % self.batch_size
                if remaining > 0:
                    edge_tracker.update(remaining)
                edge_tracker.finish()
                
        except Exception as e:
            if self.enable_progress_tracking and 'edge_tracker' in locals():
                edge_tracker.finish()
            raise
    
    def get_edge_by_segment_id(self, segment_id: str) -> Optional[Tuple[int, int]]:
        """Get edge (node pair) corresponding to a segment ID."""
        for edge, seg_id in self.edge_to_segment.items():
            if seg_id == segment_id:
                return edge
        return None
    
    def get_network_components(self) -> List[List[int]]:
        """Get connected components of the network."""
        return list(nx.connected_components(self.graph))
    
    def is_connected(self) -> bool:
        """Check if the graph is fully connected."""
        return nx.is_connected(self.graph)
    
    def calculate_basic_metrics(self) -> Dict[str, Union[float, int]]:
        """Calculate basic network metrics with timing."""
        start_time = time.time()
        
        metrics = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'is_connected': self.is_connected(),
            'num_components': nx.number_connected_components(self.graph),
            'density': nx.density(self.graph)
        }
        
        if self.graph.number_of_nodes() > 0:
            degrees = dict(self.graph.degree())
            metrics['average_degree'] = sum(degrees.values()) / len(degrees)
            metrics['max_degree'] = max(degrees.values()) if degrees else 0
            metrics['min_degree'] = min(degrees.values()) if degrees else 0
        
        calculation_time = time.time() - start_time
        metrics['calculation_time'] = calculation_time
        
        self.logger.info(f"Basic metrics calculated in {calculation_time:.2f}s")
        
        return metrics


# Enhanced utility functions with progress tracking
def create_enhanced_graph_from_segments(
    road_segments: gpd.GeoDataFrame,
    segment_id_col: str = 'segment_id',
    weight_col: Optional[str] = None,
    tolerance: float = 0.001,
    enable_progress_tracking: bool = True,
    use_spatial_index: bool = True,
    logger: Optional[logging.Logger] = None
) -> Tuple[nx.Graph, 'EnhancedRoadNetworkGraph']:
    """
    Enhanced function to create a network graph from road segments with optimization.
    
    Parameters:
    -----------
    road_segments : gpd.GeoDataFrame
        GeoDataFrame containing road segments
    segment_id_col : str
        Column name for segment IDs
    weight_col : str, optional
        Column name for edge weights
    tolerance : float
        Tolerance for node merging
    enable_progress_tracking : bool
        Whether to enable progress tracking
    use_spatial_index : bool
        Whether to use spatial indexing
    logger : logging.Logger, optional
        Logger instance
        
    Returns:
    --------
    Tuple[nx.Graph, EnhancedRoadNetworkGraph]
        NetworkX graph and the graph builder instance
    """
    # Create enhanced graph builder
    graph_builder = EnhancedRoadNetworkGraph(
        tolerance=tolerance,
        enable_progress_tracking=enable_progress_tracking,
        use_spatial_index=use_spatial_index,
        logger=logger
    )
    
    # Create graph with optimization
    graph = graph_builder.create_graph_from_segments(
        road_segments=road_segments,
        segment_id_col=segment_id_col,
        weight_col=weight_col
    )
    
    return graph, graph_builder


def benchmark_graph_creation(
    road_segments: gpd.GeoDataFrame,
    segment_id_col: str = 'segment_id',
    logger: Optional[logging.Logger] = None
) -> Dict[str, float]:
    """
    Benchmark different graph creation methods.
    
    Parameters:
    -----------
    road_segments : gpd.GeoDataFrame
        Road segments to test
    segment_id_col : str
        Segment ID column
    logger : logging.Logger, optional
        Logger instance
        
    Returns:
    --------
    Dict[str, float]
        Timing results for different methods
    """
    logger = logger or logging.getLogger(__name__)
    logger.info(f"Benchmarking graph creation with {len(road_segments)} segments")
    
    results = {}
    
    # Test enhanced method without spatial index
    logger.info("Testing enhanced method without spatial index...")
    start_time = time.time()
    enhanced_builder = EnhancedRoadNetworkGraph(
        use_spatial_index=False,
        enable_progress_tracking=False
    )
    enhanced_graph = enhanced_builder.create_graph_from_segments(road_segments, segment_id_col)
    results['enhanced_no_index'] = time.time() - start_time
    
    # Test enhanced method with spatial index (if available)
    if RTREE_AVAILABLE:
        logger.info("Testing enhanced method with spatial index...")
        start_time = time.time()
        enhanced_spatial_builder = EnhancedRoadNetworkGraph(
            use_spatial_index=True,
            enable_progress_tracking=False
        )
        enhanced_spatial_graph = enhanced_spatial_builder.create_graph_from_segments(road_segments, segment_id_col)
        results['enhanced_with_index'] = time.time() - start_time
    else:
        logger.info("Skipping spatial index test (rtree not available)")
    
    # Log results
    logger.info("Benchmark results:")
    for method, time_taken in results.items():
        speedup = results['enhanced_no_index'] / time_taken if time_taken > 0 else 1
        logger.info(f"  {method}: {time_taken:.2f}s (speedup: {speedup:.1f}x)")
    
    return results


# Backward compatibility - keep original class available
from .network_analysis import RoadNetworkGraph
