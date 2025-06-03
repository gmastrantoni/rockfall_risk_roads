"""
exposure_analysis_utils.py

Utility functions for analyzing and visualizing integrated exposure assessment results.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import folium
from shapely.geometry import Point


def create_exposure_statistics_report(gdf: gpd.GeoDataFrame, 
                                    output_file: str = 'exposure_statistics.txt') -> Dict:
    """
    Generate comprehensive statistics report for exposure analysis.
    
    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        Road segments with exposure analysis
    output_file : str
        Output file path for the report
        
    Returns:
    --------
    Dict
        Dictionary containing key statistics
    """
    stats = {}
    
    # Overall statistics
    stats['total_segments'] = len(gdf)
    stats['total_length_km'] = gdf.geometry.length.sum() / 1000
    
    # Exposure class distribution
    exposure_dist = gdf['exposure_class'].value_counts().to_dict()
    stats['exposure_distribution'] = exposure_dist
    
    # Critical segments
    critical = gdf[gdf['exposure_class'].isin(['High', 'Very High'])]
    stats['critical_segments'] = len(critical)
    stats['critical_length_km'] = critical.geometry.length.sum() / 1000
    stats['critical_percentage'] = (len(critical) / len(gdf)) * 100
    
    # Network disruption potential
    disconnecting = gdf[gdf['disconnects_graph'] == True]
    stats['disconnecting_segments'] = len(disconnecting)
    
    # Average scores by road type (if available)
    if 'tr_str_ty' in gdf.columns:
        type_stats = gdf.groupby('tr_str_ty').agg({
            'exposure_factor': 'mean',
            'intrinsic_value_score': 'mean',
            'relevance_score': 'mean',
            'OBJECTID': 'count'
        }).round(3)
        stats['by_road_type'] = type_stats.to_dict()
    
    # Write report
    with open(output_file, 'w') as f:
        f.write("INTEGRATED EXPOSURE ASSESSMENT REPORT\n")
        f.write("="*50 + "\n\n")
        
        f.write("OVERALL STATISTICS\n")
        f.write(f"Total road segments: {stats['total_segments']}\n")
        f.write(f"Total network length: {stats['total_length_km']:.1f} km\n")
        f.write(f"Critical segments: {stats['critical_segments']} ({stats['critical_percentage']:.1f}%)\n")
        f.write(f"Critical length: {stats['critical_length_km']:.1f} km\n")
        f.write(f"Network-disconnecting segments: {stats['disconnecting_segments']}\n\n")
        
        f.write("EXPOSURE CLASS DISTRIBUTION\n")
        for class_name, count in sorted(exposure_dist.items()):
            percentage = (count / stats['total_segments']) * 100
            f.write(f"{class_name:12s}: {count:5d} ({percentage:5.1f}%)\n")
        
        if 'by_road_type' in stats:
            f.write("\n\nANALYSIS BY ROAD TYPE\n")
            f.write("Type | Count | Avg Exposure | Avg Intrinsic | Avg Relevance\n")
            f.write("-"*60 + "\n")
            for road_type, type_data in stats['by_road_type']['exposure_factor'].items():
                count = stats['by_road_type']['OBJECTID'][road_type]
                intrinsic = stats['by_road_type']['intrinsic_value_score'][road_type]
                relevance = stats['by_road_type']['relevance_score'][road_type]
                f.write(f"{road_type:4s} | {count:5d} | {type_data:12.3f} | "
                       f"{intrinsic:13.3f} | {relevance:13.3f}\n")
    
    print(f"Statistics report saved to: {output_file}")
    return stats


def create_interactive_exposure_map(gdf: gpd.GeoDataFrame, 
                                  output_file: str = 'exposure_map.html',
                                  focus_on_critical: bool = True) -> folium.Map:
    """
    Create an interactive Folium map of exposure assessment results.
    
    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        Road segments with exposure analysis
    output_file : str
        Output HTML file path
    focus_on_critical : bool
        Whether to focus on critical segments only
        
    Returns:
    --------
    folium.Map
        Interactive map object
    """
    # Convert to WGS84 for web mapping
    gdf_wgs84 = gdf.to_crs('EPSG:4326')
    
    # Calculate map center
    bounds = gdf_wgs84.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    # Create base map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='CartoDB positron')
    
    # Define color mapping
    color_map = {
        'Very Low': 'green',
        'Low': 'lightgreen',
        'Medium': 'orange',
        'High': 'red',
        'Very High': 'darkred'
    }
    
    # Add segments to map
    segments_to_plot = gdf_wgs84
    if focus_on_critical:
        segments_to_plot = gdf_wgs84[gdf_wgs84['exposure_class'].isin(['High', 'Very High'])]
    
    for idx, segment in segments_to_plot.iterrows():
        # Create popup text
        popup_text = f"""
        <b>OBJECT ID:</b> {segment['OBJECTID']}<br>
        <b>Exposure Class:</b> {segment['exposure_class']}<br>
        <b>Exposure Factor:</b> {segment['exposure_factor']:.3f}<br>
        <b>Network Relevance:</b> {segment['relevance_score']:.3f}<br>
        <b>Intrinsic Value:</b> {segment['intrinsic_value_score']:.1f}<br>
        <b>Disconnects Network:</b> {segment.get('disconnects_graph', False)}
        """

        # Get line coordinates (always as list of [lat, lon] for folium)
        coords = []
        geom = segment.geometry
        if geom.geom_type == 'LineString':
            coords = [[lat, lon] for lon, lat, *_ in geom.coords]
        elif geom.geom_type == 'MultiLineString':
            for linestring in geom.geoms:
                coords.extend([[lat, lon] for lon, lat, *_ in linestring.coords])
        else:
            continue  # skip if not a line

        # Add to map
        folium.PolyLine(
            coords,
            color=color_map.get(segment['exposure_class'], 'gray'),
            weight=4,
            opacity=0.8,
            popup=folium.Popup(popup_text, max_width=300)
        ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                top: 50px; right: 50px; width: 150px; height: 150px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p style="margin: 0;"><b>Exposure Level</b></p>
    <p style="margin: 0;"><span style="color: darkred;">━━</span> Very High</p>
    <p style="margin: 0;"><span style="color: red;">━━</span> High</p>
    <p style="margin: 0;"><span style="color: orange;">━━</span> Medium</p>
    <p style="margin: 0;"><span style="color: lightgreen;">━━</span> Low</p>
    <p style="margin: 0;"><span style="color: green;">━━</span> Very Low</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    m.save(output_file)
    print(f"Interactive map saved to: {output_file}")
    
    return m


def identify_protection_priorities(gdf: gpd.GeoDataFrame, 
                                 budget_constraint: Optional[float] = None,
                                 cost_per_km: float = 100000) -> gpd.GeoDataFrame:
    """
    Identify priority segments for rockfall protection based on exposure and budget.
    
    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        Road segments with exposure analysis
    budget_constraint : float, optional
        Available budget for protection measures
    cost_per_km : float
        Estimated cost per kilometer of protection
        
    Returns:
    --------
    gpd.GeoDataFrame
        Priority segments for protection
    """
    # Sort by exposure factor (highest first)
    gdf_sorted = gdf.sort_values('exposure_factor', ascending=False).copy()
    
    # Calculate protection cost for each segment
    gdf_sorted['protection_cost'] = (gdf_sorted.geometry.length / 1000) * cost_per_km
    
    if budget_constraint:
        # Select segments within budget
        gdf_sorted['cumulative_cost'] = gdf_sorted['protection_cost'].cumsum()
        priority_segments = gdf_sorted[gdf_sorted['cumulative_cost'] <= budget_constraint]
        
        print(f"Budget constraint: €{budget_constraint:,.0f}")
        print(f"Segments selected: {len(priority_segments)}")
        print(f"Total cost: €{priority_segments['protection_cost'].sum():,.0f}")
        print(f"Total length: {priority_segments.geometry.length.sum()/1000:.1f} km")
    else:
        # Return top 20% most critical segments
        n_priority = int(len(gdf_sorted) * 0.2)
        priority_segments = gdf_sorted.head(n_priority)
        
        print(f"Top 20% segments selected: {len(priority_segments)}")
        print(f"Estimated total cost: €{priority_segments['protection_cost'].sum():,.0f}")
    
    # Add priority ranking
    priority_segments['priority_rank'] = range(1, len(priority_segments) + 1)
    
    return priority_segments


def export_for_risk_model(gdf: gpd.GeoDataFrame, 
                         output_file: str = 'exposure_for_risk_model.csv') -> pd.DataFrame:
    """
    Export exposure data in format suitable for integration with risk models.
    
    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        Road segments with exposure analysis
    output_file : str
        Output CSV file path
        
    Returns:
    --------
    pd.DataFrame
        Formatted exposure data
    """
    # Select relevant columns for risk modeling
    risk_columns = [
        'OBJECTID',
        'exposure_factor',
        'exposure_class',
        'relevance_score',
        'intrinsic_value_score',
        'betweenness_centrality',
        'disconnects_graph',
        'length_m'
    ]
    
    # Create DataFrame
    df_risk = gdf.copy()
    df_risk['length_m'] = df_risk.geometry.length
    
    # Add centroid coordinates
    centroids = df_risk.geometry.centroid
    df_risk['centroid_x'] = centroids.x
    df_risk['centroid_y'] = centroids.y
    
    # Select columns (only those that exist)
    export_columns = [col for col in risk_columns if col in df_risk.columns]
    export_columns.extend(['centroid_x', 'centroid_y'])
    
    df_export = df_risk[export_columns]
    
    # Save to CSV
    df_export.to_csv(output_file, index=False)
    print(f"Exposure data exported to: {output_file}")
    
    return df_export


def compare_analysis_methods(gdf: gpd.GeoDataFrame) -> None:
    """
    Compare different approaches to exposure assessment.
    
    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        Road segments with integrated exposure analysis
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Plot 1: Distribution comparison
    data_compare = pd.DataFrame({
        'Network Only': gdf['relevance_score'],
        'Intrinsic Only': (gdf['intrinsic_value_score'] - 1) / 4,  # Normalize to 0-1
        'Integrated': gdf['exposure_factor']
    })
    data_compare.boxplot(ax=axes[0])
    axes[0].set_title('Exposure Score Distribution by Method')
    axes[0].set_ylabel('Normalized Score (0-1)')
    
    # Plot 2: Correlation matrix
    corr_data = gdf[['relevance_score', 'intrinsic_value_score', 'exposure_factor']].corr()
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=axes[1])
    axes[1].set_title('Correlation Between Exposure Components')
    
    # Plot 3: Ranking comparison
    gdf['rank_network'] = gdf['relevance_score'].rank(ascending=False)
    gdf['rank_intrinsic'] = gdf['intrinsic_value_score'].rank(ascending=False)
    gdf['rank_integrated'] = gdf['exposure_factor'].rank(ascending=False)
    
    axes[2].scatter(gdf['rank_network'], gdf['rank_integrated'], alpha=0.5, label='Network vs Integrated')
    axes[2].scatter(gdf['rank_intrinsic'], gdf['rank_integrated'], alpha=0.5, label='Intrinsic vs Integrated')
    axes[2].set_xlabel('Component Ranking')
    axes[2].set_ylabel('Integrated Ranking')
    axes[2].set_title('Ranking Comparison')
    axes[2].legend()
    axes[2].set_aspect('equal')
    
    # Plot 4: Top segments comparison
    top_n = 50
    top_network = set(gdf.nlargest(top_n, 'relevance_score')['OBJECTID'])
    top_intrinsic = set(gdf.nlargest(top_n, 'intrinsic_value_score')['OBJECTID'])
    top_integrated = set(gdf.nlargest(top_n, 'exposure_factor')['OBJECTID'])
    
    from matplotlib_venn import venn3
    venn3([top_network, top_intrinsic, top_integrated], 
          set_labels=('Network\nRelevance', 'Intrinsic\nValue', 'Integrated\nExposure'),
          ax=axes[3])
    axes[3].set_title(f'Overlap of Top {top_n} Segments by Method')
    
    plt.tight_layout()
    plt.savefig('exposure_method_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print comparison statistics
    print("\nMETHOD COMPARISON STATISTICS")
    print("="*50)
    print(f"Correlation network-intrinsic: {gdf['relevance_score'].corr(gdf['intrinsic_value_score']):.3f}")
    print(f"Segments in top {top_n} by all methods: {len(top_network & top_intrinsic & top_integrated)}")
    print(f"Unique to network relevance: {len(top_network - top_intrinsic - top_integrated)}")
    print(f"Unique to intrinsic value: {len(top_intrinsic - top_network - top_integrated)}")