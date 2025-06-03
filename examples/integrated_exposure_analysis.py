"""
integrated_exposure_analysis.py

Example script demonstrating the integration of intrinsic value and network relevance
modules for comprehensive exposure assessment in rockfall risk analysis.

Author: Professional Data Scientist
Date: June 2025
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.road.network_analysis2 import RoadNetworkGraph, calculate_betweenness_centrality
from src.exposure.network_relevance2 import NetworkRelevanceAnalyzer, calculate_network_exposure
from src.exposure.intrinsic_value import IntrinsicValueCalculator


def main():
    """
    Main workflow for integrated road network exposure analysis.
    """
    
    # Step 1: Load road network data
    print("Loading road network data...")
    # Replace with your actual road network shapefile
    road_segments = gpd.read_file('data/input/tr_str.gpkg')
    
    # Ensure we have a OBJECTID column
    if 'OBJECTID' not in road_segments.columns:
        road_segments['OBJECTID'] = road_segments.index.astype(str)
    
    print(f"Loaded {len(road_segments)} road segments")
    print(f"Available columns: {road_segments.columns.tolist()}")
    
    # Step 2: Calculate intrinsic value based on road characteristics
    print("\nCalculating intrinsic value of road segments...")
    intrinsic_calculator = IntrinsicValueCalculator(
        type_weight=0.4,
        function_weight=0.3,
        condition_weight=0.2,
        toll_weight=0.1,
        type_column='tr_str_ty',
        function_column='tr_str_cf',
        condition_column='tr_str_sta',
        toll_column='tr_str_ped'
    )
    
    # Calculate intrinsic values
    road_segments = intrinsic_calculator.calculate_for_dataframe(
        road_segments,
        output_column='intrinsic_value_score'
    )
    
    print(f"Intrinsic value statistics:")
    print(road_segments['intrinsic_value_score'].describe())
    
    # Step 3: Create network graph
    print("\nCreating network graph from road segments...")
    road_network = RoadNetworkGraph(tolerance=0.001)  # 1 meter tolerance
    graph = road_network.create_graph_from_segments(
        road_segments, 
        segment_id_col='OBJECTID',
        weight_col='length'  # Use segment length as weight for routing
    )
    
    # Check network connectivity
    print(f"Network connected: {road_network.is_connected()}")
    print(f"Number of components: {len(road_network.get_network_components())}")
    
    # Get basic metrics
    metrics = road_network.calculate_basic_metrics()
    print("\nBasic Network Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Step 4: Analyze network relevance
    print("\nAnalyzing network relevance...")
    relevance_analyzer = NetworkRelevanceAnalyzer(graph, road_network)
    
    # Calculate baseline metrics
    baseline = relevance_analyzer.calculate_baseline_metrics()
    print(f"Baseline average shortest path: {baseline['avg_shortest_path']:.2f}")
    
    # Analyze all segments (use parallel processing for large networks)
    print("Evaluating impact of removing each segment...")
    df_impacts = relevance_analyzer.analyze_all_segments(parallel=True, n_jobs=-1)
    
    # Step 5: Classify segments by relevance
    print("Classifying segments by network relevance...")
    df_classified = relevance_analyzer.classify_segments_by_relevance(
        df_impacts, 
        classification_method='composite'
    )
    
    # Print top 10 most critical segments
    print("\nTop 10 Most Critical Road Segments (Network Relevance):")
    print(df_classified[['OBJECTID', 'relevance_score', 'relevance_class', 
                        'betweenness_centrality', 'disconnects_graph']].head(10))
    
    # Step 6: Export network relevance results back to GeoDataFrame
    print("\nIntegrating network relevance with intrinsic value...")
    gdf_with_relevance = relevance_analyzer.export_relevance_to_geodataframe(
        road_segments, 
        df_classified,
        segment_id_col='OBJECTID'
    )
    
    # Step 7: Calculate integrated exposure factor
    print("Calculating integrated exposure factors...")
    gdf_final = calculate_network_exposure(
        gdf_with_relevance,
        relevance_weight=0.6,  # 60% weight on network importance
        intrinsic_weight=0.4,  # 40% weight on physical characteristics
        intrinsic_value_col='intrinsic_value_score'
    )
    
    # Save results
    print("\nSaving results...")
    gdf_final.to_file('road_segments_with_integrated_exposure.shp')
    
    # Also save as CSV for further analysis
    df_results = pd.DataFrame(gdf_final.drop('geometry', axis=1))
    df_results.to_csv('integrated_exposure_analysis.csv', index=False)
    
    # Create comprehensive visualization
    print("Creating visualizations...")
    create_integrated_exposure_maps(gdf_final)
    create_exposure_correlation_plot(gdf_final)
    
    # Summary statistics
    print("\n" + "="*60)
    print("INTEGRATED EXPOSURE ANALYSIS SUMMARY")
    print("="*60)
    print("\nExposure Class Distribution:")
    print(gdf_final['exposure_class'].value_counts())
    print(f"\nTotal segments analyzed: {len(gdf_final)}")
    print(f"Critical segments (High/Very High exposure): {len(gdf_final[gdf_final['exposure_class'].isin(['High', 'Very High'])])}")
    
    # Identify segments with high exposure from both factors
    high_both = gdf_final[
        (gdf_final['relevance_score'] > 0.6) & 
        (gdf_final['intrinsic_value_score'] > 3.5)
    ]
    print(f"\nSegments with both high network relevance AND high intrinsic value: {len(high_both)}")
    
    # Export critical segments separately
    if len(high_both) > 0:
        high_both.to_file('critical_exposure_segments.shp')
        print("Critical segments saved to 'critical_exposure_segments.shp'")
    
    return gdf_final


def create_integrated_exposure_maps(gdf: gpd.GeoDataFrame):
    """
    Create comprehensive map visualizations of the exposure analysis.
    
    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        Road segments with integrated exposure analysis
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    # Map 1: Intrinsic Value
    gdf.plot(column='intrinsic_value_score', 
             cmap='YlOrRd', 
             linewidth=2,
             legend=True,
             ax=axes[0],
             legend_kwds={'label': 'Score (1-5)'})
    axes[0].set_title('Intrinsic Value Score\n(Road Type, Function, Condition, Toll)', fontsize=14)
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    
    # Map 2: Network Relevance
    gdf.plot(column='relevance_score', 
             cmap='RdYlBu_r', 
             linewidth=2,
             legend=True,
             ax=axes[1],
             legend_kwds={'label': 'Score (0-1)'})
    axes[1].set_title('Network Relevance Score\n(Connectivity Impact)', fontsize=14)
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    
    # Map 3: Integrated Exposure Factor
    gdf.plot(column='exposure_factor',
             cmap='RdYlBu_r',
             linewidth=2,
             legend=True,
             ax=axes[2],
             legend_kwds={'label': 'Factor (0-1)'})
    axes[2].set_title('Integrated Exposure Factor\n(Combined Analysis)', fontsize=14)
    axes[2].set_xlabel('Longitude')
    axes[2].set_ylabel('Latitude')
    
    # Map 4: Critical segments only
    critical_segments = gdf[gdf['exposure_class'].isin(['High', 'Very High'])]
    gdf.plot(color='lightgray', linewidth=0.5, ax=axes[3])
    if len(critical_segments) > 0:
        critical_segments.plot(
            column='exposure_factor',
            cmap='Reds',
            linewidth=3,
            legend=True,
            ax=axes[3],
            legend_kwds={'label': 'Exposure Factor'}
        )
    axes[3].set_title('Critical Exposure Segments\n(High & Very High Only)', fontsize=14)
    axes[3].set_xlabel('Longitude')
    axes[3].set_ylabel('Latitude')
    
    plt.tight_layout()
    plt.savefig('integrated_exposure_analysis_maps.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_exposure_correlation_plot(gdf: gpd.GeoDataFrame):
    """
    Create scatter plot showing correlation between intrinsic value and network relevance.
    
    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        Road segments with exposure analysis
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter plot of intrinsic vs network relevance
    scatter = ax1.scatter(
        gdf['intrinsic_value_score'],
        gdf['relevance_score'],
        c=gdf['exposure_factor'],
        cmap='RdYlBu_r',
        alpha=0.6,
        s=50
    )
    ax1.set_xlabel('Intrinsic Value Score (1-5)', fontsize=12)
    ax1.set_ylabel('Network Relevance Score (0-1)', fontsize=12)
    ax1.set_title('Exposure Factor Components Correlation', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Exposure Factor', fontsize=12)
    
    # Add quadrant labels
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=3, color='gray', linestyle='--', alpha=0.5)
    ax1.text(1.5, 0.8, 'Low Intrinsic\nHigh Network', ha='center', va='center', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax1.text(4.5, 0.8, 'High Both\n(Critical)', ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    ax1.text(1.5, 0.2, 'Low Both', ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax1.text(4.5, 0.2, 'High Intrinsic\nLow Network', ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Box plot of exposure factors by class
    exposure_data = []
    for class_name in ['Very Low', 'Low', 'Medium', 'High', 'Very High']:
        class_data = gdf[gdf['exposure_class'] == class_name]['exposure_factor'].values
        for val in class_data:
            exposure_data.append({'Exposure Class': class_name, 'Exposure Factor': val})
    
    df_box = pd.DataFrame(exposure_data)
    if len(df_box) > 0:
        sns.boxplot(data=df_box, x='Exposure Class', y='Exposure Factor', ax=ax2)
        ax2.set_title('Exposure Factor Distribution by Class', fontsize=14)
        ax2.set_ylabel('Exposure Factor', fontsize=12)
        ax2.set_xlabel('Exposure Class', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('exposure_correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def generate_priority_report(gdf: gpd.GeoDataFrame, top_n: int = 20):
    """
    Generate a priority report for rockfall protection measures.
    
    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        Road segments with exposure analysis
    top_n : int
        Number of top priority segments to report
    """
    # Sort by exposure factor
    priority_segments = gdf.nlargest(top_n, 'exposure_factor')
    
    print("\n" + "="*80)
    print(f"TOP {top_n} PRIORITY SEGMENTS FOR ROCKFALL PROTECTION")
    print("="*80)
    
    for idx, segment in priority_segments.iterrows():
        print(f"\nOBJECTID: {segment['OBJECTID']}")
        print(f"  Exposure Factor: {segment['exposure_factor']:.3f}")
        print(f"  Exposure Class: {segment['exposure_class']}")
        print(f"  Network Relevance: {segment['relevance_score']:.3f} ({segment['relevance_class']})")
        print(f"  Intrinsic Value: {segment['intrinsic_value_score']:.1f}/5")
        print(f"  Critical Features:")
        if segment.get('disconnects_graph', False):
            print(f"    - DISCONNECTS NETWORK if removed")
        if segment.get('betweenness_centrality', 0) > 0.1:
            print(f"    - High traffic corridor (betweenness: {segment['betweenness_centrality']:.3f})")
        if segment.get('intrinsic_value_score', 0) >= 4:
            print(f"    - High-value infrastructure (type/function/condition)")
        print(f"  Length: {segment.geometry.length:.1f} meters")


if __name__ == "__main__":
    # Run the main integrated analysis
    gdf_results = main()
    
    # Generate priority report
    generate_priority_report(gdf_results, top_n=20)
    
    # # Optional: Export results for different stakeholders
    # # For emergency services - focus on network disruption
    # emergency_priority = gdf_results[gdf_results['disconnects_graph'] == True]
    # if len(emergency_priority) > 0:
    #     emergency_priority.to_file('emergency_priority_segments.shp')
    
    # # For maintenance planning - focus on high-value infrastructure
    # maintenance_priority = gdf_results[gdf_results['intrinsic_value_score'] >= 4]
    # if len(maintenance_priority) > 0:
    #     maintenance_priority.to_file('maintenance_priority_segments.shp')