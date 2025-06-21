"""
Visualization Utilities

This module provides functions for visualizing the results of rockfall risk assessment.
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from typing import Union, List, Tuple, Optional, Dict, Any
import pandas as pd


def plot_classified_segments(
    gdf: gpd.GeoDataFrame,
    column: str = 'risk_class_final',
    title: str = 'Road Segments Classification',
    cmap: Optional[Union[str, mcolors.Colormap]] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    legend: bool = True,
    basemap: bool = False
) -> plt.Figure:
    """
    Plot road segments colored by classification.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing road segments
    column : str, optional
        Column to use for coloring, by default 'risk_class_final'
    title : str, optional
        Title for the plot, by default 'Road Segments Classification'
    cmap : Optional[Union[str, mcolors.Colormap]], optional
        Colormap to use, by default None
    figsize : Tuple[int, int], optional
        Figure size (width, height) in inches, by default (12, 8)
    save_path : Optional[str], optional
        Path to save the figure, by default None
    legend : bool, optional
        Whether to show a legend, by default True
    basemap : bool, optional
        Whether to add a basemap, by default False

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define color mapping for common classification columns
    if column == 'risk_class_final' and cmap is None:
        # Define custom colors for risk classes
        class_colors = {
        'Very Low': '#1a9850',
        'Low': '#91cf60',
        'Moderate': '#ffffbf',
        'High': '#fc8d59',
        'Very High': '#d73027',
        'Runout Zone': '#d73027',
        'Area of Attention': "#3871f7",
        'Not at Risk': "#6a00ff"
        }
        
        # Handle unknown categories
        categories = gdf[column].unique()
        for cat in categories:
            if cat not in class_colors:
                class_colors[cat] = '#000000'  # Default to black
        
        # Plot each category separately with appropriate color
        for category, color in class_colors.items():
            category_gdf = gdf[gdf[column] == category]
            if not category_gdf.empty:
                category_gdf.plot(ax=ax, color=color, label=category)
    
    elif column in ['hazard_class', 'exposure_class', 'network_relevance_class']:
        # Define custom colors for classification
        class_colors = {
            'Very Low': '#1a9850',
            'Low': '#91cf60',
            'Moderate': '#ffffbf',
            'High': '#fc8d59',
            'Very High': '#d73027'
        }
    elif column == 'vulnerability':
        # Define custom colors for classification
        class_colors = {
            0: '#1a9850',
            1: '#d73027',
            }
        
        # Handle unknown categories
        categories = gdf[column].unique()
        for cat in categories:
            if cat not in class_colors:
                class_colors[cat] = '#000000'  # Default to black
        
        # Plot each category separately with appropriate color
        for category, color in class_colors.items():
            category_gdf = gdf[gdf[column] == category]
            if not category_gdf.empty:
                category_gdf.plot(ax=ax, color=color, label=category)
    
    else:
        # Use default colormap for other columns
        if cmap is None:
            cmap = 'viridis'
        
        gdf.plot(ax=ax, column=column, cmap=cmap, legend=legend)
    
    # Add basemap if requested
    if basemap:
        try:
            ctx = __import__('contextily')
            ctx.add_basemap(
                ax, 
                crs=gdf.crs.to_string(),
                source=ctx.providers.OpenStreetMap.Mapnik
            )
        except ImportError:
            print("Contextily not available for adding basemap. Install with: pip install contextily")
        except Exception as e:
            print(f"Error adding basemap: {e}")
    
    # Set title and add legend if applicable
    ax.set_title(title)
    
    if legend and column in ['risk_class_final', 'hazard_class', 'vulnerability', 'exposure_class', 'network_relevance_class']:
        # Manually create legend
        from matplotlib.lines import Line2D
        
        if column == 'risk_class_final':
            class_colors = {
                'Runout Zone': '#d73027',
                'Area of Attention': '#fdae61',
                'Not at Risk': '#1a9850'
            }
        else:
            class_colors = {
                'Very Low': '#1a9850',
                'Low': '#91cf60',
                'Moderate': '#ffffbf',
                'High': '#fc8d59',
                'Very High': '#d73027'
            }
        
        # Only include classes that are present in the data
        categories = gdf[column].unique()
        legend_elements = [
            Line2D([0], [0], color=class_colors[cat], lw=4, label=cat)
            for cat in categories if cat in class_colors
        ]
        
        ax.legend(handles=legend_elements, loc='best')
    
    # Remove axes
    ax.set_axis_off()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_hazard_parameters(
    gdf: gpd.GeoDataFrame,
    parameters: List[str] = ['susceptibility_max', 'velocity_max', 'energy_max'],
    titles: Optional[List[str]] = None,
    cmaps: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot hazard parameters for road segments.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing road segments with hazard parameters
    parameters : List[str], optional
        List of parameter columns to plot, by default ['susceptibility_max', 'velocity_max', 'energy_max']
    titles : Optional[List[str]], optional
        List of titles for each parameter, by default None
    cmaps : Optional[List[str]], optional
        List of colormaps for each parameter, by default None
    figsize : Tuple[int, int], optional
        Figure size (width, height) in inches, by default (15, 5)
    save_path : Optional[str], optional
        Path to save the figure, by default None

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    # Validate input
    if not all(param in gdf.columns for param in parameters):
        missing = [param for param in parameters if param not in gdf.columns]
        raise ValueError(f"Parameters not found in GeoDataFrame: {missing}")
    
    # Set default titles if not provided
    if titles is None:
        titles = [param.replace('_', ' ').title() for param in parameters]
    
    # Set default colormaps if not provided
    if cmaps is None:
        cmaps = ['Reds', 'Blues', 'Greens']
        # Ensure we have enough colormaps
        if len(cmaps) < len(parameters):
            cmaps.extend(['viridis'] * (len(parameters) - len(cmaps)))
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, len(parameters), figsize=figsize)
    
    # Handle case where only one parameter is plotted
    if len(parameters) == 1:
        axes = [axes]
    
    # Plot each parameter
    for i, (param, title, cmap) in enumerate(zip(parameters, titles, cmaps)):
        gdf.plot(column=param, ax=axes[i], cmap=cmap, legend=True)
        axes[i].set_title(title)
        axes[i].set_axis_off()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_risk_matrix(
    gdf: gpd.GeoDataFrame,
    x_column: str = 'network_relevance_score',
    y_column: str = 'risk_score_normalized',
    x_label: str = 'Network Relevance',
    y_label: str = 'Risk Score',
    title: str = 'Risk vs Network Relevance',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a risk matrix.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing road segments with risk and relevance scores
    x_column : str, optional
        Column for x-axis, by default 'network_relevance_score'
    y_column : str, optional
        Column for y-axis, by default 'risk_score_normalized'
    x_label : str, optional
        Label for x-axis, by default 'Network Relevance'
    y_label : str, optional
        Label for y-axis, by default 'Risk Score'
    title : str, optional
        Title for the plot, by default 'Risk vs Network Relevance'
    figsize : Tuple[int, int], optional
        Figure size (width, height) in inches, by default (10, 8)
    save_path : Optional[str], optional
        Path to save the figure, by default None

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    # Validate input
    if x_column not in gdf.columns or y_column not in gdf.columns:
        missing = []
        if x_column not in gdf.columns:
            missing.append(x_column)
        if y_column not in gdf.columns:
            missing.append(y_column)
        raise ValueError(f"Columns not found in GeoDataFrame: {missing}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot
    scatter = ax.scatter(
        gdf[x_column],
        gdf[y_column],
        c=gdf[y_column],
        cmap='YlOrRd',
        alpha=0.7,
        s=30,
        edgecolors='black',
        linewidths=0.5
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(y_label)
    
    # Set axis labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set axis limits with some padding
    ax.set_xlim(
        max(0, gdf[x_column].min() - 0.5),
        min(5, gdf[x_column].max() + 0.5)
    )
    ax.set_ylim(
        max(0, gdf[y_column].min() - 0.5),
        min(5, gdf[y_column].max() + 0.5)
    )
    
    # Draw risk levels
    # Add background colors for risk zones
    ax.axhspan(0, 1.5, color='#1a9850', alpha=0.2, label='Very Low Risk')
    ax.axhspan(1.5, 2.5, color='#91cf60', alpha=0.2, label='Low Risk')
    ax.axhspan(2.5, 3.5, color='#ffffbf', alpha=0.2, label='Moderate Risk')
    ax.axhspan(3.5, 4.5, color='#fc8d59', alpha=0.2, label='High Risk')
    ax.axhspan(4.5, 5.0, color='#d73027', alpha=0.2, label='Very High Risk')
    
    # Add legend for risk zones
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1a9850', alpha=0.2, label='Very Low Risk'),
        Patch(facecolor='#91cf60', alpha=0.2, label='Low Risk'),
        Patch(facecolor='#ffffbf', alpha=0.2, label='Moderate Risk'),
        Patch(facecolor='#fc8d59', alpha=0.2, label='High Risk'),
        Patch(facecolor='#d73027', alpha=0.2, label='Very High Risk')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_summary_statistics(
    gdf: gpd.GeoDataFrame,
    class_column: str = 'risk_class_final',
    metric_column: str = 'segment_length',
    title: str = 'Summary by Risk Class',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot summary statistics for road segments.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing road segments
    class_column : str, optional
        Column containing classification, by default 'risk_class_final'
    metric_column : str, optional
        Column containing metric to summarize, by default 'segment_length'
    title : str, optional
        Title for the plot, by default 'Summary by Risk Class'
    figsize : Tuple[int, int], optional
        Figure size (width, height) in inches, by default (12, 6)
    save_path : Optional[str], optional
        Path to save the figure, by default None

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    # Validate input
    if class_column not in gdf.columns:
        raise ValueError(f"Class column '{class_column}' not found in GeoDataFrame")
    if metric_column not in gdf.columns:
        raise ValueError(f"Metric column '{metric_column}' not found in GeoDataFrame")
    
    # Calculate summary statistics
    summary = gdf.groupby(class_column)[metric_column].agg(['sum', 'count'])
    summary.columns = ['Total Length (m)', 'Count']
    summary['Percentage'] = (summary['Total Length (m)'] / summary['Total Length (m)'].sum()) * 100
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot total length
    summary['Total Length (m)'].plot(
        kind='bar',
        ax=ax1,
        color='skyblue',
        edgecolor='black'
    )
    ax1.set_title('Total Length by Class')
    ax1.set_ylabel('Length (m)')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(summary['Total Length (m)']):
        ax1.text(
            i,
            v + (summary['Total Length (m)'].max() * 0.02),
            f'{v:.0f}',
            ha='center',
            va='bottom',
            fontweight='bold'
        )
    
    # Plot percentage
    summary['Percentage'].plot(
        kind='pie',
        ax=ax2,
        autopct='%1.1f%%',
        startangle=90,
        shadow=False,
        explode=[0.05] * len(summary),
        wedgeprops={'edgecolor': 'black'}
    )
    ax2.set_title('Percentage by Class')
    ax2.set_ylabel('')
    
    # Set colors for specific risk classes
    if class_column == 'risk_class_final':
        colors = {
            'Runout Zone': '#d73027',
            'Area of Attention': '#fdae61',
            'Not at Risk': '#1a9850'
        }
        
        # Update bar colors
        bars = ax1.patches
        for i, bar in enumerate(bars):
            class_name = summary.index[i]
            if class_name in colors:
                bar.set_facecolor(colors[class_name])
        
        # Update pie chart colors
        wedges = ax2.patches
        for i, wedge in enumerate(wedges):
            class_name = summary.index[i]
            if class_name in colors:
                wedge.set_facecolor(colors[class_name])
    
    # Set title
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def create_risk_component_map(
    gdf: gpd.GeoDataFrame,
    components: List[str] = ['hazard_class', 'vulnerability', 'exposure_class', 'risk_class_final'],
    titles: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (20, 15),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a multi-panel map showing risk components and final risk.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing road segments with risk components
    components : List[str], optional
        List of component columns to plot, by default ['hazard_class', 'vulnerability', 'exposure_class', 'risk_class_final']
    titles : Optional[List[str]], optional
        List of titles for each component, by default None
    figsize : Tuple[int, int], optional
        Figure size (width, height) in inches, by default (20, 15)
    save_path : Optional[str], optional
        Path to save the figure, by default None

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    # Validate input
    for component in components:
        if component not in gdf.columns:
            raise ValueError(f"Component column '{component}' not found in GeoDataFrame")
    
    # Set default titles if not provided
    if titles is None:
        titles = [component.replace('_class', '').replace('_', ' ').title() for component in components]
    
    # Calculate number of rows and columns for subplots
    n_components = len(components)
    if n_components <= 2:
        n_rows, n_cols = 1, n_components
    elif n_components <= 4:
        n_rows, n_cols = 2, 2
    else:
        n_rows = int(np.ceil(n_components / 3))
        n_cols = min(n_components, 3)
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes array for easier indexing
    if n_rows > 1 or n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Define color mapping for components
    class_colors = {
        'Very Low': '#1a9850',
        'Low': '#91cf60',
        'Moderate': '#ffffbf',
        'High': '#fc8d59',
        'Very High': '#d73027',
        'Runout Zone': '#d73027',
        'Area of Attention': "#3871f7",
        'Not at Risk': "#6a00ff"
    }
    
    # Plot each component
    for i, (component, title) in enumerate(zip(components, titles)):
        if i < len(axes):
            # Get unique categories
            categories = gdf[component].unique()
            
            # Plot each category separately with appropriate color
            for category in categories:
                category_gdf = gdf[gdf[component] == category]
                if not category_gdf.empty:
                    color = class_colors.get(category, '#000000')  # Default to black if not found
                    category_gdf.plot(ax=axes[i], color=color, label=category)
            
            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color=class_colors.get(cat, '#000000'), lw=4, label=cat)
                for cat in categories
            ]
            axes[i].legend(handles=legend_elements, loc='best')
            
            # Set title and remove axes
            axes[i].set_title(title)
            axes[i].set_axis_off()
    
    # Hide any unused subplots
    for i in range(len(components), len(axes)):
        axes[i].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig
