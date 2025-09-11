"""
Aggregate Statistics Generator for HDRI Analysis
Rewritten with clean architecture for better maintainability.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd


# ==============================
# Data Collection
# ==============================

def collect_experiment_metrics(experiment_dir: Path) -> Dict[str, List]:
    """Collect all metrics from an experiment directory."""
    metrics_data = {
        'hdri_names': [],
        # HDRI metrics
        'global_color': [],
        'global_intensity': [],
        'dc_color': [],
        'dominant_color': [],
        'area_intensity': [],
        'dominant_direction': [],
        # Person render metrics
        'person_0001_intensity': [],
        'person_0002_intensity': [],
        'person_0001_front_intensity': [],
        'person_0001_left_intensity': [],
        'person_0001_right_intensity': [],
        'person_0001_top_intensity': [],
        'person_0001_bottom_intensity': [],
        'person_0002_front_intensity': [],
        'person_0002_left_intensity': [],
        'person_0002_right_intensity': [],
        'person_0002_top_intensity': [],
        'person_0002_bottom_intensity': []
    }

    # Find all HDRI subdirectories
    subdirs_found = 0
    for hdri_dir in experiment_dir.iterdir():
        if hdri_dir.is_dir():
            subdirs_found += 1
            hdri_name = hdri_dir.name

            # Load naive metrics
            naive_metrics_path = hdri_dir / f"{hdri_name}_naive_metrics.json"
            if naive_metrics_path.exists():
                with open(naive_metrics_path, 'r') as f:
                    naive_data = json.load(f)
                    metrics_data['hdri_names'].append(hdri_name)
                    metrics_data['global_color'].append(naive_data.get('global_color', [0, 0, 0]))
                    metrics_data['global_intensity'].append(naive_data.get('global_intensity', 0.0))

            # Load SPH metrics
            sph_metrics_path = hdri_dir / f"{hdri_name}_sph_metrics.json"
            if sph_metrics_path.exists():
                with open(sph_metrics_path, 'r') as f:
                    sph_data = json.load(f)
                    # Only add if we haven't already added from naive metrics
                    if hdri_name not in metrics_data['hdri_names']:
                        metrics_data['hdri_names'].append(hdri_name)
                        metrics_data['global_color'].append([0, 0, 0])
                        metrics_data['global_intensity'].append(0.0)

                    metrics_data['dc_color'].append(sph_data.get('dc_color', [0, 0, 0]))
                    metrics_data['dominant_color'].append(sph_data.get('dominant_color', [0, 0, 0]))
                    metrics_data['area_intensity'].append(sph_data.get('area_intensity', [0, 0, 0]))
                    metrics_data['dominant_direction'].append(sph_data.get('dominant_direction', [0, 0, 0]))

            # Load render metrics
            render_metrics_path = hdri_dir / "blender_renders" / "render_metrics.json"
            if render_metrics_path.exists():
                with open(render_metrics_path, 'r') as f:
                    render_data = json.load(f)
                    # Only add if we haven't already added from previous metrics
                    if hdri_name not in metrics_data['hdri_names']:
                        metrics_data['hdri_names'].append(hdri_name)
                        metrics_data['global_color'].append([0, 0, 0])
                        metrics_data['global_intensity'].append(0.0)
                        metrics_data['dc_color'].append([0, 0, 0])
                        metrics_data['dominant_color'].append([0, 0, 0])
                        metrics_data['area_intensity'].append([0, 0, 0])
                        metrics_data['dominant_direction'].append([0, 0, 0])
                    
                    # Add person render metrics
                    metrics_data['person_0001_intensity'].append(render_data.get('person_0001_intensity', 0.0))
                    metrics_data['person_0002_intensity'].append(render_data.get('person_0002_intensity', 0.0))
                    metrics_data['person_0001_front_intensity'].append(render_data.get('person_0001_front_intensity', 0.0))
                    metrics_data['person_0001_left_intensity'].append(render_data.get('person_0001_left_intensity', 0.0))
                    metrics_data['person_0001_right_intensity'].append(render_data.get('person_0001_right_intensity', 0.0))
                    metrics_data['person_0001_top_intensity'].append(render_data.get('person_0001_top_intensity', 0.0))
                    metrics_data['person_0001_bottom_intensity'].append(render_data.get('person_0001_bottom_intensity', 0.0))
                    metrics_data['person_0002_front_intensity'].append(render_data.get('person_0002_front_intensity', 0.0))
                    metrics_data['person_0002_left_intensity'].append(render_data.get('person_0002_left_intensity', 0.0))
                    metrics_data['person_0002_right_intensity'].append(render_data.get('person_0002_right_intensity', 0.0))
                    metrics_data['person_0002_top_intensity'].append(render_data.get('person_0002_top_intensity', 0.0))
                    metrics_data['person_0002_bottom_intensity'].append(render_data.get('person_0002_bottom_intensity', 0.0))

    print(f"Collected metrics from {len(metrics_data['hdri_names'])} HDRIs out of {subdirs_found} directories")
    
    # Pad missing values to ensure consistency
    expected_count = len(metrics_data['hdri_names'])
    for key, values in metrics_data.items():
        if key != 'hdri_names' and len(values) != expected_count:
            print(f"WARNING: {key} has {len(values)} values, expected {expected_count}")
            while len(values) < expected_count:
                if 'color' in key or 'direction' in key:
                    values.append([0, 0, 0])
                else:
                    values.append(0.0)
    
    return metrics_data


def create_dataframe_from_metrics(metrics_data: Dict[str, List]) -> pd.DataFrame:
    """Convert metrics data to a pandas DataFrame for easier analysis."""
    data_dict = {}
    
    # Basic info
    data_dict['hdri_name'] = metrics_data['hdri_names']
    data_dict['global_intensity'] = metrics_data['global_intensity']
    
    # Unpack RGB values into separate columns
    for i, rgb_list in enumerate(metrics_data['global_color']):
        if i == 0:
            data_dict['global_r'] = []
            data_dict['global_g'] = []
            data_dict['global_b'] = []
        data_dict['global_r'].append(rgb_list[0] if len(rgb_list) > 0 else 0)
        data_dict['global_g'].append(rgb_list[1] if len(rgb_list) > 1 else 0)
        data_dict['global_b'].append(rgb_list[2] if len(rgb_list) > 2 else 0)
    
    # DC color
    for i, rgb_list in enumerate(metrics_data['dc_color']):
        if i == 0:
            data_dict['dc_r'] = []
            data_dict['dc_g'] = []
            data_dict['dc_b'] = []
        data_dict['dc_r'].append(rgb_list[0] if len(rgb_list) > 0 else 0)
        data_dict['dc_g'].append(rgb_list[1] if len(rgb_list) > 1 else 0)
        data_dict['dc_b'].append(rgb_list[2] if len(rgb_list) > 2 else 0)
    
    # Dominant color
    for i, rgb_list in enumerate(metrics_data['dominant_color']):
        if i == 0:
            data_dict['dominant_r'] = []
            data_dict['dominant_g'] = []
            data_dict['dominant_b'] = []
        data_dict['dominant_r'].append(rgb_list[0] if len(rgb_list) > 0 else 0)
        data_dict['dominant_g'].append(rgb_list[1] if len(rgb_list) > 1 else 0)
        data_dict['dominant_b'].append(rgb_list[2] if len(rgb_list) > 2 else 0)
    
    # Area intensity
    for i, rgb_list in enumerate(metrics_data['area_intensity']):
        if i == 0:
            data_dict['area_r'] = []
            data_dict['area_g'] = []
            data_dict['area_b'] = []
        data_dict['area_r'].append(rgb_list[0] if len(rgb_list) > 0 else 0)
        data_dict['area_g'].append(rgb_list[1] if len(rgb_list) > 1 else 0)
        data_dict['area_b'].append(rgb_list[2] if len(rgb_list) > 2 else 0)
    
    # Dominant direction
    for i, dir_list in enumerate(metrics_data['dominant_direction']):
        if i == 0:
            data_dict['dominant_direction_x'] = []
            data_dict['dominant_direction_y'] = []
            data_dict['dominant_direction_z'] = []
        data_dict['dominant_direction_x'].append(dir_list[0] if len(dir_list) > 0 else 0)
        data_dict['dominant_direction_y'].append(dir_list[1] if len(dir_list) > 1 else 0)
        data_dict['dominant_direction_z'].append(dir_list[2] if len(dir_list) > 2 else 0)
    
    # Person render metrics (already scalar)
    person_metrics = [
        'person_0001_intensity', 'person_0002_intensity',
        'person_0001_front_intensity', 'person_0001_left_intensity', 
        'person_0001_right_intensity', 'person_0001_top_intensity', 'person_0001_bottom_intensity',
        'person_0002_front_intensity', 'person_0002_left_intensity', 
        'person_0002_right_intensity', 'person_0002_top_intensity', 'person_0002_bottom_intensity'
    ]
    
    for metric in person_metrics:
        data_dict[metric] = metrics_data.get(metric, [0.0] * len(metrics_data['hdri_names']))
    
    return pd.DataFrame(data_dict)


# ==============================
# Visualization Data Preparation
# ==============================

def create_histogram_data(series: pd.Series, bins: int = 20) -> Tuple[List[float], List[int]]:
    """Create histogram data from pandas series."""
    if len(series.dropna()) == 0:
        return [], []
    
    counts, bin_edges = np.histogram(series.dropna(), bins=bins)
    bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
    return bin_centers, counts.tolist()


def create_rgb_histogram_data(df: pd.DataFrame, rgb_cols: List[str], bins: int = 20) -> Dict[str, Tuple[List[float], List[int]]]:
    """Create RGB histogram data from DataFrame columns."""
    result = {}
    colors = ['r', 'g', 'b']
    
    for i, color in enumerate(colors):
        if i < len(rgb_cols):
            col_name = rgb_cols[i]
            if col_name in df.columns:
                bin_centers, counts = create_histogram_data(df[col_name], bins)
                result[color] = (bin_centers, counts)
            else:
                result[color] = ([], [])
        else:
            result[color] = ([], [])
    
    return result


def rgb_to_lab(r, g, b):
    """Convert RGB to CIELAB color space."""
    # Normalize RGB values
    r, g, b = r/255.0, g/255.0, b/255.0
    
    # Convert to XYZ
    if r > 0.04045:
        r = ((r + 0.055) / 1.055) ** 2.4
    else:
        r = r / 12.92
    if g > 0.04045:
        g = ((g + 0.055) / 1.055) ** 2.4
    else:
        g = g / 12.92
    if b > 0.04045:
        b = ((b + 0.055) / 1.055) ** 2.4
    else:
        b = b / 12.92
    
    r, g, b = r * 100, g * 100, b * 100
    
    # Observer = 2°, Illuminant = D65
    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505
    
    # Normalize
    x, y, z = x/95.047, y/100.000, z/108.883
    
    # Convert to Lab
    if x > 0.008856:
        x = x ** (1/3)
    else:
        x = (7.787 * x) + (16/116)
    if y > 0.008856:
        y = y ** (1/3)
    else:
        y = (7.787 * y) + (16/116)
    if z > 0.008856:
        z = z ** (1/3)
    else:
        z = (7.787 * z) + (16/116)
    
    l = (116 * y) - 16
    a = 500 * (x - y)
    b = 200 * (y - z)
    
    return l, a, b


def prepare_visualization_data(df: pd.DataFrame) -> Dict:
    """Prepare all visualization data from DataFrame."""
    viz_data = {}
    
    # Global intensity histogram
    viz_data['global_intensity'] = create_histogram_data(df['global_intensity'])
    
    # Area intensity RGB histogram  
    viz_data['area_intensity'] = create_rgb_histogram_data(df, ['area_r', 'area_g', 'area_b'])
    
    # Person render visualizations
    if 'person_0001_intensity' in df.columns and 'person_0002_intensity' in df.columns:
        viz_data['person_average'] = {
            'person_0001': create_histogram_data(df['person_0001_intensity']),
            'person_0002': create_histogram_data(df['person_0002_intensity'])
        }
        
        # Person directional intensities
        person1_views = ['front', 'left', 'right', 'top', 'bottom']
        person2_views = ['front', 'left', 'right', 'top', 'bottom']
        
        viz_data['person_0001_directional'] = {}
        for view in person1_views:
            col_name = f'person_0001_{view}_intensity'
            if col_name in df.columns:
                viz_data['person_0001_directional'][view] = create_histogram_data(df[col_name])
        
        viz_data['person_0002_directional'] = {}
        for view in person2_views:
            col_name = f'person_0002_{view}_intensity'
            if col_name in df.columns:
                viz_data['person_0002_directional'][view] = create_histogram_data(df[col_name])
    
    # Prepare 3D color visualizations - CIELAB
    color_types = ['global', 'dc', 'dominant']
    for color_type in color_types:
        r_col = f'{color_type}_r'
        g_col = f'{color_type}_g'
        b_col = f'{color_type}_b'
        
        if all(col in df.columns for col in [r_col, g_col, b_col]):
            lab_data = []
            for idx, row in df.iterrows():
                l, a, b = rgb_to_lab(row[r_col], row[g_col], row[b_col])
                lab_data.append({
                    'name': row['hdri_name'],
                    'L': l, 'a': a, 'b': b,
                    'rgb': [int(row[r_col]), int(row[g_col]), int(row[b_col])]
                })
            viz_data[f'{color_type}_color_lab'] = lab_data
    
    # Prepare 3D color visualizations - RGB
    for color_type in color_types:
        r_col = f'{color_type}_r'
        g_col = f'{color_type}_g'
        b_col = f'{color_type}_b'
        
        if all(col in df.columns for col in [r_col, g_col, b_col]):
            rgb_data = []
            for idx, row in df.iterrows():
                rgb_data.append({
                    'name': row['hdri_name'],
                    'r': int(row[r_col]),
                    'g': int(row[g_col]),
                    'b': int(row[b_col])
                })
            viz_data[f'{color_type}_color_3d'] = rgb_data
    
    # Prepare dominant direction 3D visualization
    if all(col in df.columns for col in ['dominant_direction_x', 'dominant_direction_y', 'dominant_direction_z']):
        direction_data = []
        for idx, row in df.iterrows():
            # Include dominant color for visualization
            dominant_color = [255, 255, 255]  # Default white
            if all(col in df.columns for col in ['dominant_r', 'dominant_g', 'dominant_b']):
                dominant_color = [int(row['dominant_r']), int(row['dominant_g']), int(row['dominant_b'])]
            
            direction_data.append({
                'name': row['hdri_name'],
                'x': row['dominant_direction_x'],
                'y': row['dominant_direction_y'],
                'z': row['dominant_direction_z'],
                'color': dominant_color
            })
        viz_data['dominant_direction_3d'] = direction_data
    
    return viz_data


# ==============================
# Statistics Calculation
# ==============================

def calculate_comprehensive_stats(df: pd.DataFrame) -> Dict:
    """Calculate comprehensive statistics from DataFrame."""
    stats = {}
    
    # Basic stats
    stats['total_hdris'] = len(df)
    
    # Global intensity statistics
    if 'global_intensity' in df.columns:
        stats['global_intensity'] = {
            'mean': float(df['global_intensity'].mean()),
            'std': float(df['global_intensity'].std()),
            'min': float(df['global_intensity'].min()),
            'max': float(df['global_intensity'].max()),
            'median': float(df['global_intensity'].median()),
            'q25': float(df['global_intensity'].quantile(0.25)),
            'q75': float(df['global_intensity'].quantile(0.75))
        }
    
    # Person render statistics
    person_metrics = ['person_0001_intensity', 'person_0002_intensity']
    person_directional_metrics = [
        'person_0001_front_intensity', 'person_0001_left_intensity', 'person_0001_right_intensity', 
        'person_0001_top_intensity', 'person_0001_bottom_intensity',
        'person_0002_front_intensity', 'person_0002_left_intensity', 'person_0002_right_intensity', 
        'person_0002_top_intensity', 'person_0002_bottom_intensity'
    ]
    
    for metric in person_metrics + person_directional_metrics:
        if metric in df.columns:
            stats[metric] = {
                'mean': float(df[metric].mean()),
                'std': float(df[metric].std()),
                'min': float(df[metric].min()),
                'max': float(df[metric].max()),
                'median': float(df[metric].median()),
                'q25': float(df[metric].quantile(0.25)),
                'q75': float(df[metric].quantile(0.75))
            }
    
    # Color statistics for RGB channels
    color_metrics = ['global', 'dc', 'dominant', 'area']
    for metric in color_metrics:
        r_col, g_col, b_col = f'{metric}_r', f'{metric}_g', f'{metric}_b'
        
        if all(col in df.columns for col in [r_col, g_col, b_col]):
            stats[f'{metric}_color'] = {
                'r': {
                    'mean': float(df[r_col].mean()),
                    'std': float(df[r_col].std()),
                    'min': float(df[r_col].min()),
                    'max': float(df[r_col].max())
                },
                'g': {
                    'mean': float(df[g_col].mean()),
                    'std': float(df[g_col].std()),
                    'min': float(df[g_col].min()),
                    'max': float(df[g_col].max())
                },
                'b': {
                    'mean': float(df[b_col].mean()),
                    'std': float(df[b_col].std()),
                    'min': float(df[b_col].min()),
                    'max': float(df[b_col].max())
                }
            }
    
    return stats


# ==============================
# JavaScript Generation (Clean)
# ==============================

def to_js_safe(obj):
    """Safely convert Python object to JavaScript-compatible JSON."""
    return json.dumps(obj)


def generate_3d_plot_js(viz_data: Dict) -> str:
    """Generate JavaScript code for visually appealing 3D plots with enhanced Plotly styling."""
    js_code = []
    
    # Create an enhanced 3D plot manager with better visual design
    js_code.append("""
    // Enhanced 3D Plot Manager with Beautiful Styling
    class Plot3DManager {
        constructor(plotId, data, config) {
            this.plotId = plotId;
            this.data = data;
            this.config = config;
            this.plot = null;
            
            // Store original state
            this.originalColors = this.data.map(d => this.config.getColor(d));
            this.originalSizes = this.config.markerSize;
            
            // Filtering state
            this.filteredIndices = new Set();
        }
        
        initialize() {
            // Create single trace for clean visualization
            const trace = {
                x: this.data.map(d => this.config.getX(d)),
                y: this.data.map(d => this.config.getY(d)),
                z: this.data.map(d => this.config.getZ(d)),
                mode: 'markers',
                type: 'scatter3d',
                name: 'Data Points',
                text: this.data.map(d => d.name),
                marker: {
                    size: this.originalSizes,
                    color: this.originalColors,
                    line: { 
                        color: 'rgba(0, 0, 0, 0.3)', 
                        width: 0.5 
                    },
                    opacity: 0.9,
                    sizemode: 'diameter'
                },
                hovertemplate: this.config.hoverTemplate || 
                              '<b>%{text}</b><br>' +
                              '(%{x}, %{y}, %{z})<extra></extra>'
            };
            
            const layout = {
                title: {
                    text: '',  // Title now shown in HTML
                    font: {
                        size: 18,
                        color: '#2c3e50',
                        family: 'Arial, sans-serif'
                    },
                    y: 0.98,
                    x: 0.5,
                    xanchor: 'center'
                },
                scene: {
                    xaxis: {
                        ...this.config.xaxis,
                        gridcolor: 'rgba(200, 200, 200, 0.3)',
                        zerolinecolor: 'rgba(200, 200, 200, 0.5)',
                        linecolor: 'rgba(200, 200, 200, 0.5)',
                        tickfont: { size: 12, color: '#7f8c8d' }
                    },
                    yaxis: {
                        ...this.config.yaxis,
                        gridcolor: 'rgba(200, 200, 200, 0.3)',
                        zerolinecolor: 'rgba(200, 200, 200, 0.5)',
                        linecolor: 'rgba(200, 200, 200, 0.5)',
                        tickfont: { size: 12, color: '#7f8c8d' }
                    },
                    zaxis: {
                        ...this.config.zaxis,
                        gridcolor: 'rgba(200, 200, 200, 0.3)',
                        zerolinecolor: 'rgba(200, 200, 200, 0.5)',
                        linecolor: 'rgba(200, 200, 200, 0.5)',
                        tickfont: { size: 12, color: '#7f8c8d' }
                    },
                    aspectmode: 'cube',
                    bgcolor: '#e3f2fd',
                    camera: {
                        eye: { x: 1.8, y: 0.8, z: 0.8 },
                        center: { x: 0, y: 0, z: 0 }
                    }
                },
                showlegend: false,
                margin: { l: 0, r: 0, b: 0, t: 0, pad: 0 },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                autosize: true,
                hoverlabel: {
                    bgcolor: 'rgba(255, 255, 255, 0.95)',
                    bordercolor: 'rgba(0, 0, 0, 0.2)',
                    font: { 
                        size: 13, 
                        color: '#2c3e50',
                        family: 'Arial, sans-serif'
                    },
                    align: 'left'
                }
            };
            
            const config = {
                responsive: true,
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['lasso3d', 'select3d'],
                toImageButtonOptions: {
                    format: 'png',
                    filename: this.config.title.replace(/\s+/g, '_'),
                    height: 800,
                    width: 800,
                    scale: 2
                }
            };
            
            this.plot = Plotly.newPlot(this.plotId, [trace], layout, config);
            
            // Force resize to fill container
            window.addEventListener('resize', () => {
                Plotly.Plots.resize(this.plotId);
            });
            
            // Initial resize
            setTimeout(() => {
                Plotly.Plots.resize(this.plotId);
            }, 100);
        }
        
        updateFiltering(filteredNames) {
            // Update colors and opacity - fully transparent for unselected
            const updatedColors = this.data.map((d, i) => {
                if (!filteredNames.includes(d.name)) {
                    return 'rgba(0, 0, 0, 0)';  // Completely transparent
                }
                return this.originalColors[i];  // Keep original color
            });
            
            const updatedOpacities = this.data.map(d => {
                return filteredNames.includes(d.name) ? 1.0 : 0.0;  // Full opacity for selected, none for unselected
            });
            
            // Update both colors and opacity
            Plotly.restyle(this.plotId, {
                'marker.color': [updatedColors],
                'marker.opacity': [updatedOpacities]
            });
        }
        
        resetFiltering() {
            // Reset to original colors and opacity
            Plotly.restyle(this.plotId, {
                'marker.color': [this.originalColors],
                'marker.opacity': 0.9  // Back to original opacity
            });
        }
    }
    """)
    
    # RGB 3D plots
    color_types = ['global', 'dc', 'dominant']
    for color_type in color_types:
        key = f'{color_type}_color_3d'
        if viz_data.get(key):
            js_code.append(f"""
            // {color_type.capitalize()} Color RGB Plot
            window.{color_type}Color3DManager = new Plot3DManager(
                '{color_type}Color3D',
                window.{color_type}Color3DData,
                {{
                    title: '{color_type.capitalize()} Color',
                    getX: d => d.r,
                    getY: d => d.g,
                    getZ: d => d.b,
                    getColor: d => `rgb(${{d.r}},${{d.g}},${{d.b}})`,
                    markerSize: 12,
                    hoverTemplate: '<b>%{{text}}</b><br>' +
                                  'RGB: (%{{x}}, %{{y}}, %{{z}})<extra></extra>',
                    shadowOffset: 0.02,
                    xaxis: {{ 
                        title: {{ text: 'Red', font: {{ size: 14 }} }}, 
                        range: [0, 255], 
                        dtick: 50, 
                        showgrid: true,
                        showspikes: false
                    }},
                    yaxis: {{ 
                        title: {{ text: 'Green', font: {{ size: 14 }} }}, 
                        range: [0, 255], 
                        dtick: 50, 
                        showgrid: true,
                        showspikes: false
                    }},
                    zaxis: {{ 
                        title: {{ text: 'Blue', font: {{ size: 14 }} }}, 
                        range: [0, 255], 
                        dtick: 50, 
                        showgrid: true,
                        showspikes: false
                    }}
                }}
            );
            """)
    
    # Dominant direction 3D plot
    if viz_data.get('dominant_direction_3d'):
        js_code.append(f"""
        // Dominant Direction 3D Plot
        window.dominantDirection3DManager = new Plot3DManager(
            'dominantDirection3D',
            window.dominantDirection3DData,
            {{
                title: 'Dominant Light Direction',
                getX: d => d.x,
                getY: d => d.y,
                getZ: d => d.z,
                    getColor: d => `rgb(${{d.color[0]}},${{d.color[1]}},${{d.color[2]}})`,
                    markerSize: 14,
                    hoverTemplate: '<b>%{{text}}</b><br>' +
                                  '(%{{x:.2f}}, %{{y:.2f}}, %{{z:.2f}})<extra></extra>',
                    shadowOffset: 0.01,
                    xaxis: {{ 
                        title: {{ text: 'X', font: {{ size: 14 }} }}, 
                        range: [-1, 1], 
                        dtick: 0.5, 
                        showgrid: true,
                        showspikes: false
                    }},
                    yaxis: {{ 
                        title: {{ text: 'Y', font: {{ size: 14 }} }}, 
                        range: [-1, 1], 
                        dtick: 0.5, 
                        showgrid: true,
                        showspikes: false
                    }},
                    zaxis: {{ 
                        title: {{ text: 'Z', font: {{ size: 14 }} }}, 
                        range: [-1, 1], 
                        dtick: 0.5, 
                        showgrid: true,
                        showspikes: false
                    }}
            }}
        );
        """)
    
    # Initialize all plots
    js_code.append("""
    // Initialize all 3D plots
    function initialize3DPlots() {
        if (window.globalColor3DManager) window.globalColor3DManager.initialize();
        if (window.dcColor3DManager) window.dcColor3DManager.initialize();
        if (window.dominantColor3DManager) window.dominantColor3DManager.initialize();
        if (window.dominantDirection3DManager) window.dominantDirection3DManager.initialize();
        
        // Force all plots to resize after initialization
        setTimeout(() => {
            ['globalColor3D', 'dcColor3D', 'dominantColor3D', 'dominantDirection3D'].forEach(id => {
                if (document.getElementById(id)) {
                    Plotly.Plots.resize(id);
                }
            });
        }, 200);
    }
    """)
    if js_code:
        wrapped_code = '\n'.join(js_code) + """
        
        // Initialize 3D plots after DOM is ready
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(function() {
                initialize3DPlots();
            }, 100); // Small delay to ensure Chart.js plots are rendered first
        });
        """
        return wrapped_code
    return ''


def generate_chart_js(viz_data: Dict, stats: Dict) -> str:
    """Generate JavaScript code for all charts."""
    js_code = []
    
    # Global intensity chart
    if viz_data.get('global_intensity') and viz_data['global_intensity'][0]:
        labels = [f"{x:.3f}" for x in viz_data['global_intensity'][0]]
        data = viz_data['global_intensity'][1]
        
        js_code.append(f"""
        // Global Intensity Chart
        var globalIntensityCtx = document.getElementById('globalIntensityChart').getContext('2d');
        window.globalIntensityChart = new Chart(globalIntensityCtx, {{
            type: 'bar',
            data: {{
                labels: {to_js_safe(labels)},
                datasets: [{{
                    label: 'Count',
                    data: {to_js_safe(data)},
                    backgroundColor: 'rgba(54, 162, 235, 0.6)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{ beginAtZero: true, title: {{ display: true, text: 'Count' }} }},
                    x: {{ title: {{ display: true, text: 'Global Intensity' }} }}
                }},
                plugins: {{
                    title: {{ display: true, text: 'Global Intensity Distribution' }}
                }}
            }}
        }});
        """)
    
    # Area intensity chart
    if viz_data.get('area_intensity') and viz_data['area_intensity']['r'][0]:
        r_labels = [f"{x:.2f}" for x in viz_data['area_intensity']['r'][0]]
        r_data = viz_data['area_intensity']['r'][1]
        g_data = viz_data['area_intensity']['g'][1] 
        b_data = viz_data['area_intensity']['b'][1]
        
        js_code.append(f"""
        // Area Intensity RGB Chart
        var areaIntensityCtx = document.getElementById('areaIntensityChart').getContext('2d');
        window.areaIntensityChart = new Chart(areaIntensityCtx, {{
            type: 'bar',
            data: {{
                labels: {to_js_safe(r_labels)},
                datasets: [
                    {{ label: 'Red', data: {to_js_safe(r_data)}, backgroundColor: 'rgba(255,99,132,0.6)', borderColor: 'rgba(255,99,132,1)', borderWidth: 1 }},
                    {{ label: 'Green', data: {to_js_safe(g_data)}, backgroundColor: 'rgba(75,192,192,0.6)', borderColor: 'rgba(75,192,192,1)', borderWidth: 1 }},
                    {{ label: 'Blue', data: {to_js_safe(b_data)}, backgroundColor: 'rgba(54,162,235,0.6)', borderColor: 'rgba(54,162,235,1)', borderWidth: 1 }}
                ]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{ beginAtZero: true, title: {{ display: true, text: 'Count' }} }},
                    x: {{ title: {{ display: true, text: 'Area Intensity' }} }}
                }},
                plugins: {{
                    title: {{ display: true, text: 'Area Intensity Distribution (RGB)' }}
                }}
        }}
    }});
        """)
    
    # Person average intensity chart
    if viz_data.get('person_average'):
        p1_data = viz_data['person_average']['person_0001']
        p2_data = viz_data['person_average']['person_0002']
        
        if p1_data[0] and p2_data[0]:
            p1_labels = [f"{x:.2f}" for x in p1_data[0]]
            
            js_code.append(f"""
            // Person Average Intensity Chart
            var personAverageCtx = document.getElementById('personAverageChart').getContext('2d');
            window.personAverageChart = new Chart(personAverageCtx, {{
                type: 'bar',
                data: {{
                    labels: {to_js_safe(p1_labels)},
                    datasets: [
                        {{ 
                            label: 'Person 1', 
                            data: {to_js_safe(p1_data[1])}, 
                            backgroundColor: 'rgba(255, 99, 132, 0.6)',
                            borderColor: 'rgba(255, 99, 132, 1)',
                            borderWidth: 1 
                        }},
                        {{ 
                            label: 'Person 2', 
                            data: {to_js_safe(p2_data[1])}, 
                            backgroundColor: 'rgba(54, 162, 235, 0.6)',
                            borderColor: 'rgba(54, 162, 235, 1)',
                            borderWidth: 1 
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        y: {{ beginAtZero: true, title: {{ display: true, text: 'Count' }} }},
                        x: {{ title: {{ display: true, text: 'Average Intensity' }} }}
                    }},
                    plugins: {{
                        title: {{ display: true, text: 'Person Average Intensity Distribution' }}
                    }}
                }}
            }});
            """)
    
    # Person 1 directional chart
    if viz_data.get('person_0001_directional'):
        p1_dirs = viz_data['person_0001_directional']
        view_colors = {
            'front': 'rgba(255, 99, 132, 0.6)',
            'left': 'rgba(54, 162, 235, 0.6)',
            'right': 'rgba(255, 206, 86, 0.6)',
            'top': 'rgba(75, 192, 192, 0.6)',
            'bottom': 'rgba(153, 102, 255, 0.6)'
        }
        
        # Get labels from first available view
        first_view_data = next((data for data in p1_dirs.values() if data[0]), ([], []))
        labels = [f"{x:.2f}" for x in first_view_data[0]] if first_view_data[0] else []
        
        if labels and any(data[0] for data in p1_dirs.values()):
            datasets = []
            for view, data in p1_dirs.items():
                if data[0]:  # Has data
                    color = view_colors.get(view, 'rgba(99, 99, 99, 0.6)')
                    border_color = color.replace('0.6', '1')
                    datasets.append({
                        'label': view.capitalize(),
                        'data': data[1],
                        'backgroundColor': color,
                        'borderColor': border_color,
                        'borderWidth': 1
                    })
            
            if datasets:
                js_code.append(f"""
                // Person 1 Directional Intensity Chart
                var person1DirectionalCtx = document.getElementById('person1DirectionalChart').getContext('2d');
                window.person1DirectionalChart = new Chart(person1DirectionalCtx, {{
                    type: 'bar',
                    data: {{
                        labels: {to_js_safe(labels)},
                        datasets: {to_js_safe(datasets)}
                    }},
                    options: {{
                        responsive: true,
                        scales: {{
                            y: {{ beginAtZero: true, title: {{ display: true, text: 'Count' }} }},
                            x: {{ title: {{ display: true, text: 'Intensity' }} }}
                        }},
                        plugins: {{
                            title: {{ display: true, text: 'Person 1 - Directional Intensity Distribution' }}
                        }}
                    }}
                }});
                """)
    
    # Person 2 directional chart
    if viz_data.get('person_0002_directional'):
        p2_dirs = viz_data['person_0002_directional']
        
        # Get labels from first available view  
        first_view_data = next((data for data in p2_dirs.values() if data[0]), ([], []))
        labels = [f"{x:.2f}" for x in first_view_data[0]] if first_view_data[0] else []
        
        if labels and any(data[0] for data in p2_dirs.values()):
            datasets = []
            for view, data in p2_dirs.items():
                if data[0]:  # Has data
                    color = view_colors.get(view, 'rgba(99, 99, 99, 0.6)')
                    border_color = color.replace('0.6', '1')
                    datasets.append({
                        'label': view.capitalize(),
                        'data': data[1],
                        'backgroundColor': color,
                        'borderColor': border_color,
                        'borderWidth': 1
                    })
            
            if datasets:
                js_code.append(f"""
                // Person 2 Directional Intensity Chart
                var person2DirectionalCtx = document.getElementById('person2DirectionalChart').getContext('2d');
                window.person2DirectionalChart = new Chart(person2DirectionalCtx, {{
                    type: 'bar',
                    data: {{
                        labels: {to_js_safe(labels)},
                        datasets: {to_js_safe(datasets)}
                    }},
                    options: {{
                        responsive: true,
                        scales: {{
                            y: {{ beginAtZero: true, title: {{ display: true, text: 'Count' }} }},
                            x: {{ title: {{ display: true, text: 'Intensity' }} }}
                        }},
                        plugins: {{
                            title: {{ display: true, text: 'Person 2 - Directional Intensity Distribution' }}
                        }}
            }}
        }});
                """)
    
    # Generate 3D visualization data
    # CIELAB color spaces
    for color_type in ['global', 'dc', 'dominant']:
        key = f'{color_type}_color_lab'
        if viz_data.get(key):
            js_code.append(f"""
            // {color_type.capitalize()} Color CIELAB 3D Visualization Data
            window.{color_type}ColorLabData = {to_js_safe(viz_data[key])};
            """)
    
    # RGB 3D color spaces
    for color_type in ['global', 'dc', 'dominant']:
        key = f'{color_type}_color_3d'
        if viz_data.get(key):
            js_code.append(f"""
            // {color_type.capitalize()} Color RGB 3D Visualization Data
            window.{color_type}Color3DData = {to_js_safe(viz_data[key])};
            """)
    
    # Dominant direction 3D
    if viz_data.get('dominant_direction_3d'):
        js_code.append(f"""
        // Dominant Direction 3D Visualization Data
        window.dominantDirection3DData = {to_js_safe(viz_data['dominant_direction_3d'])};
        """)
    
    return '\n'.join(js_code)


def generate_filtering_js(df_json: str, filter_config: Dict, viz_data: Dict) -> str:
    """Generate JavaScript code for interactive filtering."""
    
    # Note: We use special markers for template literals that need to be preserved
    # TEMPLATE_START and TEMPLATE_END will be replaced with ${ and } in the final output
    
    js_template = """
    // Filtering system
    let allData = JSON_DATA_PLACEHOLDER;
    let filteredData = [...allData];
    let intensityRange = { min: 0, max: 1 };
    let personIntensityRanges = {
        average: { min: 0, max: 1 },
        person1Directional: { min: 0, max: 1 },
        person2Directional: { min: 0, max: 1 }
    };
    
    // Initialize on page load
    document.addEventListener('DOMContentLoaded', function() {
        // Calculate intensity ranges
        const intensities = allData.map(row => row.global_intensity);
        intensityRange.min = Math.min(...intensities);
        intensityRange.max = Math.max(...intensities);
        
        // Calculate person intensity ranges
        const person1Data = allData.map(row => row.person_0001_intensity).filter(v => v != null);
        const person2Data = allData.map(row => row.person_0002_intensity).filter(v => v != null);
        const allPersonData = [...person1Data, ...person2Data];
        
        if (allPersonData.length > 0) {
            personIntensityRanges.average.min = Math.min(...allPersonData);
            personIntensityRanges.average.max = Math.max(...allPersonData);
        }
        
        // Person directional ranges
        const person1Views = ['front', 'left', 'right', 'top', 'bottom'];
        const person1DirectionalData = [];
        const person2DirectionalData = [];
        
        person1Views.forEach(view => {
            const data = allData.map(row => row['person_0001_' + view + '_intensity']).filter(v => v != null);
            person1DirectionalData.push(...data);
        });
        
        person1Views.forEach(view => {
            const data = allData.map(row => row['person_0002_' + view + '_intensity']).filter(v => v != null);
            person2DirectionalData.push(...data);
        });
        
        if (person1DirectionalData.length > 0) {
            personIntensityRanges.person1Directional.min = Math.min(...person1DirectionalData);
            personIntensityRanges.person1Directional.max = Math.max(...person1DirectionalData);
        }
        
        if (person2DirectionalData.length > 0) {
            personIntensityRanges.person2Directional.min = Math.min(...person2DirectionalData);
            personIntensityRanges.person2Directional.max = Math.max(...person2DirectionalData);
        }
        
        // Set up brush selection after a small delay to ensure charts are rendered
        setTimeout(() => {
        setupBrushSelection();
            // Setup direction filtering after 3D plots are initialized
            setupDirectionFiltering();
        }, 300);
    });
    
    // Brush selection setup
    function setupBrushSelection() {
        const canvas = document.getElementById('globalIntensityChart');
        if (!canvas || !canvas.parentElement) return;
        
        // Ensure the chart is ready and has a chart area
        const chart = window.globalIntensityChart;
        if (!chart || !chart.chartArea) {
            console.warn('Chart not ready for brush selection, retrying...');
            setTimeout(setupBrushSelection, 100);
            return;
        }
        
        const chartContainer = canvas.parentElement;
        chartContainer.style.position = 'relative';
        
        // Use the chart container directly instead of creating an overlay
        const overlay = chartContainer; // Simplify by using container directly
        
        let isDragging = false;
        let startX, endX;
        let currentSelection = null;
        
        // Add mousemove handler to change cursor when over chart area
        canvas.addEventListener('mousemove', function(e) {
            if (isDragging) return; // Don't change cursor while dragging
            
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;  // Keep this relative to canvas for cursor check
            const y = e.clientY - rect.top;
            
            const chart = window.globalIntensityChart;
            if (chart && chart.chartArea) {
                const chartArea = chart.chartArea;
                
                // Change cursor based on position
                if (x >= chartArea.left && x <= chartArea.right &&
                    y >= chartArea.top && y <= chartArea.bottom) {
                    canvas.style.cursor = 'crosshair';
                } else {
                    canvas.style.cursor = 'default';
                }
            }
        });
        
        // Add mouse event listeners to the canvas
        canvas.addEventListener('mousedown', function(e) {
            const rect = canvas.getBoundingClientRect();
            const containerRect = chartContainer.getBoundingClientRect();
            
            // Calculate click position relative to the container (not canvas)
            const clickX = e.clientX - containerRect.left;
            const clickY = e.clientY - containerRect.top;
            
            // Get chart area bounds
            const chart = window.globalIntensityChart;
            if (!chart || !chart.chartArea) return;
            
            const chartArea = chart.chartArea;
            
            // Canvas offset within container
            const canvasOffsetX = rect.left - containerRect.left;
            const canvasOffsetY = rect.top - containerRect.top;
            
            // Check if click is inside the chart area horizontally (allow any Y position)
            const canvasRelativeX = clickX - canvasOffsetX;
            
            if (canvasRelativeX >= chartArea.left && canvasRelativeX <= chartArea.right) {
                
                isDragging = true;
                startX = clickX; // Use the actual click position, no constraining needed
                
                
                // Remove any existing selection
                if (currentSelection) {
                    currentSelection.remove();
                    currentSelection = null;
                }
                
                // Create initial selection at click position
                currentSelection = document.createElement('div');
                currentSelection.className = 'brush-selection';
                currentSelection.style.position = 'absolute';
                currentSelection.style.backgroundColor = 'rgba(255, 0, 0, 0.5)'; // Red line initially
                currentSelection.style.pointerEvents = 'none';
                // Position selection to match chart area exactly (accounting for canvas offset)
                currentSelection.style.top = (chartArea.top + canvasOffsetY) + 'px';
                currentSelection.style.height = (chartArea.bottom - chartArea.top) + 'px';
                currentSelection.style.left = startX + 'px';
                currentSelection.style.width = '1px';
                overlay.appendChild(currentSelection);
                
            }
        });
        
        document.addEventListener('mousemove', function(e) {
            if (!isDragging) return;
            
            const containerRect = chartContainer.getBoundingClientRect();
            endX = e.clientX - containerRect.left;  // Relative to container like startX
            
            // Get chart area bounds to constrain selection
            const chart = window.globalIntensityChart;
            if (chart && chart.chartArea) {
            const rect = canvas.getBoundingClientRect();
                const canvasOffsetX = rect.left - containerRect.left;
                const chartArea = chart.chartArea;
                
                // Constrain to chart area bounds (adjusted for container coordinates)
                const minX = chartArea.left + canvasOffsetX;
                const maxX = chartArea.right + canvasOffsetX;
                endX = Math.max(minX, Math.min(maxX, endX));
            }
            
            // Update selection position (it was already created on mousedown)
            if (currentSelection) {
                const leftPos = Math.min(startX, endX);
                const width = Math.abs(endX - startX);
                
                currentSelection.style.left = leftPos + 'px';
                currentSelection.style.width = width + 'px';
                
                // Change color and add shadow once dragging starts
                if (width > 2) {
                    currentSelection.style.backgroundColor = 'rgba(0, 123, 255, 0.3)';
                    currentSelection.style.boxShadow = 'inset 0 0 0 1px rgba(0, 123, 255, 0.8)';
                }
            }
        });
        
        document.addEventListener('mouseup', function(e) {
            if (!isDragging) return;
            isDragging = false;
            
            const containerRect = chartContainer.getBoundingClientRect();
            endX = e.clientX - containerRect.left;  // Relative to container
            
            if (Math.abs(endX - startX) > 5) {
                const leftX = Math.min(startX, endX);
                const rightX = Math.max(startX, endX);
                
                // Get the chart instance and its chart area
                const chart = window.globalIntensityChart;
                if (!chart) return;
                
                const rect = canvas.getBoundingClientRect();
                const canvasOffsetX = rect.left - containerRect.left;
                
                const chartArea = chart.chartArea;
                const chartWidth = chartArea.right - chartArea.left;
                const chartLeft = chartArea.left + canvasOffsetX;  // Adjust for canvas offset
                
                // Convert pixel coordinates to chart area coordinates
                const leftChartX = Math.max(0, leftX - chartLeft);
                const rightChartX = Math.min(chartWidth, rightX - chartLeft);
                
                // Convert to ratios within the chart area
                const leftRatio = leftChartX / chartWidth;
                const rightRatio = rightChartX / chartWidth;
                
                // Clamp ratios to valid range
                const clampedLeftRatio = Math.max(0, Math.min(1, leftRatio));
                const clampedRightRatio = Math.max(0, Math.min(1, rightRatio));
                
                const minIntensity = intensityRange.min + clampedLeftRatio * (intensityRange.max - intensityRange.min);
                const maxIntensity = intensityRange.min + clampedRightRatio * (intensityRange.max - intensityRange.min);
                
                
                applyFilter(minIntensity, maxIntensity);
            } else {
                // Remove selection if drag was too small
                if (currentSelection) {
                    currentSelection.remove();
                    currentSelection = null;
                }
            }
        });
    }
    
    // Apply filter function
    function applyFilter(minIntensity, maxIntensity) {
        currentIntensityRange = { min: minIntensity, max: maxIntensity };
        
        // Update range display
        document.getElementById('selection-range').textContent = minIntensity.toFixed(3) + ' - ' + maxIntensity.toFixed(3);
        
        // Apply combined filters
        applyCombinedFilters();
    }
    
    // Clear filter function
    function clearFilter() {
        currentIntensityRange = null;
        
        // Remove visual selection
        const selections = document.querySelectorAll('.brush-selection');
            selections.forEach(sel => sel.remove());
        
        document.getElementById('selection-range').textContent = 'Full Range';
        
        // Apply combined filters
        applyCombinedFilters();
    }
    
    // Update all charts with filtered data
    function updateAllChartsWithFilter() {
        updatePersonAverageChart();
        updatePersonDirectionalChart('person1DirectionalChart', 'person_0001');
        updatePersonDirectionalChart('person2DirectionalChart', 'person_0002');
        update3DVisualizations();
    }
    
    // Update person average chart
    function updatePersonAverageChart() {
        const chart = window.personAverageChart;
        if (!chart) return;
        
        const bins = 20;
        const min = personIntensityRanges.average.min;
        const max = personIntensityRanges.average.max;
        const binWidth = (max - min) / bins;
        
        // Create consistent labels
        const labels = [];
        for (let i = 0; i < bins; i++) {
            labels.push(((min + (i + 0.5) * binWidth)).toFixed(2));
        }
        
        // Get filtered person data
        const person1Data = filteredData.map(row => row.person_0001_intensity).filter(v => v != null);
        const person2Data = filteredData.map(row => row.person_0002_intensity).filter(v => v != null);
        
        // Create histograms
        const person1Histogram = new Array(bins).fill(0);
        const person2Histogram = new Array(bins).fill(0);
        
        person1Data.forEach(value => {
            const binIndex = Math.min(Math.floor((value - min) / binWidth), bins - 1);
            if (binIndex >= 0) person1Histogram[binIndex]++;
        });
        
        person2Data.forEach(value => {
            const binIndex = Math.min(Math.floor((value - min) / binWidth), bins - 1);
            if (binIndex >= 0) person2Histogram[binIndex]++;
        });
        
        // Update chart
        chart.data.labels = labels;
        chart.data.datasets[0].data = person1Histogram;
        chart.data.datasets[1].data = person2Histogram;
        chart.update();
    }
    
    // Update person directional chart
    function updatePersonDirectionalChart(chartId, personKey) {
        const chart = window[chartId.replace('Chart', 'Chart')];
        if (!chart) return;
        
        const bins = 20;
        const rangeKey = personKey === 'person_0001' ? 'person1Directional' : 'person2Directional';
        const min = personIntensityRanges[rangeKey].min;
        const max = personIntensityRanges[rangeKey].max;
        const binWidth = (max - min) / bins;
        
        // Create consistent labels
        const labels = [];
        for (let i = 0; i < bins; i++) {
            labels.push(((min + (i + 0.5) * binWidth)).toFixed(2));
        }
        
        const views = ['front', 'left', 'right', 'top', 'bottom'];
        
        // Update each view dataset
        views.forEach(view => {
            const columnName = personKey + '_' + view + '_intensity';
            const viewData = filteredData.map(row => row[columnName]).filter(v => v != null && !isNaN(v));
            
            const histogram = new Array(bins).fill(0);
            viewData.forEach(value => {
                const binIndex = Math.min(Math.floor((value - min) / binWidth), bins - 1);
                if (binIndex >= 0) histogram[binIndex]++;
            });
            
            // Find dataset by label
            const datasetIdx = chart.data.datasets.findIndex(ds => 
                ds.label && ds.label.toLowerCase() === view.toLowerCase()
            );
            
            if (datasetIdx >= 0) {
                chart.data.datasets[datasetIdx].data = histogram;
            }
        });
        
        chart.data.labels = labels;
        chart.update();
    }
    
    // Filter states
    let currentIntensityRange = null;
    let directionFilterActive = false;
    let directionFilterCenter = null;
    let directionFilterRadius = 0.5;
    let directionFilteredData = [];
    
    // Setup direction filtering for dominant light direction plot
    function setupDirectionFiltering() {
        if (!window.dominantDirection3DData) return;
        
        // Find the 3D plots section
        const plotsSection = document.querySelector('.intensity-section:has(#dominantDirection3D)');
        if (!plotsSection) return;
        
        // Create a wrapper for the plots grid and controls
        const wrapper = document.createElement('div');
        wrapper.style.cssText = 'display: flex; gap: 20px; align-items: start;';
        
        // Move the existing grid into the wrapper
        const gridContainer = plotsSection.querySelector('div[style*="grid"]');
        wrapper.appendChild(gridContainer);
        
        // Create control panel on the right matching intensity filter style
        const controlPanel = document.createElement('div');
        controlPanel.id = 'direction-filter-controls';
        controlPanel.style.cssText = `
            background: #f8f8f8;
            padding: 15px;
            border: 1px inset #c0c0c0;
            width: 280px;
            flex-shrink: 0;
            font-family: 'Courier New', monospace;
        `;
        
        controlPanel.innerHTML = `
            <div style="background: #e0e0e0; color: #000; padding: 5px; margin: -15px -15px 15px -15px; border: 1px outset #c0c0c0; font-weight: bold;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 14px;">🎯 Direction Filter</span>
                    <button id="clear-direction-filter" style="padding: 2px 10px; background: #f0f0f0; color: #000; border: 1px outset #c0c0c0; font-family: 'Courier New', monospace; font-size: 11px; cursor: pointer;">Clear</button>
                </div>
            </div>
            <div style="margin-bottom: 15px;">
                <label style="display: block; margin-bottom: 5px; color: #000; font-weight: bold; font-size: 12px;">
                    Azimuth (θ): <span id="azimuth-value" style="font-weight: normal; color: #0066cc;">0°</span>
                </label>
                <input type="range" id="filter-azimuth" min="-180" max="180" step="5" value="0" style="width: 100%; height: 20px; cursor: pointer;">
            </div>
            <div style="margin-bottom: 15px;">
                <label style="display: block; margin-bottom: 5px; color: #000; font-weight: bold; font-size: 12px;">
                    Elevation (φ): <span id="elevation-value" style="font-weight: normal; color: #0066cc;">0°</span>
                </label>
                <input type="range" id="filter-elevation" min="-90" max="90" step="5" value="0" style="width: 100%; height: 20px; cursor: pointer;">
            </div>
            <div style="margin-bottom: 15px;">
                <label style="display: block; margin-bottom: 5px; color: #000; font-weight: bold; font-size: 12px;">
                    Width: <span id="cone-width-value" style="font-weight: normal; color: #0066cc;">30°</span>
                </label>
                <input type="range" id="filter-cone-width" min="5" max="90" step="5" value="30" style="width: 100%; height: 20px; cursor: pointer;">
            </div>
            <div style="margin-bottom: 15px;">
                <label style="display: block; margin-bottom: 5px; color: #000; font-weight: bold; font-size: 12px;">
                    Height: <span id="cone-height-value" style="font-weight: normal; color: #0066cc;">30°</span>
                </label>
                <input type="range" id="filter-cone-height" min="5" max="90" step="5" value="30" style="width: 100%; height: 20px; cursor: pointer;">
            </div>
            <div id="direction-filter-status" style="margin-top: 15px; padding: 8px; background: #fff; border: 1px inset #c0c0c0; font-size: 12px; color: #666; text-align: center; border-radius: 2px;">
                Move sliders to select a direction cone
            </div>
        `;
        
        // Add control panel to wrapper
        wrapper.appendChild(controlPanel);
        
        // Insert wrapper after the h2
        const h2 = plotsSection.querySelector('h2');
        h2.parentNode.insertBefore(wrapper, h2.nextSibling);
        
        // Spherical filter parameters
        let azimuth = 0;    // -180 to 180 degrees
        let elevation = 0;  // -90 to 90 degrees  
        let coneWidth = 30; // cone width in degrees
        let coneHeight = 30; // cone height in degrees
        
        // Handle azimuth slider
        const azimuthSlider = document.getElementById('filter-azimuth');
        const azimuthValue = document.getElementById('azimuth-value');
        azimuthSlider.addEventListener('input', (e) => {
            azimuth = parseFloat(e.target.value);
            azimuthValue.textContent = azimuth + '°';
            updateDirectionFilter();
        });
        
        // Handle elevation slider
        const elevationSlider = document.getElementById('filter-elevation');
        const elevationValue = document.getElementById('elevation-value');
        elevationSlider.addEventListener('input', (e) => {
            elevation = parseFloat(e.target.value);
            elevationValue.textContent = elevation + '°';
            updateDirectionFilter();
        });
        
        // Handle cone width slider
        const coneWidthSlider = document.getElementById('filter-cone-width');
        const coneWidthValue = document.getElementById('cone-width-value');
        coneWidthSlider.addEventListener('input', (e) => {
            coneWidth = parseFloat(e.target.value);
            coneWidthValue.textContent = coneWidth + '°';
            updateDirectionFilter();
        });
        
        // Handle cone height slider
        const coneHeightSlider = document.getElementById('filter-cone-height');
        const coneHeightValue = document.getElementById('cone-height-value');
        coneHeightSlider.addEventListener('input', (e) => {
            coneHeight = parseFloat(e.target.value);
            coneHeightValue.textContent = coneHeight + '°';
            updateDirectionFilter();
        });
        
        // Handle clear button
        const clearBtn = document.getElementById('clear-direction-filter');
        clearBtn.addEventListener('click', clearDirectionFilter);
        
        // Button hover effect
        clearBtn.addEventListener('mousedown', function() {
            this.style.border = '1px inset #c0c0c0';
            this.style.background = '#e0e0e0';
        });
        clearBtn.addEventListener('mouseup', function() {
            this.style.border = '1px outset #c0c0c0';
            this.style.background = '#f0f0f0';
        });
        clearBtn.addEventListener('mouseleave', function() {
            this.style.border = '1px outset #c0c0c0';
            this.style.background = '#f0f0f0';
        });
        clearBtn.addEventListener('mouseenter', function() {
            this.style.background = '#f8f8f8';
        });
        
        // Store filter parameters globally
        window.directionFilterParams = {
            getAzimuth: () => azimuth,
            getElevation: () => elevation,
            getConeWidth: () => coneWidth,
            getConeHeight: () => coneHeight
        };
    }
    
    function updateDirectionFilter() {
        if (!window.directionFilterParams) return;
        
        const azimuth = window.directionFilterParams.getAzimuth();
        const elevation = window.directionFilterParams.getElevation();
        const coneWidth = window.directionFilterParams.getConeWidth();
        const coneHeight = window.directionFilterParams.getConeHeight();
        
        // Convert spherical to Cartesian for the cone center direction
        const azimuthRad = azimuth * Math.PI / 180;
        const elevationRad = elevation * Math.PI / 180;
        
        // Spherical to Cartesian conversion
        const centerX = Math.cos(elevationRad) * Math.cos(azimuthRad);
        const centerY = Math.cos(elevationRad) * Math.sin(azimuthRad);
        const centerZ = Math.sin(elevationRad);
        
        // Calculate perpendicular vectors for elliptical cone
        let perpX, perpY, perpZ;
        if (Math.abs(centerZ) < 0.9) {
            perpX = -centerY;
            perpY = centerX;
            perpZ = 0;
        } else {
            perpX = 0;
            perpY = -centerZ;
            perpZ = centerY;
        }
        
        // Normalize
        const perpMag = Math.sqrt(perpX*perpX + perpY*perpY + perpZ*perpZ);
        if (perpMag > 0) {
            perpX /= perpMag;
            perpY /= perpMag;
            perpZ /= perpMag;
        }
        
        // Second perpendicular via cross product
        const perp2X = centerY * perpZ - centerZ * perpY;
        const perp2Y = centerZ * perpX - centerX * perpZ;
        const perp2Z = centerX * perpY - centerY * perpX;
        
        // Filter data based on elliptical cone
        directionFilteredData = allData.filter(row => {
            const dirData = window.dominantDirection3DData.find(d => d.name === row.hdri_name);
            if (!dirData) return false;
            
            // Project vector onto cone coordinate system
            const alongAxis = dirData.x * centerX + dirData.y * centerY + dirData.z * centerZ;
            const alongWidth = dirData.x * perpX + dirData.y * perpY + dirData.z * perpZ;
            const alongHeight = dirData.x * perp2X + dirData.y * perp2Y + dirData.z * perp2Z;
            
            // Check if within elliptical cone
            if (alongAxis <= 0) return false; // Behind cone
            
            const widthAngle = Math.atan2(Math.abs(alongWidth), alongAxis) * 180 / Math.PI;
            const heightAngle = Math.atan2(Math.abs(alongHeight), alongAxis) * 180 / Math.PI;
            
            // Elliptical check
            const widthRatio = widthAngle / coneWidth;
            const heightRatio = heightAngle / coneHeight;
            
            return (widthRatio * widthRatio + heightRatio * heightRatio) <= 1;
        });
        
        // Update status
        const status = document.getElementById('direction-filter-status');
        if (directionFilteredData.length > 0) {
            directionFilterActive = true;
            status.textContent = `Selected: ${directionFilteredData.length} HDRIs within elliptical cone`;
            status.style.color = '#28a745';
        } else {
            status.textContent = 'No HDRIs in selected cone';
            status.style.color = '#dc3545';
        }
        
        // Update visualization to show selection cone
        addSelectionEllipticalCone(centerX, centerY, centerZ, coneWidth, coneHeight, perpX, perpY, perpZ, perp2X, perp2Y, perp2Z);
        
        // Apply combined filters
        applyCombinedFilters();
    }
    
    function addSelectionEllipticalCone(centerX, centerY, centerZ, coneWidth, coneHeight, perpX, perpY, perpZ, perp2X, perp2Y, perp2Z) {
        // Generate elliptical cone visualization
        const ellipseLines = generateEllipticalConeLines(centerX, centerY, centerZ, coneWidth, coneHeight, perpX, perpY, perpZ, perp2X, perp2Y, perp2Z);
        
        // Create line trace for cone visualization
        const coneTrace = {
            type: 'scatter3d',
            mode: 'lines',
            x: ellipseLines.x,
            y: ellipseLines.y,
            z: ellipseLines.z,
            line: {
                color: 'rgb(100, 150, 255)',
                width: 3
            },
            opacity: 0.6,
            hoverinfo: 'skip',
            showlegend: false
        };
        
        // Remove old cone traces if any
        const plotDiv = document.getElementById('dominantDirection3D');
        if (plotDiv.data && plotDiv.data.length > 1) {
            while (plotDiv.data.length > 1) {
                Plotly.deleteTraces('dominantDirection3D', [1]);
            }
        }
        
        // Add new cone trace
        Plotly.addTraces('dominantDirection3D', [coneTrace]);
    }
    
    function generateEllipticalConeLines(cx, cy, cz, widthAngle, heightAngle, perpX, perpY, perpZ, perp2X, perp2Y, perp2Z) {
        const x = [], y = [], z = [];
        const widthRad = widthAngle * Math.PI / 180;
        const heightRad = heightAngle * Math.PI / 180;
        
        // Create elliptical base on unit sphere
        const numPoints = 32;
        
        for (let i = 0; i <= numPoints; i++) {
            const t = (i / numPoints) * 2 * Math.PI;
            const cosT = Math.cos(t);
            const sinT = Math.sin(t);
            
            // Elliptical radius at this angle
            const ellipseAngle = Math.atan2(
                Math.abs(sinT) * heightRad,
                Math.abs(cosT) * widthRad
            );
            
            // Point on elliptical cone base
            const baseX = cx * Math.cos(ellipseAngle) + 
                         (perpX * cosT * Math.tan(widthRad) + perp2X * sinT * Math.tan(heightRad)) * Math.sin(ellipseAngle);
            const baseY = cy * Math.cos(ellipseAngle) + 
                         (perpY * cosT * Math.tan(widthRad) + perp2Y * sinT * Math.tan(heightRad)) * Math.sin(ellipseAngle);
            const baseZ = cz * Math.cos(ellipseAngle) + 
                         (perpZ * cosT * Math.tan(widthRad) + perp2Z * sinT * Math.tan(heightRad)) * Math.sin(ellipseAngle);
            
            // Normalize to unit sphere
            const mag = Math.sqrt(baseX*baseX + baseY*baseY + baseZ*baseZ);
            if (mag > 0) {
                x.push(baseX/mag);
                y.push(baseY/mag);
                z.push(baseZ/mag);
            }
        }
        
        // Add lines from origin to ellipse at cardinal points
        const cardinalPoints = [0, Math.floor(numPoints/4), Math.floor(numPoints/2), Math.floor(3*numPoints/4)];
        for (const idx of cardinalPoints) {
            x.push(null); y.push(null); z.push(null); // Break line
            x.push(0); y.push(0); z.push(0); // Origin
            x.push(x[idx]); y.push(y[idx]); z.push(z[idx]); // Point on ellipse
        }
        
        return { x, y, z };
    }
    
    function addSelectionCone(centerX, centerY, centerZ, coneAngle) {
        // Generate cone visualization lines (lightweight approach)
        const coneLines = generateConeLines(centerX, centerY, centerZ, coneAngle);
        
        // Create line trace for cone visualization
        const coneTrace = {
            type: 'scatter3d',
            mode: 'lines',
            x: coneLines.x,
            y: coneLines.y,
            z: coneLines.z,
            line: {
                color: 'rgb(100, 150, 255)',
                width: 3
            },
            opacity: 0.6,
            hoverinfo: 'skip',
            showlegend: false
        };
        
        // Remove old cone traces if any
        const plotDiv = document.getElementById('dominantDirection3D');
        if (plotDiv.data && plotDiv.data.length > 1) {
            // Keep only the main data trace
            while (plotDiv.data.length > 1) {
                Plotly.deleteTraces('dominantDirection3D', [1]);
            }
        }
        
        // Add new cone trace
        Plotly.addTraces('dominantDirection3D', [coneTrace]);
    }
    
    function generateConeLines(cx, cy, cz, coneAngle) {
        const x = [], y = [], z = [];
        const coneAngleRad = coneAngle * Math.PI / 180;
        
        // Create circular base of cone on unit sphere
        const numPoints = 24;
        
        // Find two perpendicular vectors to the cone axis
        let perpX, perpY, perpZ;
        if (Math.abs(cz) < 0.9) {
            // Use z-axis as reference
            perpX = -cy;
            perpY = cx;
            perpZ = 0;
        } else {
            // Use x-axis as reference
            perpX = 0;
            perpY = -cz;
            perpZ = cy;
        }
        
        // Normalize perpendicular vector
        const perpMag = Math.sqrt(perpX*perpX + perpY*perpY + perpZ*perpZ);
        if (perpMag > 0) {
            perpX /= perpMag;
            perpY /= perpMag;
            perpZ /= perpMag;
        }
        
        // Get second perpendicular vector via cross product
        const perp2X = cy * perpZ - cz * perpY;
        const perp2Y = cz * perpX - cx * perpZ;
        const perp2Z = cx * perpY - cy * perpX;
        
        // Draw cone outline
        for (let i = 0; i <= numPoints; i++) {
            const angle = (i / numPoints) * 2 * Math.PI;
            const cosAngle = Math.cos(angle);
            const sinAngle = Math.sin(angle);
            
            // Point on cone base circle
            const baseX = cx * Math.cos(coneAngleRad) + 
                         (perpX * cosAngle + perp2X * sinAngle) * Math.sin(coneAngleRad);
            const baseY = cy * Math.cos(coneAngleRad) + 
                         (perpY * cosAngle + perp2Y * sinAngle) * Math.sin(coneAngleRad);
            const baseZ = cz * Math.cos(coneAngleRad) + 
                         (perpZ * cosAngle + perp2Z * sinAngle) * Math.sin(coneAngleRad);
            
            // Normalize to unit sphere
            const mag = Math.sqrt(baseX*baseX + baseY*baseY + baseZ*baseZ);
            x.push(baseX/mag);
            y.push(baseY/mag);
            z.push(baseZ/mag);
        }
        
        // Add lines from center to a few points on the circle
        for (let i = 0; i < 4; i++) {
            const idx = Math.floor(i * numPoints / 4);
            x.push(null); y.push(null); z.push(null); // Break line
            x.push(0); y.push(0); z.push(0); // Origin
            x.push(x[idx]); y.push(y[idx]); z.push(z[idx]); // Point on circle
        }
        
        return { x, y, z };
    }
    
    function clearDirectionFilter() {
        directionFilterActive = false;
        directionFilteredData = [];
        
        // Reset sliders
        document.getElementById('filter-azimuth').value = 0;
        document.getElementById('filter-elevation').value = 0;
        document.getElementById('filter-cone-width').value = 30;
        document.getElementById('filter-cone-height').value = 30;
        document.getElementById('azimuth-value').textContent = '0°';
        document.getElementById('elevation-value').textContent = '0°';
        document.getElementById('cone-width-value').textContent = '30°';
        document.getElementById('cone-height-value').textContent = '30°';
        
        // Update status
        document.getElementById('direction-filter-status').textContent = 'Move sliders to select a direction cone';
        document.getElementById('direction-filter-status').style.color = '#6c757d';
        
        // Remove cone trace
        const plotDiv = document.getElementById('dominantDirection3D');
        if (plotDiv.data && plotDiv.data.length > 1) {
            while (plotDiv.data.length > 1) {
                Plotly.deleteTraces('dominantDirection3D', [1]);
            }
        }
        
        applyCombinedFilters();
    }
    
    function applyCombinedFilters() {
        // Combine intensity and direction filters with AND logic
        let combinedFilteredData = allData;
        
        // Apply intensity filter if active
        if (currentIntensityRange) {
            combinedFilteredData = combinedFilteredData.filter(row => 
                row.global_intensity >= currentIntensityRange.min && 
                row.global_intensity <= currentIntensityRange.max
            );
        }
        
        // Apply direction filter if active
        if (directionFilterActive && directionFilteredData.length > 0) {
            const directionNames = directionFilteredData.map(d => d.hdri_name);
            combinedFilteredData = combinedFilteredData.filter(row => 
                directionNames.includes(row.hdri_name)
            );
        }
        
        // Update global filtered data
        filteredData = combinedFilteredData;
        
        // Update all visualizations
        updateAllChartsWithFilter();
        update3DVisualizations();
        
        // Update individual reports list
        const currentSearchTerm = document.getElementById('hdri-search')?.value?.toLowerCase() || '';
        filterHDRILinks(currentSearchTerm);
        
        // Update status
        const statusElement = document.getElementById('chart-status');
        if (statusElement) {
            if (filteredData.length < allData.length) {
                statusElement.textContent = 'Filtered: ' + filteredData.length + ' HDRIs';
                statusElement.style.color = '#4CAF50';
            } else {
                statusElement.textContent = 'Showing all ' + allData.length + ' HDRIs';
                statusElement.style.color = '#666';
            }
        }
        
        // Update filter indicator
        const hasFilters = currentIntensityRange || directionFilterActive;
        document.getElementById('intensity-filter-indicator').style.display = hasFilters ? 'inline' : 'none';
        if (hasFilters) {
            const filterTexts = [];
            if (currentIntensityRange) filterTexts.push('Intensity');
            if (directionFilterActive) filterTexts.push('Direction');
            document.getElementById('intensity-filter-indicator').textContent = `Filters: ${filterTexts.join(' & ')}`;
        }
    }
    
    
    // Update 3D visualizations
    function update3DVisualizations() {
        const filteredNames = filteredData.map(row => row.hdri_name);
        const isFiltered = filteredData.length < allData.length;
        
        if (isFiltered) {
            // Update all plot managers with filtered data
            if (window.globalColor3DManager) {
                window.globalColor3DManager.updateFiltering(filteredNames);
            }
            if (window.dcColor3DManager) {
                window.dcColor3DManager.updateFiltering(filteredNames);
            }
            if (window.dominantColor3DManager) {
                window.dominantColor3DManager.updateFiltering(filteredNames);
            }
            if (window.dominantDirection3DManager) {
                window.dominantDirection3DManager.updateFiltering(filteredNames);
            }
        } else {
            // Reset all plots to original state
            if (window.globalColor3DManager) {
                window.globalColor3DManager.resetFiltering();
            }
            if (window.dcColor3DManager) {
                window.dcColor3DManager.resetFiltering();
            }
            if (window.dominantColor3DManager) {
                window.dominantColor3DManager.resetFiltering();
            }
            if (window.dominantDirection3DManager) {
                window.dominantDirection3DManager.resetFiltering();
            }
        }
    }
    
    // Search functionality
    function setupSearch() {
        const searchInput = document.getElementById('hdri-search');
        if (!searchInput) return;
        
        searchInput.addEventListener('input', function(e) {
            const searchTerm = e.target.value.toLowerCase();
            filterHDRILinks(searchTerm);
        });
    }
    
    function filterHDRILinks(searchTerm = '') {
        const links = document.querySelectorAll('.individual-link');
        let visibleCount = 0;
        let totalCount = links.length;
        
        // Get currently filtered HDRI names from intensity filtering
        const filteredHdriNames = filteredData.map(row => row.hdri_name);
        const isIntensityFiltered = filteredData.length < allData.length;
        
        links.forEach(link => {
            const hdriName = link.dataset.hdriName;
            const hdriNameLower = hdriName.toLowerCase();
            
            // Check both search term and intensity filtering
            const matchesSearch = !searchTerm || hdriNameLower.includes(searchTerm);
            const matchesIntensity = !isIntensityFiltered || filteredHdriNames.includes(hdriName);
            
            if (matchesSearch && matchesIntensity) {
                link.style.display = '';
                // Add class for intensity filtering visual feedback
                if (isIntensityFiltered) {
                    link.classList.add('intensity-filtered');
                } else {
                    link.classList.remove('intensity-filtered');
                }
                visibleCount++;
            } else {
                link.style.display = 'none';
                link.classList.remove('intensity-filtered');
            }
        });
        
        // Update counter with more detailed info
        const counter = document.getElementById('hdri-counter');
        const intensityIndicator = document.getElementById('intensity-filter-indicator');
        
        if (counter) {
            if (visibleCount === totalCount) {
                counter.textContent = `${totalCount} files`;
            } else {
                let counterText = `${visibleCount} of ${totalCount} files`;
                if (isIntensityFiltered && searchTerm) {
                    counterText += ` (intensity + search)`;
                } else if (isIntensityFiltered) {
                    counterText += ` (intensity filtered)`;
                } else if (searchTerm) {
                    counterText += ` (search filtered)`;
                }
                counter.textContent = counterText;
            }
        }
        
        // Show/hide intensity filter indicator
        if (intensityIndicator) {
            if (isIntensityFiltered) {
                intensityIndicator.style.display = 'block';
            } else {
                intensityIndicator.style.display = 'none';
            }
        }
    }
    
    // Initialize search on page load
    document.addEventListener('DOMContentLoaded', function() {
        setupSearch();
    });
    
    // Expose clear function globally
    window.clearBrushSelection = clearFilter;
    """
    
    # Replace placeholder with actual data
    js_template = js_template.replace('JSON_DATA_PLACEHOLDER', df_json)
    
    return js_template


# ==============================
# Statistics Tables Generation
# ==============================

def generate_stats_tables(stats: Dict) -> str:
    """Generate HTML for statistics tables."""
    tables = []
    
    # Global intensity stats
    if 'global_intensity' in stats:
        s = stats['global_intensity']
        tables.append(f"""
        <div class="stats-table">
            <h4>Global Intensity Statistics</h4>
            <table>
                <tr><td>Mean:</td><td>{s['mean']:.4f}</td></tr>
                <tr><td>Std Dev:</td><td>{s['std']:.4f}</td></tr>
                <tr><td>Min:</td><td>{s['min']:.4f}</td></tr>
                <tr><td>Max:</td><td>{s['max']:.4f}</td></tr>
                <tr><td>Median:</td><td>{s['median']:.4f}</td></tr>
            </table>
        </div>
        """)
    
    # Person render statistics
    if 'person_0001_intensity' in stats and 'person_0002_intensity' in stats:
        p1 = stats['person_0001_intensity']
        p2 = stats['person_0002_intensity']
        tables.append(f"""
        <div class="stats-table">
            <h4>Person Render Average Intensity Statistics</h4>
            <table>
                <tr><th></th><th>Person 1</th><th>Person 2</th></tr>
                <tr><td>Mean:</td><td>{p1['mean']:.3f}</td><td>{p2['mean']:.3f}</td></tr>
                <tr><td>Std Dev:</td><td>{p1['std']:.3f}</td><td>{p2['std']:.3f}</td></tr>
                <tr><td>Min:</td><td>{p1['min']:.3f}</td><td>{p2['min']:.3f}</td></tr>
                <tr><td>Max:</td><td>{p1['max']:.3f}</td><td>{p2['max']:.3f}</td></tr>
                <tr><td>Median:</td><td>{p1['median']:.3f}</td><td>{p2['median']:.3f}</td></tr>
            </table>
        </div>
        """)
    
    # Person directional statistics
    directional_views = ['front', 'left', 'right', 'top', 'bottom']
    
    # Person 1 directional
    person1_directional = {}
    for view in directional_views:
        key = f'person_0001_{view}_intensity'
        if key in stats:
            person1_directional[view] = stats[key]
    
    if person1_directional:
        tables.append(f"""
        <div class="stats-table">
            <h4>Person 1 Directional Intensity Statistics</h4>
            <table>
                <tr><th></th>{''.join(f'<th>{view.capitalize()}</th>' for view in directional_views if view in person1_directional)}</tr>
                <tr><td>Mean:</td>{''.join(f"<td>{person1_directional[view]['mean']:.3f}</td>" for view in directional_views if view in person1_directional)}</tr>
                <tr><td>Std Dev:</td>{''.join(f"<td>{person1_directional[view]['std']:.3f}</td>" for view in directional_views if view in person1_directional)}</tr>
                <tr><td>Min:</td>{''.join(f"<td>{person1_directional[view]['min']:.3f}</td>" for view in directional_views if view in person1_directional)}</tr>
                <tr><td>Max:</td>{''.join(f"<td>{person1_directional[view]['max']:.3f}</td>" for view in directional_views if view in person1_directional)}</tr>
                <tr><td>Median:</td>{''.join(f"<td>{person1_directional[view]['median']:.3f}</td>" for view in directional_views if view in person1_directional)}</tr>
            </table>
        </div>
        """)
    
    # Person 2 directional  
    person2_directional = {}
    for view in directional_views:
        key = f'person_0002_{view}_intensity'
        if key in stats:
            person2_directional[view] = stats[key]
    
    if person2_directional:
        tables.append(f"""
        <div class="stats-table">
            <h4>Person 2 Directional Intensity Statistics</h4>
            <table>
                <tr><th></th>{''.join(f'<th>{view.capitalize()}</th>' for view in directional_views if view in person2_directional)}</tr>
                <tr><td>Mean:</td>{''.join(f"<td>{person2_directional[view]['mean']:.3f}</td>" for view in directional_views if view in person2_directional)}</tr>
                <tr><td>Std Dev:</td>{''.join(f"<td>{person2_directional[view]['std']:.3f}</td>" for view in directional_views if view in person2_directional)}</tr>
                <tr><td>Min:</td>{''.join(f"<td>{person2_directional[view]['min']:.3f}</td>" for view in directional_views if view in person2_directional)}</tr>
                <tr><td>Max:</td>{''.join(f"<td>{person2_directional[view]['max']:.3f}</td>" for view in directional_views if view in person2_directional)}</tr>
                <tr><td>Median:</td>{''.join(f"<td>{person2_directional[view]['median']:.3f}</td>" for view in directional_views if view in person2_directional)}</tr>
            </table>
        </div>
        """)
    
    return '\n'.join(tables)


# ==============================
# HTML Template Generation
# ==============================

def generate_html_template(
    experiment_name: str,
    num_hdris: int,
    hdri_names: List[str],
    charts_js: str,
    filtering_js: str,
    stats_tables: str,
    plots_3d_js: str
) -> str:
    """Generate the complete HTML template."""
    
    individual_links = ''.join([
        f'<a href="{name}/{name}_report.html" class="individual-link" data-hdri-name="{name}">{name}</a>' 
        for name in hdri_names
    ])
    
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Aggregate Statistics — {experiment_name}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
    * {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }}
    
    body {{
        font-family: 'Courier New', monospace;
        line-height: 1.4;
        color: #000;
        background: #c0c0c0;
        min-height: 100vh;
        margin: 0;
        padding: 0;
    }}
    
    .container {{
        max-width: 100%;
        margin: 0;
        padding: 10px;
    }}
    
    .header {{
        text-align: center;
        color: #000;
        margin-bottom: 10px;
        background: #808080;
        padding: 10px;
        border: 2px inset #c0c0c0;
    }}
    
    .header h1 {{
        font-size: 1.6rem;
        margin: 0;
        font-weight: bold;
    }}
    
    .navigation-section {{
        background: #fff;
        padding: 10px;
        border: 2px inset #c0c0c0;
        margin-bottom: 10px;
    }}
    
    .navigation-section h2 {{
        color: #000;
        margin: 0 0 10px 0;
        font-size: 1.2rem;
        background: #c0c0c0;
        padding: 5px;
        border: 1px outset #c0c0c0;
        font-weight: bold;
    }}
    
    .individual-links {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 5px;
        max-height: 400px;
        overflow-y: auto;
        border: 1px inset #c0c0c0;
        padding: 10px;
        background: #f8f8f8;
        scrollbar-width: thin;
        scrollbar-color: #808080 #c0c0c0;
    }}
    
    .individual-links::-webkit-scrollbar {{
        width: 16px;
    }}
    
    .individual-links::-webkit-scrollbar-track {{
        background: #c0c0c0;
        border: 1px inset #c0c0c0;
    }}
    
    .individual-links::-webkit-scrollbar-thumb {{
        background: #808080;
        border: 1px outset #808080;
        border-radius: 0;
    }}
    
    .individual-links::-webkit-scrollbar-thumb:hover {{
        background: #606060;
    }}
    
    .individual-link {{
        background: #f0f0f0;
        padding: 8px;
        border: 1px outset #c0c0c0;
        text-decoration: none;
        color: #000;
        text-align: center;
        font-size: 0.9rem;
    }}
    
    .individual-link:hover {{
        background: #e0e0e0;
        border: 1px inset #c0c0c0;
    }}
    
    .individual-link.intensity-filtered {{
        background: #e8f5e8 !important;
        border: 1px outset #4CAF50 !important;
    }}
    
    .individual-link.intensity-filtered:hover {{
        background: #d0f0d0 !important;
        border: 1px inset #4CAF50 !important;
    }}
    
    .overview-section {{
        background: #fff;
        padding: 10px;
        border: 2px inset #c0c0c0;
        margin-bottom: 10px;
    }}
    
    .overview-section h2 {{
        color: #000;
        margin: 0 0 10px 0;
        font-size: 1.2rem;
        background: #c0c0c0;
        padding: 5px;
        border: 1px outset #c0c0c0;
        font-weight: bold;
    }}
    
    .overview-stats {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
    }}
    
    .overview-item {{
        background: #f0f0f0;
        padding: 10px;
        border: 1px inset #c0c0c0;
        text-align: center;
    }}
    
    .overview-item h4 {{
        color: #000;
        margin-bottom: 5px;
        font-size: 0.9rem;
        font-weight: bold;
    }}
    
    .overview-item .value {{
        font-size: 1.5rem;
        font-weight: bold;
        color: #000;
    }}
    
    .intensity-section {{
        background: #fff;
        padding: 15px;
        border: 2px inset #c0c0c0;
        margin-bottom: 20px;
    }}
    
    .intensity-section h2 {{
        color: #000;
        margin: 0 0 15px 0;
        font-size: 1.3rem;
        background: #c0c0c0;
        padding: 8px;
        border: 1px outset #c0c0c0;
        font-weight: bold;
    }}
    
    .chart-container {{
        margin-bottom: 20px;
        background: #f8f8f8;
        padding: 15px;
        border: 1px inset #c0c0c0;
        position: relative;
    }}
    
    .chart-container canvas {{
        max-height: 300px;
        display: block;
    }}
    
    /* Special styling for 3D plot containers */
    .chart-container:has([id$="3D"]) {{
        padding: 2px;
        background: #e3f2fd;
        margin-bottom: 10px;
    }}
    
    .chart-container > div[id$="3D"] {{
        width: 100% !important;
        height: 100% !important;
    }}
    
    .chart-controls {{
        margin-top: 10px;
        text-align: center;
        position: relative;
        z-index: 20;
    }}
    
    .chart-controls button {{
        background: #c0c0c0;
        border: 1px outset #c0c0c0;
        padding: 5px 10px;
        cursor: pointer;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        color: #000;
        margin: 0 5px;
    }}
    
    .chart-controls button:hover {{
        background: #e0e0e0;
    }}
    
    .chart-controls button:active {{
        border: 1px inset #c0c0c0;
    }}
    
    .stats-section {{
        background: #fff;
        padding: 15px;
        border: 2px inset #c0c0c0;
        margin-bottom: 20px;
    }}
    
    .stats-section h2 {{
        color: #000;
        margin: 0 0 15px 0;
        font-size: 1.3rem;
        background: #c0c0c0;
        padding: 8px;
        border: 1px outset #c0c0c0;
        font-weight: bold;
    }}
    
    .stats-table {{
        margin-bottom: 20px;
        background: #f8f8f8;
        padding: 10px;
        border: 1px inset #c0c0c0;
    }}
    
    .stats-table h4 {{
        color: #000;
        margin: 0 0 10px 0;
        font-size: 1rem;
        background: #e0e0e0;
        padding: 5px;
        border: 1px outset #c0c0c0;
        font-weight: bold;
    }}
    
    .stats-table table {{
        width: 100%;
        border-collapse: collapse;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
    }}
    
    .stats-table th,
    .stats-table td {{
        padding: 5px 10px;
        border: 1px solid #ccc;
        text-align: left;
    }}
    
    .stats-table th {{
        background: #e0e0e0;
        font-weight: bold;
    }}
    
    .stats-table td {{
        background: #fff;
    }}
    
    .footer {{
        text-align: center;
        color: #000;
        margin-top: 20px;
        background: #c0c0c0;
        padding: 10px;
        border: 1px inset #c0c0c0;
        font-size: 0.9rem;
    }}
    
    #selection-range {{
        font-weight: bold;
        color: #4CAF50;
    }}
    
    #chart-status {{
        color: #666;
        font-size: 0.9rem;
    }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>Aggregate Lighting Analysis Statistics</h1>
        <p>Experiment: {experiment_name}</p>
        <p>Dataset: {num_hdris} HDRI files</p>
    </div>

    <div class="navigation-section">
        <h2>Individual Reports</h2>
        <div style="margin-bottom: 10px;">
            <input type="text" id="hdri-search" placeholder="Search HDRI names..." 
                   style="width: calc(100% - 200px); padding: 8px; font-family: 'Courier New', monospace; 
                          border: 1px inset #c0c0c0; background: #fff; display: inline-block;">
            <span id="hdri-counter" style="float: right; padding: 8px; background: #e0e0e0; 
                  border: 1px outset #c0c0c0; font-weight: bold;">
                {num_hdris} files
            </span>
        </div>
        <div id="intensity-filter-indicator" style="display: none; margin-bottom: 10px; padding: 5px; 
             background: #e8f5e8; border: 1px solid #4CAF50; font-size: 0.9rem; color: #2e7d32;">
            🎯 Reports filtered by global intensity range
        </div>
        <div class="individual-links">
            {individual_links}
        </div>
    </div>

    <div class="overview-section">
        <h2>Dataset Overview</h2>
        <div class="overview-stats">
            <div class="overview-item"><h4>Total HDRIs</h4><div class="value">{num_hdris}</div></div>
            <div class="overview-item"><h4>Experiment</h4><div class="value">{experiment_name}</div></div>
            <div class="overview-item"><h4>Analysis Type</h4><div class="value">Lighting Distribution</div></div>
        </div>
    </div>

    <div class="intensity-section">
        <h2>Intensity Distribution & Interactive Filtering</h2>
        
        <!-- Main intensity charts -->
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:20px;">
            <div class="chart-container">
                <canvas id="globalIntensityChart"></canvas>
                <div class="chart-controls">
                    <span>🎯 Click & Drag to Filter: </span>
                    <span id="selection-range">Full Range</span>
                    <button onclick="clearBrushSelection()">Clear</button>
                    <span id="chart-status">Ready</span>
                    </div>
                        </div>
            <div class="chart-container">
                <canvas id="areaIntensityChart"></canvas>
        </div>
    </div>

        <!-- Person render charts -->
        <h3 style="color:#000; margin:20px 0 15px 0; font-size:1.1rem; background:#e0e0e0; padding:5px; border:1px outset #c0c0c0;">Person Render Intensity Analysis</h3>
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:20px;">
            <div class="chart-container"><canvas id="personAverageChart"></canvas></div>
            <div class="chart-container"><canvas id="person1DirectionalChart"></canvas></div>
            <div class="chart-container"><canvas id="person2DirectionalChart"></canvas></div>
        </div>
    </div>

    <div class="intensity-section">
        <h2>3D Color & Light Direction Analysis</h2>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px; width:100%;">
            <div class="chart-container" style="height:350px;">
                <h3 style="text-align:center; margin:2px 0; font-size:14px; color:#000;">Dominant Color</h3>
                <div id="dominantColor3D" style="height:calc(100% - 20px);"></div>
            </div>
            <div class="chart-container" style="height:350px;">
                <h3 style="text-align:center; margin:2px 0; font-size:14px; color:#000;">Dominant Light Direction</h3>
                <div id="dominantDirection3D" style="height:calc(100% - 20px);"></div>
            </div>
            <div class="chart-container" style="height:350px;">
                <h3 style="text-align:center; margin:2px 0; font-size:14px; color:#000;">Global Color</h3>
                <div id="globalColor3D" style="height:calc(100% - 20px);"></div>
            </div>
            <div class="chart-container" style="height:350px;">
                <h3 style="text-align:center; margin:2px 0; font-size:14px; color:#000;">DC Color</h3>
                <div id="dcColor3D" style="height:calc(100% - 20px);"></div>
            </div>
        </div>
    </div>

    <div class="stats-section">
        <h2>Summary Statistics</h2>
        {stats_tables}
    </div>

    <div class="footer">
        <p>Generated by LightingStudio Aggregate Analysis Pipeline</p>
        <p>HDRI files: {', '.join(hdri_names[:5])}{'...' if len(hdri_names) > 5 else ''}</p>
    </div>
</div>

<script>
{charts_js}

{filtering_js}

{plots_3d_js}
</script>
</body>
</html>
"""


# ==============================
# Main Orchestration
# ==============================

def generate_aggregate_statistics_html(experiment_dir: Path) -> str:
    """Generate an HTML page with aggregate statistics for an experiment."""
    print(f"Starting aggregate statistics generation for: {experiment_dir}")
    
    # Step 1: Collect raw metrics
    metrics_data = collect_experiment_metrics(experiment_dir)
    
    if not metrics_data['hdri_names']:
        raise ValueError(f"No HDRI data found in experiment directory: {experiment_dir}")
    
    print(f"Raw metrics data collected:")
    for key, values in metrics_data.items():
        print(f"  {key}: {len(values)} items")
    
    # Step 2: Create DataFrame
    df = create_dataframe_from_metrics(metrics_data)
    experiment_name = experiment_dir.name
    print(f"Processing {len(df)} HDRIs for experiment: {experiment_name}")
    print(f"Created DataFrame with shape: {df.shape}")
    
    # Step 3: Prepare visualization data
    print(f"Using all {len(df)} HDRIs for visualization.")
    print("Creating visualizations...")
    viz_data = prepare_visualization_data(df)
    
    # Log visualization creation details
    if viz_data.get('global_intensity') and viz_data['global_intensity'][0]:
        print(f"  Creating intensity histograms from {len(df)} intensity values...")
        print(f"  Global intensity histogram: {len(viz_data['global_intensity'][0])} bins")
    
    if viz_data.get('area_intensity') and viz_data['area_intensity']['r'][0]:
        print(f"  Area intensity histogram: {len(viz_data['area_intensity']['r'][0])} bins")
    
    if viz_data.get('person_average'):
        print("  Creating person render intensity histograms...")
        print("  Person average intensity histograms created")
    
    if viz_data.get('person_0001_directional'):
        print("  Person directional intensity histograms created")
    
    # Log 3D visualization creation
    print("  Creating CIELAB visualizations...")
    for color_type in ['global', 'dc', 'dominant']:
        key = f'{color_type}_color_lab'
        if viz_data.get(key):
            print(f"  {color_type.capitalize()} color LAB: {len(viz_data[key])} points")
    
    print("  Creating 3D RGB visualizations...")
    for color_type in ['global', 'dc', 'dominant']:
        key = f'{color_type}_color_3d'
        if viz_data.get(key):
            print(f"  {color_type.capitalize()} color 3D: {len(viz_data[key])} points")
    
    print("  Creating 3D dominant direction visualization...")
    if viz_data.get('dominant_direction_3d'):
        print(f"  Dominant direction 3D: {len(viz_data['dominant_direction_3d'])} points")
    
    # Step 4: Calculate statistics
    stats = calculate_comprehensive_stats(df)
    
    # Step 5: Generate JavaScript
    charts_js = generate_chart_js(viz_data, stats)
    filtering_js = generate_filtering_js(df.to_json(orient='records'), {}, viz_data)
    plots_3d_js = generate_3d_plot_js(viz_data)
    
    # Step 6: Generate statistics tables
    stats_tables = generate_stats_tables(stats)
    
    # Step 7: Generate final HTML
    html_content = generate_html_template(
        experiment_name,
        len(metrics_data['hdri_names']),
        metrics_data['hdri_names'],
        charts_js,
        filtering_js,
        stats_tables,
        plots_3d_js
    )
    
    # Step 8: Write output file
    html_path = experiment_dir / f"{experiment_name}_aggregate_statistics.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Generated aggregate statistics HTML: {html_path}")
    return str(html_path)


# ==============================
# Command Line Interface
# ==============================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate aggregate statistics for HDRI analysis experiment')
    parser.add_argument('experiment_dir', type=str, help='Path to experiment directory')
    
    args = parser.parse_args()
    experiment_dir = Path(args.experiment_dir)
    
    if not experiment_dir.exists():
        print(f"Error: Experiment directory does not exist: {experiment_dir}")
        exit(1)

    try:
        html_path = generate_aggregate_statistics_html(experiment_dir)
        print(f"Successfully generated aggregate statistics: {html_path}")
    except Exception as e:
        print(f"Error generating aggregate statistics: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
