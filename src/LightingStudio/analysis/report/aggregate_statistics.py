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
    """Generate JavaScript code for 3D plots."""
    js_code = []
    
    # RGB 3D plots
    color_types = ['global', 'dc', 'dominant']
    for color_type in color_types:
        key = f'{color_type}_color_3d'
        if viz_data.get(key):
            data = viz_data[key]
            js_code.append(f"""
            // {color_type.capitalize()} Color RGB Plot
            (function() {{
                const data = window.{color_type}Color3DData;
                const trace = {{
                    x: data.map(d => d.r),
                    y: data.map(d => d.g),
                    z: data.map(d => d.b),
                    mode: 'markers',
                    type: 'scatter3d',
                    text: data.map(d => d.name),
                    marker: {{
                        size: 10,
                        color: data.map(d => `rgb(${{d.r}},${{d.g}},${{d.b}})`),
                        line: {{ color: 'black', width: 1 }}
                    }},
                    hoverinfo: 'text'
                }};
                
                const layout = {{
                    title: {{
                        text: '{color_type.capitalize()} Color',
                        y: 0.95
                    }},
                    scene: {{
                        xaxis: {{ title: 'Red', range: [0, 255], dtick: 50, showgrid: true }},
                        yaxis: {{ title: 'Green', range: [0, 255], dtick: 50, showgrid: true }},
                        zaxis: {{ title: 'Blue', range: [0, 255], dtick: 50, showgrid: true }},
                        aspectmode: 'cube',
                        bgcolor: 'rgb(230, 240, 250)'  // Light blue background
                    }},
                    showlegend: false,
                    margin: {{ l: 0, r: 0, b: 0, t: 80 }},
                    paper_bgcolor: 'rgb(240, 245, 250)'  // Very light blue
                }};
                
                window.{color_type}Color3DPlot = Plotly.newPlot('{color_type}Color3D', [trace], layout);
            }})();
            """)
    
    # Dominant direction 3D plot
    if viz_data.get('dominant_direction_3d'):
        js_code.append(f"""
        // Dominant Direction 3D Plot
        (function() {{
            const data = window.dominantDirection3DData;
            const trace = {{
                x: data.map(d => d.x),
                y: data.map(d => d.y),
                z: data.map(d => d.z),
                mode: 'markers',
                type: 'scatter3d',
                text: data.map(d => d.name),
                marker: {{
                    size: 11,
                    color: data.map(d => `rgb(${{d.color[0]}},${{d.color[1]}},${{d.color[2]}})`),
                    line: {{ color: 'black', width: 1 }}
                }},
                hoverinfo: 'text',
                showlegend: false
            }};
            
            const layout = {{
                title: {{
                    text: 'Dominant Light Direction (Unit Vectors)',
                    y: 0.95
                }},
                    scene: {{
                        xaxis: {{ title: 'X', range: [-1, 1], dtick: 0.5, showgrid: true }},
                        yaxis: {{ title: 'Y', range: [-1, 1], dtick: 0.5, showgrid: true }},
                        zaxis: {{ title: 'Z', range: [-1, 1], dtick: 0.5, showgrid: true }},
                        aspectmode: 'cube',
                        bgcolor: 'rgb(230, 240, 250)'  // Light blue background
                    }},
                    showlegend: false,
                    margin: {{ l: 0, r: 0, b: 0, t: 80 }},
                    paper_bgcolor: 'rgb(240, 245, 250)'  // Very light blue
            }};
            
            window.dominantDirection3DPlot = Plotly.newPlot('dominantDirection3D', [trace], layout);
        }})();
        """)
    
    # Wrap all 3D plot initialization in a DOMContentLoaded listener
    if js_code:
        wrapped_code = """
        // Initialize 3D plots after DOM is ready
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(function() {
                """ + '\n'.join(js_code) + """
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
        }, 200);
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
        filteredData = allData.filter(row => {
            return row.global_intensity >= minIntensity && row.global_intensity <= maxIntensity;
        });
        
        // Update range display
        document.getElementById('selection-range').textContent = minIntensity.toFixed(3) + ' - ' + maxIntensity.toFixed(3);
        
        // Update status
        const statusElement = document.getElementById('chart-status');
        if (statusElement) {
            statusElement.textContent = 'Filtered: ' + filteredData.length + ' HDRIs';
            statusElement.style.color = '#4CAF50';
        }
        
        // Update all charts
        updateAllChartsWithFilter();
        
        // Update individual reports list
        const currentSearchTerm = document.getElementById('hdri-search')?.value?.toLowerCase() || '';
        filterHDRILinks(currentSearchTerm);
    }
    
    // Clear filter function
    function clearFilter() {
        filteredData = [...allData];
        
        // Remove visual selection
        const selections = document.querySelectorAll('.brush-selection');
        selections.forEach(sel => sel.remove());
        
        document.getElementById('selection-range').textContent = 'Full Range';
        
        const statusElement = document.getElementById('chart-status');
        if (statusElement) {
            statusElement.textContent = 'Showing all ' + allData.length + ' HDRIs';
            statusElement.style.color = '#666';
        }
        
        updateAllChartsWithFilter();
        
        // Update individual reports list
        const currentSearchTerm = document.getElementById('hdri-search')?.value?.toLowerCase() || '';
        filterHDRILinks(currentSearchTerm);
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
    
    // Update 3D visualizations
    function update3DVisualizations() {
        const filteredNames = filteredData.map(row => row.hdri_name);
        const isFiltered = filteredData.length < allData.length;
        
        // Update RGB plots
        if (window.globalColor3DPlot) {
            const allGlobalData = window.globalColor3DData;
            const updatedColors = allGlobalData.map(d => {
                if (isFiltered && !filteredNames.includes(d.name)) {
                    return 'rgba(200, 200, 200, 0.3)'; // Gray out unselected
                }
                return `rgb(${d.r},${d.g},${d.b})`;
            });
            Plotly.restyle('globalColor3D', {
                'marker.color': [updatedColors]
                // Don't change size - keep original
            });
        }
        
        if (window.dcColor3DPlot) {
            const allDcData = window.dcColor3DData;
            const updatedColors = allDcData.map(d => {
                if (isFiltered && !filteredNames.includes(d.name)) {
                    return 'rgba(200, 200, 200, 0.3)';
                }
                return `rgb(${d.r},${d.g},${d.b})`;
            });
            Plotly.restyle('dcColor3D', {
                'marker.color': [updatedColors]
                // Don't change size - keep original
            });
        }
        
        if (window.dominantColor3DPlot) {
            const allDominantData = window.dominantColor3DData;
            const updatedColors = allDominantData.map(d => {
                if (isFiltered && !filteredNames.includes(d.name)) {
                    return 'rgba(200, 200, 200, 0.3)';
                }
                return `rgb(${d.r},${d.g},${d.b})`;
            });
            Plotly.restyle('dominantColor3D', {
                'marker.color': [updatedColors]
                // Don't change size - keep original
            });
        }
        
        // Update dominant direction plot
        if (window.dominantDirection3DPlot) {
            const allDirData = window.dominantDirection3DData;
            const updatedColors = allDirData.map(d => {
                if (isFiltered && !filteredNames.includes(d.name)) {
                    return 'rgba(200, 200, 200, 0.3)';
                }
                return `rgb(${d.color[0]},${d.color[1]},${d.color[2]})`;
            });
            Plotly.restyle('dominantDirection3D', {
                'marker.color': [updatedColors]
                // Don't change size - keep original
            });
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
        <h2>3D Color Distribution (RGB)</h2>
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:20px;">
            <div class="chart-container"><div id="globalColor3D" style="height:400px;"></div></div>
            <div class="chart-container"><div id="dcColor3D" style="height:400px;"></div></div>
            <div class="chart-container"><div id="dominantColor3D" style="height:400px;"></div></div>
        </div>
    </div>

    <div class="intensity-section">
        <h2>Dominant Light Direction</h2>
        <div class="chart-container" style="max-width:600px; margin:0 auto;">
            <div id="dominantDirection3D" style="height:500px;"></div>
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
