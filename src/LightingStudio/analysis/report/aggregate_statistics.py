# """
# Aggregate Statistics Generator for HDRI Analysis
# Generates aggregate statistics webpage showing distribution plots across entire dataset.
# """

# import json
# import math
# from pathlib import Path
# from typing import Dict, List, Tuple, Optional
# import numpy as np
# import colorsys

# def collect_experiment_metrics(experiment_dir: Path) -> Dict[str, List]:
#     """
#     Collect all metrics from an experiment directory.
    
#     :param experiment_dir: Path to experiment directory (e.g., jasmine-bat)
#     :return: Dictionary containing lists of metrics for each category
#     """
    
#     metrics_data = {
#         'hdri_names': [],
#         # Naive metrics
#         'global_color': [],
#         'global_intensity': [],
#         # SPH metrics
#         'dc_color': [],
#         'dominant_color': [],
#         'area_intensity': []
#     }
    
#     # Find all HDRI subdirectories
#     for hdri_dir in experiment_dir.iterdir():
#         if hdri_dir.is_dir():
#             hdri_name = hdri_dir.name
            
#             # Load naive metrics
#             naive_metrics_path = hdri_dir / f"{hdri_name}_naive_metrics.json"
#             if naive_metrics_path.exists():
#                 with open(naive_metrics_path, 'r') as f:
#                     naive_data = json.load(f)
#                     metrics_data['hdri_names'].append(hdri_name)
#                     metrics_data['global_color'].append(naive_data.get('global_color', [0, 0, 0]))
#                     metrics_data['global_intensity'].append(naive_data.get('global_intensity', 0.0))
            
#             # Load SPH metrics
#             sph_metrics_path = hdri_dir / f"{hdri_name}_sph_metrics.json"
#             if sph_metrics_path.exists():
#                 with open(sph_metrics_path, 'r') as f:
#                     sph_data = json.load(f)
#                     # Only add if we haven't already added from naive metrics
#                     if hdri_name not in metrics_data['hdri_names']:
#                         metrics_data['hdri_names'].append(hdri_name)
#                         # Add placeholder naive metrics if missing
#                         metrics_data['global_color'].append([0, 0, 0])
#                         metrics_data['global_intensity'].append(0.0)
                    
#                     metrics_data['dc_color'].append(sph_data.get('dc_color', [0, 0, 0]))
#                     metrics_data['dominant_color'].append(sph_data.get('dominant_color', [0, 0, 0]))
#                     metrics_data['area_intensity'].append(sph_data.get('area_intensity', [0, 0, 0]))
    
#     return metrics_data


# def create_histogram_data(values: List[float], bins: int = 20) -> Tuple[List[float], List[float]]:
#     """
#     Create histogram data for a list of values.
    
#     :param values: List of numeric values
#     :param bins: Number of histogram bins
#     :return: Tuple of (bin_centers, counts)
#     """
#     if not values:
#         return [], []
    
#     values = np.array(values)
#     hist, bin_edges = np.histogram(values, bins=bins)
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
#     return bin_centers.tolist(), hist.tolist()


# def create_rgb_histogram_data(rgb_values: List[List[float]], bins: int = 20) -> Dict[str, Tuple[List[float], List[float]]]:
#     """
#     Create histogram data for RGB values.
    
#     :param rgb_values: List of [R, G, B] values
#     :param bins: Number of histogram bins
#     :return: Dictionary with 'r', 'g', 'b' keys containing histogram data
#     """
#     if not rgb_values:
#         return {'r': ([], []), 'g': ([], []), 'b': ([], [])}
    
#     rgb_array = np.array(rgb_values)
    
#     result = {}
#     for i, channel in enumerate(['r', 'g', 'b']):
#         channel_values = rgb_array[:, i]
#         result[channel] = create_histogram_data(channel_values.tolist(), bins)
    
#     return result


# def rgb_to_hsv_normalized(rgb: List[float]) -> Tuple[float, float, float]:
#     """
#     Convert RGB values to HSV with proper normalization.
    
#     :param rgb: [R, G, B] values (0-255 range)
#     :return: (H, S, V) where H is 0-360, S and V are 0-1
#     """
#     r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
#     h, s, v = colorsys.rgb_to_hsv(r, g, b)
#     return h * 360, s, v  # Convert hue to degrees


# def create_color_scatter_data(rgb_values: List[List[float]], hdri_names: List[str]) -> Dict:
#     """
#     Create scatter plot data for color visualization.
    
#     :param rgb_values: List of [R, G, B] values
#     :param hdri_names: List of corresponding HDRI names
#     :return: Dictionary with scatter plot data for color wheel and color space
#     """
#     if not rgb_values:
#         return {'polar': [], 'cartesian': [], 'colors': [], 'names': []}
    
#     scatter_data = {
#         'polar': [],      # For color wheel: [hue_angle, saturation_radius]
#         'cartesian': [],  # For color rectangle: [hue_x, saturation_y] 
#         'colors': [],     # Hex color strings
#         'names': [],      # HDRI names
#         'brightness': []  # Brightness values for sizing
#     }
    
#     for i, rgb in enumerate(rgb_values):
#         if len(rgb) >= 3:
#             h, s, v = rgb_to_hsv_normalized(rgb)
            
#             # Polar coordinates for color wheel (hue = angle, saturation = radius)
#             hue_rad = math.radians(h)
#             x_polar = s * math.cos(hue_rad)
#             y_polar = s * math.sin(hue_rad)
#             scatter_data['polar'].append([x_polar, y_polar])
            
#             # Cartesian coordinates for color rectangle
#             scatter_data['cartesian'].append([h, s])
            
#             # Convert to display color (ensure minimum brightness for visibility)
#             display_v = max(v, 0.7)  # Ensure colors are bright enough to see
#             display_rgb = colorsys.hsv_to_rgb(h/360, s, display_v)
#             hex_color = f"#{int(display_rgb[0]*255):02x}{int(display_rgb[1]*255):02x}{int(display_rgb[2]*255):02x}"
#             scatter_data['colors'].append(hex_color)
            
#             scatter_data['brightness'].append(v)
#             scatter_data['names'].append(hdri_names[i] if i < len(hdri_names) else f"HDRI_{i}")
    
#     return scatter_data


# def generate_aggregate_statistics_html(experiment_dir: Path) -> str:
#     """
#     Generate an HTML page with aggregate statistics for an experiment.
    
#     :param experiment_dir: Path to experiment directory
#     :return: Path to the generated HTML file
#     """
    
#     # Collect all metrics from the experiment
#     metrics_data = collect_experiment_metrics(experiment_dir)
    
#     if not metrics_data['hdri_names']:
#         raise ValueError(f"No HDRI data found in experiment directory: {experiment_dir}")
    
#     experiment_name = experiment_dir.name
#     num_hdris = len(metrics_data['hdri_names'])
    
#     # Generate visualization data for each metric
#     visualizations = {}
    
#     # Intensity histograms
#     visualizations['global_intensity'] = create_histogram_data(metrics_data['global_intensity'])
#     visualizations['area_intensity'] = create_rgb_histogram_data(metrics_data['area_intensity'])
    
#     # Color scatter plots (only for actual colors)
#     visualizations['global_color'] = create_color_scatter_data(metrics_data['global_color'], metrics_data['hdri_names'])
#     visualizations['dc_color'] = create_color_scatter_data(metrics_data['dc_color'], metrics_data['hdri_names'])
#     visualizations['dominant_color'] = create_color_scatter_data(metrics_data['dominant_color'], metrics_data['hdri_names'])
    
#     # Calculate summary statistics
#     stats = calculate_summary_stats(metrics_data)
    
#     # Generate HTML content
#     html_content = _generate_aggregate_html_template(
#         experiment_name, 
#         num_hdris, 
#         visualizations, 
#         stats,
#         metrics_data['hdri_names']
#     )
    
#     # Write HTML file
#     html_path = experiment_dir / f"{experiment_name}_aggregate_statistics.html"
#     with open(html_path, 'w', encoding='utf-8') as f:
#         f.write(html_content)
    
#     print(f"Generated aggregate statistics HTML: {html_path}")
#     return str(html_path)


# def calculate_summary_stats(metrics_data: Dict[str, List]) -> Dict:
#     """Calculate summary statistics for all metrics."""
    
#     stats = {}
    
#     # Global intensity stats
#     if metrics_data['global_intensity']:
#         values = np.array(metrics_data['global_intensity'])
#         stats['global_intensity'] = {
#             'mean': float(np.mean(values)),
#             'std': float(np.std(values)),
#             'min': float(np.min(values)),
#             'max': float(np.max(values)),
#             'median': float(np.median(values))
#         }
    
#     # RGB stats for each color metric
#     for metric_name in ['global_color', 'dc_color', 'dominant_color', 'area_intensity']:
#         if metrics_data[metric_name]:
#             rgb_array = np.array(metrics_data[metric_name])
#             stats[metric_name] = {}
#             for i, channel in enumerate(['r', 'g', 'b']):
#                 values = rgb_array[:, i]
#                 stats[metric_name][channel] = {
#                     'mean': float(np.mean(values)),
#                     'std': float(np.std(values)),
#                     'min': float(np.min(values)),
#                     'max': float(np.max(values)),
#                     'median': float(np.median(values))
#                 }
    
#     return stats


# def _generate_aggregate_html_template(
#     experiment_name: str,
#     num_hdris: int,
#     visualizations: Dict,
#     stats: Dict,
#     hdri_names: List[str]
# ) -> str:
#     """Generate the HTML template for aggregate statistics."""
    
#     # Convert histogram data to JavaScript format
#     def to_js_array(data):
#         return str(data).replace("'", '"')
    
#     # Generate chart data for each metric
#     charts_js = ""
    
#     # Global Intensity Chart (remains as histogram)
#     if visualizations['global_intensity'][0]:  # Check if we have data
#         charts_js += f"""
#         // Global Intensity Chart
#         var globalIntensityCtx = document.getElementById('globalIntensityChart').getContext('2d');
#         new Chart(globalIntensityCtx, {{
#             type: 'bar',
#             data: {{
#                 labels: {to_js_array([f"{x:.3f}" for x in visualizations['global_intensity'][0]])},
#                 datasets: [{{
#                     label: 'Count',
#                     data: {to_js_array(visualizations['global_intensity'][1])},
#                     backgroundColor: 'rgba(54, 162, 235, 0.6)',
#                     borderColor: 'rgba(54, 162, 235, 1)',
#                     borderWidth: 1
#                 }}]
#             }},
#             options: {{
#                 responsive: true,
#                 plugins: {{
#                     title: {{
#                         display: true,
#                         text: 'Global Intensity Distribution'
#                     }}
#                 }},
#                 scales: {{
#                     y: {{
#                         beginAtZero: true,
#                         title: {{
#                             display: true,
#                             text: 'Count'
#                         }}
#                     }},
#                     x: {{
#                         title: {{
#                             display: true,
#                             text: 'Global Intensity'
#                         }}
#                     }}
#                 }}
#             }}
#         }});
#         """
    
#     # Area Intensity RGB Histogram
#     if visualizations['area_intensity']['r'][0]:  # Check if we have data
#         charts_js += f"""
#         // Area Intensity RGB Histogram
#         var areaIntensityCtx = document.getElementById('areaIntensityChart').getContext('2d');
#         new Chart(areaIntensityCtx, {{
#             type: 'bar',
#             data: {{
#                 labels: {to_js_array([f"{x:.2f}" for x in visualizations['area_intensity']['r'][0]])},
#                 datasets: [{{
#                     label: 'Red',
#                     data: {to_js_array(visualizations['area_intensity']['r'][1])},
#                     backgroundColor: 'rgba(255, 99, 132, 0.6)',
#                     borderColor: 'rgba(255, 99, 132, 1)',
#                     borderWidth: 1
#                 }}, {{
#                     label: 'Green',
#                     data: {to_js_array(visualizations['area_intensity']['g'][1])},
#                     backgroundColor: 'rgba(75, 192, 192, 0.6)',
#                     borderColor: 'rgba(75, 192, 192, 1)',
#                     borderWidth: 1
#                 }}, {{
#                     label: 'Blue',
#                     data: {to_js_array(visualizations['area_intensity']['b'][1])},
#                     backgroundColor: 'rgba(54, 162, 235, 0.6)',
#                     borderColor: 'rgba(54, 162, 235, 1)',
#                     borderWidth: 1
#                 }}]
#             }},
#             options: {{
#                 responsive: true,
#                 plugins: {{
#                     title: {{
#                         display: true,
#                         text: 'Dominant Area Intensity Distribution (RGB)'
#                     }}
#                 }},
#                 scales: {{
#                     y: {{
#                         beginAtZero: true,
#                         title: {{
#                             display: true,
#                             text: 'Count'
#                         }}
#                     }},
#                     x: {{
#                         title: {{
#                             display: true,
#                             text: 'Intensity Value'
#                         }}
#                     }}
#                 }}
#             }}
#         }});
#         """
    
#     # Color scatter charts (only for actual colors)
#     color_metrics = [
#         ('global_color', 'Global Color', 'globalColorChart'),
#         ('dc_color', 'L=0 Color (DC Term)', 'dcColorChart'),
#         ('dominant_color', 'Dominant Color', 'dominantColorChart')
#     ]
    
#     for metric_key, metric_title, chart_id in color_metrics:
#         viz_data = visualizations[metric_key]
#         if viz_data['polar']:  # Check if we have data
#             # Prepare scatter data for Chart.js
#             scatter_points = []
#             for i, (x, y) in enumerate(viz_data['polar']):
#                 scatter_points.append({
#                     'x': x,
#                     'y': y
#                 })
            
#             colors_js = to_js_array(viz_data['colors'])
#             names_js = to_js_array(viz_data['names']) 
#             brightness_js = to_js_array(viz_data['brightness'])
            
#             charts_js += f"""
#             // {metric_title} Color Wheel
#             var {chart_id.replace('Chart', '')}Ctx = document.getElementById('{chart_id}').getContext('2d');
#             var {chart_id.replace('Chart', '')}Colors = {colors_js};
#             var {chart_id.replace('Chart', '')}Names = {names_js};
#             var {chart_id.replace('Chart', '')}Brightness = {brightness_js};
            
#             new Chart({chart_id.replace('Chart', '')}Ctx, {{
#                 type: 'scatter',
#                 data: {{
#                     datasets: [{{
#                         label: '{metric_title}',
#                         data: {to_js_array(scatter_points)},
#                         pointBackgroundColor: {chart_id.replace('Chart', '')}Colors,
#                         pointBorderColor: '#000000',
#                         pointBorderWidth: 1,
#                         pointRadius: {chart_id.replace('Chart', '')}Brightness.map(b => Math.max(6, b * 10))
#                     }}]
#                 }},
#                 options: {{
#                     responsive: true,
#                     maintainAspectRatio: true,
#                     plugins: {{
#                         title: {{
#                             display: true,
#                             text: '{metric_title} Color Distribution',
#                             font: {{
#                                 size: 14,
#                                 weight: 'bold'
#                             }}
#                         }},
#                         legend: {{
#                             display: false
#                         }},
#                         tooltip: {{
#                             callbacks: {{
#                                 title: function(context) {{
#                                     return {chart_id.replace('Chart', '')}Names[context[0].dataIndex];
#                                 }},
#                                 label: function(context) {{
#                                     const color = {chart_id.replace('Chart', '')}Colors[context.dataIndex];
#                                     const brightness = {chart_id.replace('Chart', '')}Brightness[context.dataIndex];
#                                     return [`Color: ${{color}}`, `Brightness: ${{brightness.toFixed(2)}}`];
#                                 }}
#                             }}
#                         }}
#                     }},
#                     scales: {{
#                         x: {{
#                             type: 'linear',
#                             min: -1.1,
#                             max: 1.1,
#                             display: false
#                         }},
#                         y: {{
#                             type: 'linear',
#                             min: -1.1,
#                             max: 1.1,
#                             display: false
#                         }}
#                     }},
#                     aspectRatio: 1,
#                     interaction: {{
#                         intersect: false
#                     }}
#                 }}
#             }});
#             """
    
#     # Generate statistics tables
#     stats_tables = ""
    
#     # Global Intensity Stats
#     if 'global_intensity' in stats:
#         stat = stats['global_intensity']
#         stats_tables += f"""
#         <div class="stats-table">
#             <h4>Global Intensity Statistics</h4>
#             <table>
#                 <tr><td>Mean:</td><td>{stat['mean']:.4f}</td></tr>
#                 <tr><td>Std Dev:</td><td>{stat['std']:.4f}</td></tr>
#                 <tr><td>Min:</td><td>{stat['min']:.4f}</td></tr>
#                 <tr><td>Max:</td><td>{stat['max']:.4f}</td></tr>
#                 <tr><td>Median:</td><td>{stat['median']:.4f}</td></tr>
#             </table>
#         </div>
#         """
    
#     # RGB Stats Tables
#     for metric_key, metric_title in [
#         ('global_color', 'Global Color'),
#         ('dc_color', 'L=0 Color (DC Term)'),
#         ('dominant_color', 'Dominant Color'),
#         ('area_intensity', 'Dominant Area Intensity')
#     ]:
#         if metric_key in stats:
#             stat = stats[metric_key]
#             stats_tables += f"""
#             <div class="stats-table">
#                 <h4>{metric_title} Statistics</h4>
#                 <table>
#                     <tr><th></th><th>Red</th><th>Green</th><th>Blue</th></tr>
#                     <tr><td>Mean:</td><td>{stat['r']['mean']:.1f}</td><td>{stat['g']['mean']:.1f}</td><td>{stat['b']['mean']:.1f}</td></tr>
#                     <tr><td>Std Dev:</td><td>{stat['r']['std']:.1f}</td><td>{stat['g']['std']:.1f}</td><td>{stat['b']['std']:.1f}</td></tr>
#                     <tr><td>Min:</td><td>{stat['r']['min']:.1f}</td><td>{stat['g']['min']:.1f}</td><td>{stat['b']['min']:.1f}</td></tr>
#                     <tr><td>Max:</td><td>{stat['r']['max']:.1f}</td><td>{stat['g']['max']:.1f}</td><td>{stat['b']['max']:.1f}</td></tr>
#                     <tr><td>Median:</td><td>{stat['r']['median']:.1f}</td><td>{stat['g']['median']:.1f}</td><td>{stat['b']['median']:.1f}</td></tr>
#                 </table>
#             </div>
#             """
    
#     return f"""
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Aggregate Statistics - {experiment_name}</title>
#     <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
#     <style>
#         * {{
#             margin: 0;
#             padding: 0;
#             box-sizing: border-box;
#         }}
        
#         body {{
#             font-family: 'Courier New', monospace;
#             line-height: 1.4;
#             color: #000;
#             background: #c0c0c0;
#             min-height: 100vh;
#             margin: 0;
#             padding: 0;
#         }}
        
#         .container {{
#             max-width: 1400px;
#             margin: 0 auto;
#             padding: 10px;
#         }}
        
#         .header {{
#             text-align: center;
#             color: #000;
#             margin-bottom: 20px;
#             background: #808080;
#             padding: 10px;
#             border: 2px inset #c0c0c0;
#         }}
        
#         .header h1 {{
#             font-size: 2rem;
#             margin-bottom: 5px;
#             font-weight: bold;
#         }}
        
#         .header p {{
#             font-size: 1.2rem;
#             margin: 5px 0;
#         }}
        
#         .content {{
#             display: grid;
#             grid-template-columns: 2fr 1fr;
#             gap: 20px;
#         }}
        
#         .charts-section {{
#             background: #ffffff;
#             padding: 15px;
#             border: 2px inset #c0c0c0;
#         }}
        
#         .stats-section {{
#             background: #ffffff;
#             padding: 15px;
#             border: 2px inset #c0c0c0;
#             max-height: 800px;
#             overflow-y: auto;
#         }}
        
#         .section h2 {{
#             color: #000;
#             margin: 0 0 15px 0;
#             font-size: 1.3rem;
#             background: #c0c0c0;
#             padding: 8px;
#             border: 1px outset #c0c0c0;
#             font-weight: bold;
#         }}
        
#         .chart-container {{
#             margin-bottom: 30px;
#             background: #f8f8f8;
#             padding: 15px;
#             border: 1px inset #c0c0c0;
#         }}
        
#         .chart-container canvas {{
#             max-height: 300px;
#         }}
        
#         .color-chart-container {{
#             margin-bottom: 30px;
#             background: #f8f8f8;
#             padding: 15px;
#             border: 1px inset #c0c0c0;
#             display: flex;
#             justify-content: center;
#             align-items: center;
#         }}
        
#         .color-chart-container canvas {{
#             max-width: 350px;
#             max-height: 350px;
#             width: 350px;
#             height: 350px;
#         }}
        
#         .color-wheel-bg {{
#             position: relative;
#             width: 350px;
#             height: 350px;
#             border-radius: 50%;
#             background: conic-gradient(from 0deg, #ff0000, #ffff00, #00ff00, #00ffff, #0000ff, #ff00ff, #ff0000);
#             opacity: 0.3;
#         }}
        
#         .color-wheel-bg canvas {{
#             position: absolute;
#             top: 0;
#             left: 0;
#         }}
        
#         .stats-table {{
#             margin-bottom: 20px;
#             background: #f8f8f8;
#             padding: 10px;
#             border: 1px inset #c0c0c0;
#         }}
        
#         .stats-table h4 {{
#             color: #000;
#             margin: 0 0 10px 0;
#             font-size: 1rem;
#             background: #e0e0e0;
#             padding: 5px;
#             border: 1px outset #c0c0c0;
#             font-weight: bold;
#         }}
        
#         .stats-table table {{
#             width: 100%;
#             border-collapse: collapse;
#             font-size: 0.9rem;
#         }}
        
#         .stats-table th, .stats-table td {{
#             padding: 4px 8px;
#             border: 1px solid #808080;
#             text-align: left;
#         }}
        
#         .stats-table th {{
#             background: #d0d0d0;
#             font-weight: bold;
#         }}
        
#         .stats-table td:first-child {{
#             background: #e8e8e8;
#             font-weight: bold;
#         }}
        
#         .overview-section {{
#             background: #ffffff;
#             padding: 15px;
#             border: 2px inset #c0c0c0;
#             margin-bottom: 20px;
#             grid-column: 1 / -1;
#         }}
        
#         .overview-stats {{
#             display: grid;
#             grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
#             gap: 15px;
#             margin-top: 10px;
#         }}
        
#         .overview-item {{
#             background: #f0f0f0;
#             padding: 10px;
#             border: 1px inset #c0c0c0;
#             text-align: center;
#         }}
        
#         .overview-item h4 {{
#             color: #000;
#             margin-bottom: 5px;
#             font-size: 0.9rem;
#             font-weight: bold;
#         }}
        
#         .overview-item .value {{
#             font-size: 1.5rem;
#             font-weight: bold;
#             color: #000;
#         }}
        
#         .footer {{
#             text-align: center;
#             color: #000;
#             margin-top: 20px;
#             background: #c0c0c0;
#             padding: 10px;
#             border: 1px inset #c0c0c0;
#             font-size: 0.9rem;
#             grid-column: 1 / -1;
#         }}
        
#         @media (max-width: 1024px) {{
#             .content {{
#                 grid-template-columns: 1fr;
#             }}
            
#             .overview-stats {{
#                 grid-template-columns: repeat(2, 1fr);
#             }}
#         }}
        
#         @media (max-width: 768px) {{
#             .container {{
#                 padding: 5px;
#             }}
            
#             .overview-stats {{
#                 grid-template-columns: 1fr;
#             }}
            
#             .header h1 {{
#                 font-size: 1.5rem;
#             }}
            
#             .header p {{
#                 font-size: 1rem;
#             }}
#         }}
#     </style>
# </head>
# <body>
#     <div class="container">
#         <div class="header">
#             <h1>Aggregate Lighting Analysis Statistics</h1>
#             <p>Experiment: {experiment_name}</p>
#             <p>Dataset: {num_hdris} HDRI files</p>
#         </div>
        
#         <div class="overview-section">
#             <h2>Dataset Overview</h2>
#             <div class="overview-stats">
#                 <div class="overview-item">
#                     <h4>Total HDRIs</h4>
#                     <div class="value">{num_hdris}</div>
#                 </div>
#                 <div class="overview-item">
#                     <h4>Experiment</h4>
#                     <div class="value">{experiment_name}</div>
#                 </div>
#                 <div class="overview-item">
#                     <h4>Analysis Type</h4>
#                     <div class="value">Lighting Distribution</div>
#                 </div>
#             </div>
#         </div>
        
#         <div class="content">
#             <div class="charts-section">
#                 <h2>Distribution Plots</h2>
                
#                 <div class="chart-container">
#                     <canvas id="globalIntensityChart"></canvas>
#                 </div>
                
#                 <div class="color-chart-container">
#                     <div class="color-wheel-bg">
#                         <canvas id="globalColorChart"></canvas>
#                     </div>
#                 </div>
                
#                 <div class="color-chart-container">
#                     <div class="color-wheel-bg">
#                         <canvas id="dcColorChart"></canvas>
#                     </div>
#                 </div>
                
#                 <div class="color-chart-container">
#                     <div class="color-wheel-bg">
#                         <canvas id="dominantColorChart"></canvas>
#                     </div>
#                 </div>
                
#                 <div class="chart-container">
#                     <canvas id="areaIntensityChart"></canvas>
#                 </div>
#             </div>
            
#             <div class="stats-section">
#                 <h2>Summary Statistics</h2>
#                 {stats_tables}
#             </div>
#         </div>
        
#         <div class="footer">
#             <p>Generated by LightingStudio Aggregate Analysis Pipeline</p>
#             <p>HDRI files: {', '.join(hdri_names[:5])}{'...' if len(hdri_names) > 5 else ''}</p>
#         </div>
#     </div>
    
#     <script>
#         {charts_js}
#     </script>
# </body>
# </html>
#     """


# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description="Generate aggregate statistics for an experiment")
#     parser.add_argument("experiment_dir", type=str, help="Path to experiment directory")
#     args = parser.parse_args()
    
#     experiment_dir = Path(args.experiment_dir)
#     if not experiment_dir.exists() or not experiment_dir.is_dir():
#         print(f"Error: Experiment directory does not exist: {experiment_dir}")
#         exit(1)
    
#     try:
#         html_path = generate_aggregate_statistics_html(experiment_dir)
#         print(f"Successfully generated aggregate statistics: {html_path}")
#     except Exception as e:
#         print(f"Error generating aggregate statistics: {e}")
#         exit(1)


"""
Aggregate Statistics Generator for HDRI Analysis (with CIE L*a*b* visuals)
Adds perceptual color visualizations using the CIELAB a*b* plane with a
background gamut slice at adjustable L*.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import colorsys

# ==============================
# Data collection (unchanged)
# ==============================

def collect_experiment_metrics(experiment_dir: Path) -> Dict[str, List]:
    """
    Collect all metrics from an experiment directory.
    :param experiment_dir: Path to experiment directory (e.g., jasmine-bat)
    :return: Dictionary containing lists of metrics for each category
    """
    metrics_data = {
        'hdri_names': [],
        # Naive metrics
        'global_color': [],
        'global_intensity': [],
        # SPH metrics
        'dc_color': [],
        'dominant_color': [],
        'area_intensity': [],
        'dominant_direction': []
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
                        # Add placeholder naive metrics if missing
                        metrics_data['global_color'].append([0, 0, 0])
                        metrics_data['global_intensity'].append(0.0)

                    metrics_data['dc_color'].append(sph_data.get('dc_color', [0, 0, 0]))
                    metrics_data['dominant_color'].append(sph_data.get('dominant_color', [0, 0, 0]))
                    metrics_data['area_intensity'].append(sph_data.get('area_intensity', [0, 0, 0]))
                    metrics_data['dominant_direction'].append(sph_data.get('dominant_direction', [0, 0, 0]))

    print(f"Collected metrics from {len(metrics_data['hdri_names'])} HDRIs out of {subdirs_found} directories")
    
    # Debug: Check data consistency
    expected_count = len(metrics_data['hdri_names'])
    for key, values in metrics_data.items():
        if len(values) != expected_count:
            print(f"WARNING: Data length mismatch for {key}: expected {expected_count}, got {len(values)}")
    
    return metrics_data


# ==============================
# Basic histogram helpers (unchanged)
# ==============================

def create_histogram_data(values: List[float], bins: int = 20) -> Tuple[List[float], List[float]]:
    if not values:
        return [], []
    values = np.array(values)
    hist, bin_edges = np.histogram(values, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers.tolist(), hist.tolist()


def create_rgb_histogram_data(rgb_values: List[List[float]], bins: int = 20) -> Dict[str, Tuple[List[float], List[float]]]:
    if not rgb_values:
        return {'r': ([], []), 'g': ([], []), 'b': ([], [])}
    rgb_array = np.array(rgb_values)
    result = {}
    for i, channel in enumerate(['r', 'g', 'b']):
        channel_values = rgb_array[:, i]
        result[channel] = create_histogram_data(channel_values.tolist(), bins)
    return result


# ==============================
# RGB -> HSV (for legacy color wheel)
# ==============================

def rgb_to_hsv_normalized(rgb: List[float]) -> Tuple[float, float, float]:
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return h * 360, s, v  # H in degrees, S & V in 0-1


def create_color_scatter_data(rgb_values: List[List[float]], hdri_names: List[str]) -> Dict:
    if not rgb_values:
        return {'polar': [], 'cartesian': [], 'colors': [], 'names': [], 'brightness': []}

    scatter_data = {
        'polar': [],      # [x_polar, y_polar] from hue & saturation
        'cartesian': [],  # [h, s]
        'colors': [],     # hex color
        'names': [],
        'brightness': []  # V from HSV
    }

    for i, rgb in enumerate(rgb_values):
        if len(rgb) >= 3:
            h, s, v = rgb_to_hsv_normalized(rgb)
            hue_rad = math.radians(h)
            x_polar = s * math.cos(hue_rad)
            y_polar = s * math.sin(hue_rad)
            scatter_data['polar'].append([x_polar, y_polar])
            scatter_data['cartesian'].append([h, s])

            # Use a brighter swatch for visibility in HSV view
            display_v = max(v, 0.7)
            display_rgb = colorsys.hsv_to_rgb(h/360, s, display_v)
            hex_color = f"#{int(display_rgb[0]*255):02x}{int(display_rgb[1]*255):02x}{int(display_rgb[2]*255):02x}"
            scatter_data['colors'].append(hex_color)

            scatter_data['brightness'].append(v)
            scatter_data['names'].append(hdri_names[i] if i < len(hdri_names) else f"HDRI_{i}")

    return scatter_data


# ==============================
# sRGB -> CIELAB conversion (vectorized, D65 white)
# ==============================

_D65 = np.array([0.95047, 1.00000, 1.08883])  # Xn, Yn, Zn
_M_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])

_EPSILON = 216/24389  # 0.008856...
_KAPPA = 24389/27     # 903.296...


def _srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """srgb in [0,1]; apply inverse companding."""
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def _f_inv(t: np.ndarray) -> np.ndarray:
    # Inverse of f used in Lab conversion
    t3 = t ** 3
    return np.where(t3 > _EPSILON, t3, (t * 116 - 16) / _KAPPA)


def srgb_array_to_lab(rgb_values: List[List[float]]) -> np.ndarray:
    """Convert a list/array of sRGB [0-255] colors to CIELAB (D65). Returns Nx3 array (L*, a*, b*)."""
    if not rgb_values:
        return np.zeros((0, 3), dtype=np.float64)
    rgb = np.asarray(rgb_values, dtype=np.float64)
    rgb = np.clip(rgb, 0, 255) / 255.0
    linear = _srgb_to_linear(rgb)
    xyz = linear @ _M_RGB_TO_XYZ.T

    # Normalize by white
    xr, yr, zr = xyz[:, 0] / _D65[0], xyz[:, 1] / _D65[1], xyz[:, 2] / _D65[2]
    fx, fy, fz = np.cbrt(xr), np.cbrt(yr), np.cbrt(zr)

    # Use epsilon/kappa branch for low values
    fx = np.where(xr > _EPSILON, fx, ( _KAPPA * xr + 16) / 116)
    fy = np.where(yr > _EPSILON, fy, ( _KAPPA * yr + 16) / 116)
    fz = np.where(zr > _EPSILON, fz, ( _KAPPA * zr + 16) / 116)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    lab = np.stack([L, a, b], axis=1)
    return lab


def create_lab_scatter_data(rgb_values: List[List[float]], hdri_names: List[str]) -> Dict:
    """Create scatter plot data on the CIELAB a*b* plane."""
    if not rgb_values:
        return {'ab': [], 'L': [], 'colors': [], 'names': []}

    try:
        lab = srgb_array_to_lab(rgb_values)
        ab = lab[:, 1:3].tolist()  # a*, b*
        L = lab[:, 0].tolist()     # L*

        # Use the original sRGB swatch for point color (no forced brightening)
        def rgb_to_hex(rgb):
            r, g, b = [int(np.clip(c, 0, 255)) for c in rgb]
            return f"#{r:02x}{g:02x}{b:02x}"

        colors = [rgb_to_hex(rgb) for rgb in rgb_values]
        names = [hdri_names[i] if i < len(hdri_names) else f"HDRI_{i}" for i in range(len(rgb_values))]

        return {'ab': ab, 'L': L, 'colors': colors, 'names': names}
    except Exception as e:
        print(f"Error in create_lab_scatter_data: {e}")
        return {'ab': [], 'L': [], 'colors': [], 'names': []}


def create_3d_rgb_scatter_data(rgb_values: List[List[float]], hdri_names: List[str]) -> Dict:
    """Create 3D scatter plot data for RGB values as 3D coordinates."""
    if not rgb_values:
        return {'xyz': [], 'colors': [], 'names': []}

    try:
        # Normalize RGB values to 0-1 range for better 3D visualization
        rgb_array = np.array(rgb_values)
        rgb_normalized = rgb_array / 255.0
        
        # Create 3D coordinates where R=x, G=y, B=z
        xyz_coords = rgb_normalized.tolist()
        
        # Use the original RGB values as colors for the points
        def rgb_to_hex(rgb):
            r, g, b = [int(np.clip(c, 0, 255)) for c in rgb]
            return f"rgb({r}, {g}, {b})"

        colors = [rgb_to_hex(rgb) for rgb in rgb_values]
        names = [hdri_names[i] if i < len(hdri_names) else f"HDRI_{i}" for i in range(len(rgb_values))]

        return {'xyz': xyz_coords, 'colors': colors, 'names': names}
    except Exception as e:
        print(f"Error in create_3d_rgb_scatter_data: {e}")
        return {'xyz': [], 'colors': [], 'names': []}


def create_3d_direction_scatter_data(direction_values: List[List[float]], hdri_names: List[str], dominant_colors: List[List[float]] = None) -> Dict:
    """Create 3D scatter plot data for dominant direction vectors."""
    if not direction_values:
        return {'xyz': [], 'colors': [], 'names': []}

    try:
        # Direction values should already be normalized unit vectors
        # Create 3D coordinates where first component=x, second=y, third=z
        xyz_coords = [[float(d[0]), float(d[1]), float(d[2])] for d in direction_values if len(d) >= 3]
        
        # Use the actual dominant colors if provided, otherwise fall back to direction-based colors
        if dominant_colors and len(dominant_colors) == len(direction_values):
            def rgb_to_color_string(rgb):
                r, g, b = [int(np.clip(c, 0, 255)) for c in rgb]
                return f"rgb({r}, {g}, {b})"
            
            colors = [rgb_to_color_string(color) for color in dominant_colors if len(color) >= 3]
        else:
            # Fallback: create colors based on direction - use direction components as RGB
            # Map from [-1,1] to [0,255] for color visualization
            def direction_to_rgb_color(direction):
                # Normalize direction components from [-1,1] to [0,1] then to [0,255]
                r = int(np.clip((direction[0] + 1) * 127.5, 0, 255))
                g = int(np.clip((direction[1] + 1) * 127.5, 0, 255))
                b = int(np.clip((direction[2] + 1) * 127.5, 0, 255))
                return f"rgb({r}, {g}, {b})"

            colors = [direction_to_rgb_color(direction) for direction in direction_values if len(direction) >= 3]
        
        names = [hdri_names[i] if i < len(hdri_names) else f"HDRI_{i}" for i in range(len(direction_values))]

        return {'xyz': xyz_coords, 'colors': colors, 'names': names}
    except Exception as e:
        print(f"Error in create_3d_direction_scatter_data: {e}")
        return {'xyz': [], 'colors': [], 'names': []}


# ==============================
# Main HTML generator (extended with LAB views)
# ==============================

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
    
    return pd.DataFrame(data_dict)


def create_filter_configuration_for_existing(df: pd.DataFrame, stats: Dict) -> Dict[str, Any]:
    """Create filter configuration for existing aggregate statistics interface."""
    
    filters = {}
    
    # Global intensity filter (primary request)
    if 'global_intensity' in stats:
        intensity_stats = stats['global_intensity']
        filters['global_intensity'] = {
            'label': 'Global Intensity',
            'type': 'range',
            'min': float(intensity_stats['min']),
            'max': float(intensity_stats['max']),
            'default_min': float(intensity_stats['q25']),
            'default_max': float(intensity_stats['q75']),
            'step': (float(intensity_stats['max']) - float(intensity_stats['min'])) / 100,
            'column': 'global_intensity'
        }
    
    # Brightness filters for different color metrics
    color_metrics = ['global', 'dc', 'dominant', 'area']
    for metric in color_metrics:
        brightness_key = f'{metric}_brightness'
        if brightness_key in stats:
            brightness_stats = stats[brightness_key]
            filters[f'{metric}_brightness'] = {
                'label': f'{metric.title()} Brightness',
                'type': 'range',
                'min': float(brightness_stats['min']),
                'max': float(brightness_stats['max']),
                'default_min': float(brightness_stats['min']),
                'default_max': float(brightness_stats['max']),
                'step': (float(brightness_stats['max']) - float(brightness_stats['min'])) / 100,
                'column': f'{metric}_brightness'  # This will be calculated dynamically
            }
    
    # Individual RGB channel filters for global color
    for channel in ['r', 'g', 'b']:
        col_name = f'global_{channel}'
        if col_name in df.columns:
            channel_min = float(df[col_name].min())
            channel_max = float(df[col_name].max())
            filters[f'global_{channel}'] = {
                'label': f'Global {channel.upper()} Channel',
                'type': 'range',
                'min': channel_min,
                'max': channel_max,
                'default_min': channel_min,
                'default_max': channel_max,
                'step': (channel_max - channel_min) / 100,
                'column': col_name
            }
    
    return filters


def _generate_filter_controls_html_for_existing(filter_config: Dict) -> str:
    """Generate HTML for dashboard-style filter controls."""
    
    if not filter_config:
        return ""
    
    # Dashboard-style summary cards
    html = '''
    <div class="dashboard-summary">
        <h2>📊 HDRI Analytics Dashboard</h2>
        <div class="summary-cards">
            <div class="summary-card">
                <div class="card-icon">🎯</div>
                <div class="card-content">
                    <div class="card-value" id="filtered-count-dashboard">0</div>
                    <div class="card-label">Filtered HDRIs</div>
                </div>
            </div>
            <div class="summary-card">
                <div class="card-icon">📈</div>
                <div class="card-content">
                    <div class="card-value" id="avg-intensity-dashboard">—</div>
                    <div class="card-label">Avg Intensity</div>
                </div>
            </div>
            <div class="summary-card">
                <div class="card-icon">⚡</div>
                <div class="card-content">
                    <div class="card-value" id="filter-efficiency-dashboard">100%</div>
                    <div class="card-label">Data Shown</div>
                </div>
            </div>
        </div>
    </div>
    '''
    
    return html


def _generate_range_filter_html_for_existing(filter_key: str, config: Dict) -> str:
    """Generate HTML for a single range filter in existing interface."""
    
    return f'''
    <div style="margin-bottom:15px;">
        <label for="{filter_key}" style="display:block; font-weight:bold; margin-bottom:5px;">{config['label']}</label>
        <div style="display:flex; align-items:center; gap:5px; margin-bottom:5px;">
            <input type="range" 
                   id="{filter_key}_min" 
                   style="flex:1;"
                   min="{config['min']}" 
                   max="{config['max']}" 
                   step="{config['step']}" 
                   value="{config['default_min']}"
                   oninput="updateFilterExisting('{filter_key}')">
            <input type="range" 
                   id="{filter_key}_max" 
                   style="flex:1;"
                   min="{config['min']}" 
                   max="{config['max']}" 
                   step="{config['step']}" 
                   value="{config['default_max']}"
                   oninput="updateFilterExisting('{filter_key}')">
        </div>
        <div style="display:flex; gap:5px; align-items:center; font-size:0.8rem;">
            Min: <input type="number" id="{filter_key}_min_val" value="{config['default_min']:.3f}" 
                       step="{config['step']}" style="width:70px; padding:2px; font-size:0.8rem;"
                       onchange="updateFilterFromInputExisting('{filter_key}', 'min')">
            Max: <input type="number" id="{filter_key}_max_val" value="{config['default_max']:.3f}"
                       step="{config['step']}" style="width:70px; padding:2px; font-size:0.8rem;"
                       onchange="updateFilterFromInputExisting('{filter_key}', 'max')">
        </div>
    </div>
    '''


def _generate_filter_javascript_for_existing(df_json: str, filter_config: Dict, hdri_names: List[str]) -> str:
    """Generate JavaScript code for filtering functionality in existing interface."""
    
    def to_js(x):
        return str(x).replace("'", '"')
    
    return f'''
// Filtering system for existing aggregate statistics
let allDataExisting = {df_json};
let filteredDataExisting = [...allDataExisting];
let originalChartsExisting = {{}};

// Filter configuration
const filterConfigExisting = {to_js(filter_config)};

// Initialize filtering system
// Brush selection state
let brushSelection = null;
let isDragging = false;
let chartsInitialized = false;
let plotly3dReady = false;
let intensityChart = null;

// Debug flag to control logging
const DEBUG = false;
function dbg() {{
    if (!DEBUG) return;
    try {{ console.log.apply(console, arguments); }} catch (_) {{}}
}}

document.addEventListener('DOMContentLoaded', function() {{
    
    // Add calculated columns
    allDataExisting.forEach(row => {{
        row.global_brightness = (row.global_r + row.global_g + row.global_b) / 3;
        row.dc_brightness = (row.dc_r + row.dc_g + row.dc_b) / 3;
        row.dominant_brightness = (row.dominant_r + row.dominant_g + row.dominant_b) / 3;
        row.area_brightness = (row.area_r + row.area_g + row.area_b) / 3;
    }});
    
    // Calculate initial intensity range
    const intensities = allDataExisting.map(row => row.global_intensity);
    intensityRange.min = Math.min(...intensities);
    intensityRange.max = Math.max(...intensities);
    dbg(`Intensity range calculated: min=${{intensityRange.min}}, max=${{intensityRange.max}}, data points: ${{intensities.length}}`);
    dbg(`Sample intensities: ${{intensities.slice(0, 5).map(v => v.toFixed(3)).join(', ')}}`);
    
    // Don't update dashboard on initial load - wait for charts to be ready
    // updateDashboardSummary();
    
    // Initialize status text
    const statusElement = document.getElementById('chart-status');
    if (statusElement) {{
        statusElement.textContent = `Showing all ${{allDataExisting.length}} HDRIs`;
        statusElement.style.color = '#666';
    }}
    
    // Set initial dashboard values without triggering updates
    document.getElementById('filtered-count-dashboard').textContent = allDataExisting.length;
    const avgIntensity = intensities.reduce((a, b) => a + b, 0) / intensities.length;
    document.getElementById('avg-intensity-dashboard').textContent = avgIntensity.toFixed(2);
    document.getElementById('filter-efficiency-dashboard').textContent = '100%';
    
    // Initialize brush selection after charts are ready
    setTimeout(() => {{
        chartsInitialized = true;
        
        // Debug: Check if charts exist
        console.log('Checking charts after initialization:');
        console.log('Global intensity chart:', allCharts.intensity ? 'exists' : 'missing');
        console.log('Area intensity chart:', allCharts.areaIntensity ? 'exists' : 'missing');
        console.log('LAB charts - global:', allCharts.labCharts.global ? 'exists' : 'missing');
        console.log('LAB charts - dc:', allCharts.labCharts.dc ? 'exists' : 'missing');
        console.log('LAB charts - dominant:', allCharts.labCharts.dominant ? 'exists' : 'missing');
        
        // Debug: Check data
        if (allCharts.areaIntensity) {{
            console.log('Area intensity datasets:', allCharts.areaIntensity.data.datasets.length);
            allCharts.areaIntensity.data.datasets.forEach((ds, idx) => {{
                console.log(`Dataset ${{idx}}: ${{ds.label}}, data points: ${{ds.data.length}}`);
            }});
        }}
        
        initializeBrushSelection();
        
        // Wait a bit more for 3D plots to be ready
        setTimeout(() => {{
            console.log('Checking 3D plots:');
            const globalRgbDiv = document.getElementById('globalRgb3dChart');
            const globalRgbReady = globalRgbDiv && globalRgbDiv._fullData;
            console.log('Global RGB 3D:', globalRgbReady ? 'ready' : 'not ready');
            
            const dcRgbDiv = document.getElementById('dcRgb3dChart');
            const dcRgbReady = dcRgbDiv && dcRgbDiv._fullData;
            console.log('DC RGB 3D:', dcRgbReady ? 'ready' : 'not ready');
            
            const dominantRgbDiv = document.getElementById('dominantRgb3dChart');
            const dominantRgbReady = dominantRgbDiv && dominantRgbDiv._fullData;
            console.log('Dominant RGB 3D:', dominantRgbReady ? 'ready' : 'not ready');
            
            const directionDiv = document.getElementById('dominantDirection3dChart');
            const directionReady = directionDiv && directionDiv._fullData;
            console.log('Direction 3D:', directionReady ? 'ready' : 'not ready');
            
            // Check if all 3D plots are ready
            if (globalRgbReady && dcRgbReady && dominantRgbReady && directionReady) {{
                plotly3dReady = true;
                console.log('All 3D plots are ready!');
            }} else {{
                // Try again in a bit
                setTimeout(() => {{
                    plotly3dReady = true;
                    console.log('Forcing 3D plots ready after additional wait');
                }}, 2000);
            }}
        }}, 2000);
    }}, 1500);  // Increased delay to ensure all charts are ready
}});

function updateDashboardSummary() {{
    const filteredCount = filteredDataExisting.length;
    const totalCount = allDataExisting.length;
    
    // Update dashboard cards
    document.getElementById('filtered-count-dashboard').textContent = filteredCount;
    
    if (filteredCount > 0) {{
        const intensities = filteredDataExisting.map(row => row.global_intensity);
        const avgIntensity = intensities.reduce((a, b) => a + b, 0) / intensities.length;
        
        // Calculate color variance
        document.getElementById('avg-intensity-dashboard').textContent = avgIntensity.toFixed(2);
    }} else {{
        document.getElementById('avg-intensity-dashboard').textContent = '—';
    }}
    
    const efficiency = totalCount > 0 ? Math.round((filteredCount / totalCount) * 100) : 100;
    document.getElementById('filter-efficiency-dashboard').textContent = `${{efficiency}}%`;
}}

function initializeBrushSelection() {{
    const brushOverlay = document.getElementById('intensity-brush-overlay');
    if (!brushOverlay) {{
        console.error('Brush overlay not found!');
        return;
    }}
    
    console.log('Brush overlay found, initializing events');
    
    let startX = 0;
    let currentSelection = null;
    let cachedOverlayRect = null;
    let cachedCanvasRect = null;
    let cachedChartArea = null;
    
    brushOverlay.addEventListener('mousedown', function(e) {{
        console.log('Mouse down detected at:', e.offsetX);
        isDragging = true;
        // Use clientX relative to the overlay's left for stability
        const overlayRectAtStart = brushOverlay.getBoundingClientRect();
        startX = Math.max(0, Math.min(overlayRectAtStart.width, e.clientX - overlayRectAtStart.left));
        
        // Cache coordinate system data at start of drag
        const chart = window.Chart.getChart('globalIntensityChart');
        if (chart && chart.canvas && chart.chartArea) {{
            cachedOverlayRect = brushOverlay.getBoundingClientRect();
            cachedCanvasRect = chart.canvas.getBoundingClientRect();
            cachedChartArea = {{ ...chart.chartArea }}; // Copy the object
            console.log('Cached coordinate system for drag operation');
        }}
        
        // Clear existing selection
        if (currentSelection) {{
            currentSelection.remove();
        }}
        
        // Create new selection
        currentSelection = document.createElement('div');
        currentSelection.className = 'brush-selection';
        currentSelection.style.left = startX + 'px';
        currentSelection.style.top = '0px';
        currentSelection.style.width = '0px';
        currentSelection.style.height = '100%';
        brushOverlay.appendChild(currentSelection);
        
        e.preventDefault();
    }});
    
    brushOverlay.addEventListener('mousemove', function(e) {{
        if (!isDragging || !currentSelection || !cachedOverlayRect) return;
        
        // Use cached overlay rect instead of calling getBoundingClientRect repeatedly
        // Use clientX relative to overlay for stable coordinates during drag
        const currentX = Math.max(0, Math.min(cachedOverlayRect.width, e.clientX - cachedOverlayRect.left));
        
        // Simple validation - if coordinates seem invalid, skip this update
        if (isNaN(currentX) || currentX < 0 || currentX > cachedOverlayRect.width + 100) {{
            console.warn('Invalid mouse coordinates detected, skipping update:', currentX);
            return;
        }}
        
        const width = Math.abs(currentX - startX);
        const left = Math.min(startX, currentX);
        
        // Only update if the change makes sense
        if (width >= 0 && left >= 0 && left + width <= cachedOverlayRect.width + 10) {{
            currentSelection.style.left = left + 'px';
            currentSelection.style.width = width + 'px';
        }}
    }});
    
    brushOverlay.addEventListener('mouseup', function(e) {{
        if (!isDragging || !currentSelection) return;
        
        // Clamp coordinates to overlay bounds
        const overlayRect = cachedOverlayRect || brushOverlay.getBoundingClientRect();
        const endX = Math.max(0, Math.min(overlayRect.width, e.clientX - overlayRect.left));
        
        console.log('Mouse up detected at:', endX, '(original:', e.offsetX, ')');
        isDragging = false;
        
        // Apply brush selection as filter if selection is wide enough
        if (Math.abs(endX - startX) > 5) {{
            // Get the actual chart coordinates for precise mapping
            const chart = window.Chart.getChart('globalIntensityChart');
            if (chart && chart.chartArea && chart.canvas) {{
                // Get canvas position relative to viewport
                const canvasRect = chart.canvas.getBoundingClientRect();
                const overlayRect = brushOverlay.getBoundingClientRect();
                
                // Calculate offset between overlay and canvas
                const offsetX = overlayRect.left - canvasRect.left;
                const offsetY = overlayRect.top - canvasRect.top;
                
                // Adjust mouse coordinates to be relative to canvas
                const canvasStartX = startX - offsetX;
                const canvasEndX = endX - offsetX;
                
                // Ensure proper ordering regardless of selection direction
                const leftX = Math.min(canvasStartX, canvasEndX);
                const rightX = Math.max(canvasStartX, canvasEndX);
                
                // Use actual chart area for precise coordinate mapping
                const chartArea = chart.chartArea;
                const chartLeft = chartArea.left;
                const chartRight = chartArea.right;
                const chartWidth = chartRight - chartLeft;
                
                // Map pixel positions to chart data coordinates
                // Account for the fact that canvas pixel coordinates need to be scaled
                const scale = chart.canvas.width / chart.canvas.offsetWidth;
                const scaledLeftX = leftX * scale;
                const scaledRightX = rightX * scale;
                
                // Map to ratios within the chart area
                const leftRatio = Math.max(0, Math.min(1, (scaledLeftX - chartLeft) / chartWidth));
                const rightRatio = Math.max(0, Math.min(1, (scaledRightX - chartLeft) / chartWidth));
                
                // Account for visual offset - the selection appears shifted right from the actual values
                // So we need to shift our calculated values LEFT to compensate
                const binCount = 20; // Same as in updateIntensityChartWithFilter
                const barWidthRatio = 1.0 / binCount;
                
                // Shift left by approximately one bar width to align visual selection with data
                const offsetCorrection = barWidthRatio * 1.0; // Full bar width offset
                
                // Convert to intensity values with left shift to correct for visual offset
                const minValue = intensityRange.min + Math.max(0, leftRatio - offsetCorrection) * (intensityRange.max - intensityRange.min);
                const maxValue = intensityRange.min + Math.max(0, rightRatio - offsetCorrection) * (intensityRange.max - intensityRange.min);
                
                dbg(`Canvas rect: ${{canvasRect.left}}, ${{canvasRect.top}}`);
                dbg(`Overlay rect: ${{overlayRect.left}}, ${{overlayRect.top}}`);
                dbg(`Offset: ${{offsetX}}, ${{offsetY}}`);
                dbg(`Canvas scale: ${{scale}} (canvas.width=${{chart.canvas.width}}, offsetWidth=${{chart.canvas.offsetWidth}})`);
                dbg(`Original mouse: ${{startX}}-${{endX}}px`);
                dbg(`Canvas-relative: ${{canvasStartX}}-${{canvasEndX}}px`);
                dbg(`Scaled positions: ${{scaledLeftX}}-${{scaledRightX}}px`);
                dbg(`Ordered selection: ${{leftX}}-${{rightX}}px`);
                dbg(`Chart area: left=${{chartLeft}}, right=${{chartRight}}, width=${{chartWidth}}`);
                dbg(`Ratios: ${{leftRatio.toFixed(3)}}-${{rightRatio.toFixed(3)}}`);
                dbg(`Intensity range: ${{minValue.toFixed(3)}}-${{maxValue.toFixed(3)}}`);
                
                applyBrushFilter(minValue, maxValue);
            }} else {{
                dbg('Chart, chartArea, or canvas not available for coordinate mapping');
            }}
        }}
        
        brushSelection = {{ startX, endX }};
    }});
    
    // Handle mouse leave - continue dragging but clamp coordinates
    brushOverlay.addEventListener('mouseleave', function() {{
        // Don't stop dragging, let mouseup handle it
        // This prevents issues when mouse briefly leaves the overlay
    }});
    
    // Also listen for document-level mouseup to catch releases outside overlay
    document.addEventListener('mouseup', function(e) {{
        if (isDragging && currentSelection) {{
            // Calculate relative position to overlay
            const overlayRect = cachedOverlayRect || brushOverlay.getBoundingClientRect();
            const relativeX = e.clientX - overlayRect.left;
            const endX = Math.max(0, Math.min(overlayRect.width, relativeX));
            
            dbg('Document mouse up detected at:', endX, '(client:', e.clientX, ')');
            isDragging = false;
            
            // Apply brush selection if wide enough
            if (Math.abs(endX - startX) > 5) {{
                const chart = window.Chart.getChart('globalIntensityChart');
                if (chart && chart.chartArea && chart.canvas) {{
                    // Get canvas position relative to viewport
                    const canvasRect = chart.canvas.getBoundingClientRect();
                    const overlayRect = brushOverlay.getBoundingClientRect();
                    
                    // Calculate offset between overlay and canvas
                    const offsetX = overlayRect.left - canvasRect.left;
                    
                    // Adjust mouse coordinates to be relative to canvas
                    const canvasStartX = startX - offsetX;
                    const canvasEndX = endX - offsetX;
                    
                    // Ensure proper ordering regardless of selection direction
                    const leftX = Math.min(canvasStartX, canvasEndX);
                    const rightX = Math.max(canvasStartX, canvasEndX);
                    
                    const chartArea = chart.chartArea;
                    const chartLeft = chartArea.left;
                    const chartRight = chartArea.right;
                    const chartWidth = chartRight - chartLeft;
                    
                    // Account for canvas scaling
                    const scale = chart.canvas.width / chart.canvas.offsetWidth;
                    const scaledLeftX = leftX * scale;
                    const scaledRightX = rightX * scale;
                    
                    const leftRatio = Math.max(0, Math.min(1, (scaledLeftX - chartLeft) / chartWidth));
                    const rightRatio = Math.max(0, Math.min(1, (scaledRightX - chartLeft) / chartWidth));
                    
                    // Account for visual offset - same correction as above
                    const binCount = 20;
                    const barWidthRatio = 1.0 / binCount;
                    const offsetCorrection = barWidthRatio * 1.0;
                    
                    const minValue = intensityRange.min + Math.max(0, leftRatio - offsetCorrection) * (intensityRange.max - intensityRange.min);
                    const maxValue = intensityRange.min + Math.max(0, rightRatio - offsetCorrection) * (intensityRange.max - intensityRange.min);
                    
                    dbg('Document mouseup - Canvas-relative:', canvasStartX, '-', canvasEndX);
                    dbg('Document mouseup - Ordered selection:', leftX, '-', rightX);
                    dbg('Document mouseup - Intensity range:', minValue.toFixed(3), '-', maxValue.toFixed(3));
                    
                    applyBrushFilter(minValue, maxValue);
                }}
            }}
            
            brushSelection = {{ startX, endX }};
        }}
    }});
}}

function applyBrushFilter(minIntensity, maxIntensity) {{
    // Update range inputs if they exist
    const minInput = document.getElementById('global_intensity_min');
    const maxInput = document.getElementById('global_intensity_max');
    const minVal = document.getElementById('global_intensity_min_val');
    const maxVal = document.getElementById('global_intensity_max_val');
    
    if (minInput && maxInput) {{
        minInput.value = minIntensity;
        maxInput.value = maxIntensity;
        if (minVal) minVal.value = minIntensity.toFixed(3);
        if (maxVal) maxVal.value = maxIntensity.toFixed(3);
    }}
    
    // Apply filter
    filteredDataExisting = allDataExisting.filter(row => {{
        return row.global_intensity >= minIntensity && row.global_intensity <= maxIntensity;
    }});
    
    // Update selection range display
    document.getElementById('selection-range').textContent = `${{minIntensity.toFixed(2)}} - ${{maxIntensity.toFixed(2)}}`;
    
    // Update status text
    const statusElement = document.getElementById('chart-status');
    if (statusElement) {{
        statusElement.textContent = `Filtered: ${{filteredDataExisting.length}} HDRIs`;
        statusElement.style.color = '#4CAF50';
    }}
    
    updateDashboardSummary();
    updateExistingChartsWithFilter();
    updateHDRILinksWithFilter();
    
    // Force 3D plots to be ready if they haven't been marked as such
    if (!plotly3dReady && typeof Plotly !== 'undefined') {{
        const check3DPlots = () => {{
            const globalRgbDiv = document.getElementById('globalRgb3dChart');
            const dcRgbDiv = document.getElementById('dcRgb3dChart');
            const dominantRgbDiv = document.getElementById('dominantRgb3dChart');
            const directionDiv = document.getElementById('dominantDirection3dChart');
            
            if (globalRgbDiv && globalRgbDiv._fullData && 
                dcRgbDiv && dcRgbDiv._fullData && 
                dominantRgbDiv && dominantRgbDiv._fullData && 
                directionDiv && directionDiv._fullData) {{
                plotly3dReady = true;
                updateAll3DCharts();
            }}
        }};
        
        // Try immediately and after a short delay
        check3DPlots();
        setTimeout(check3DPlots, 500);
    }}
}}

function clearBrushSelection() {{
    // Remove visual selection
    const brushOverlay = document.getElementById('intensity-brush-overlay');
    if (brushOverlay) {{
        const selections = brushOverlay.querySelectorAll('.brush-selection');
        selections.forEach(sel => sel.remove());
    }}
    
    // Reset to full range
    filteredDataExisting = [...allDataExisting];
    brushSelection = null;
    
    // Update display
    document.getElementById('selection-range').textContent = 'Full Range';
    
    // Reset status text
    const statusElement = document.getElementById('chart-status');
    if (statusElement) {{
        statusElement.textContent = `Showing all ${{allDataExisting.length}} HDRIs`;
        statusElement.style.color = '#666';
    }}
    
    updateDashboardSummary();
    updateExistingChartsWithFilter();
    updateHDRILinksWithFilter();
    
    // Reset range inputs
    const minInput = document.getElementById('global_intensity_min');
    const maxInput = document.getElementById('global_intensity_max');
    const minVal = document.getElementById('global_intensity_min_val');
    const maxVal = document.getElementById('global_intensity_max_val');
    
    if (minInput && maxInput) {{
        minInput.value = intensityRange.min;
        maxInput.value = intensityRange.max;
        if (minVal) minVal.value = intensityRange.min.toFixed(3);
        if (maxVal) maxVal.value = intensityRange.max.toFixed(3);
    }}
}}

function toggleCompactFilters() {{
    const compactFilters = document.getElementById('compact-filters');
    const toggleBtn = document.getElementById('compact-toggle');
    
    if (compactFilters.style.display === 'none') {{
        compactFilters.style.display = 'block';
        toggleBtn.textContent = '🔽 Hide Advanced';
    }} else {{
        compactFilters.style.display = 'none';
        toggleBtn.textContent = '⚙️ Advanced Filters';
    }}
}}

function updateFilterExisting(filterKey) {{
    console.log('Updating filter:', filterKey);
    
    // Update the numeric input fields
    const minSlider = document.getElementById(filterKey + '_min');
    const maxSlider = document.getElementById(filterKey + '_max');
    const minInput = document.getElementById(filterKey + '_min_val');
    const maxInput = document.getElementById(filterKey + '_max_val');
    
    // Ensure min <= max
    if (parseFloat(minSlider.value) > parseFloat(maxSlider.value)) {{
        if (event.target === minSlider) {{
            maxSlider.value = minSlider.value;
        }} else {{
            minSlider.value = maxSlider.value;
        }}
    }}
    
    minInput.value = parseFloat(minSlider.value).toFixed(3);
    maxInput.value = parseFloat(maxSlider.value).toFixed(3);
    
    // Apply all filters
    applyAllFiltersExisting();
}}

function updateFilterFromInputExisting(filterKey, bound) {{
    const input = document.getElementById(filterKey + '_' + bound + '_val');
    const slider = document.getElementById(filterKey + '_' + bound);
    
    let value = parseFloat(input.value);
    const config = filterConfigExisting[filterKey];
    
    // Clamp to valid range
    value = Math.max(config.min, Math.min(config.max, value));
    
    input.value = value.toFixed(3);
    slider.value = value;
    
    // Ensure min <= max
    const minSlider = document.getElementById(filterKey + '_min');
    const maxSlider = document.getElementById(filterKey + '_max');
    
    if (parseFloat(minSlider.value) > parseFloat(maxSlider.value)) {{
        if (bound === 'min') {{
            maxSlider.value = value;
            document.getElementById(filterKey + '_max_val').value = value.toFixed(3);
        }} else {{
            minSlider.value = value;
            document.getElementById(filterKey + '_min_val').value = value.toFixed(3);
        }}
    }}
    
    applyAllFiltersExisting();
}}

function applyAllFiltersExisting() {{
    console.log('Applying all filters...');
    
    filteredDataExisting = allDataExisting.filter(row => {{
        for (const [filterKey, config] of Object.entries(filterConfigExisting)) {{
            const minValue = parseFloat(document.getElementById(filterKey + '_min').value);
            const maxValue = parseFloat(document.getElementById(filterKey + '_max').value);
            
            let columnValue;
            if (config.column) {{
                columnValue = row[config.column];
            }} else {{
                // Handle calculated columns
                columnValue = row[filterKey];
            }}
            
            if (columnValue < minValue || columnValue > maxValue) {{
                return false;
            }}
        }}
        return true;
    }});
    
    console.log('Filtered results:', filteredDataExisting.length, 'out of', allDataExisting.length);
    
    // Update dashboard summary
    updateDashboardSummary();
    
    // Update existing charts with filtered data
    updateExistingChartsWithFilter();
    
    // Update HDRI links if they exist
    updateHDRILinksWithFilter();
}}

function updateExistingChartsWithFilter() {{
    // Don't update if charts aren't initialized yet
    if (!chartsInitialized) {{
        console.log('Charts not yet initialized, skipping update');
        return;
    }}
    
    // Update global intensity chart if it exists
    if (window.Chart && window.Chart.getChart) {{
        intensityChart = window.Chart.getChart('globalIntensityChart');
        if (intensityChart) {{
            updateIntensityChartWithFilter(intensityChart);
        }}
    }}
    
    // Update all other charts with filtered data
    updateAreaIntensityChart();
    updateAllLabCharts();
    updateAll3DCharts();
}}

function updateIntensityChartWithFilter(chart) {{
    if (!chart) return;
    
    // Use the original full range for consistent scale
    const bins = 20;
    const min = intensityRange.min;
    const max = intensityRange.max;
    const binWidth = (max - min) / bins;
    
    // Create histogram with original scale
    const histogram = new Array(bins).fill(0);
    
    // Fill histogram with filtered data, but using original scale
    if (filteredDataExisting.length > 0) {{
        const filteredIntensities = filteredDataExisting.map(row => row.global_intensity);
        filteredIntensities.forEach(intensity => {{
            const binIndex = Math.min(Math.floor((intensity - min) / binWidth), bins - 1);
            if (binIndex >= 0) {{
                histogram[binIndex]++;
            }}
        }});
    }}
    
    // Update ONLY the data array, nothing else
    chart.data.datasets[0].data = histogram;
    
    // Add visual indication of filter status
    const isFiltered = filteredDataExisting.length < allDataExisting.length;
    chart.data.datasets[0].backgroundColor = isFiltered ? 
        'rgba(76, 175, 80, 0.6)' :  // Green when filtered
        'rgba(54, 162, 235, 0.6)';  // Blue when showing all data
    chart.data.datasets[0].borderColor = isFiltered ? 
        'rgba(76, 175, 80, 1)' : 
        'rgba(54, 162, 235, 1)';
    
    // Update status text
    const statusElement = document.getElementById('chart-status');
    if (statusElement) {{
        if (isFiltered) {{
            statusElement.textContent = `Filtered view`;
            statusElement.style.color = '#4CAF50';
        }} else {{
            statusElement.textContent = 'Showing all data';
            statusElement.style.color = '#666';
        }}
    }}
    
    // Normal update - y-axis will scale, x-axis stays fixed due to chart config
    chart.update();
}}

function updateHDRILinksWithFilter() {{
    const container = document.getElementById('individualLinksContainer');
    if (container) {{
        const allLinks = container.querySelectorAll('.individual-link');
        const filteredNames = new Set(filteredDataExisting.map(row => row.hdri_name));
        
        allLinks.forEach(link => {{
            const hdriName = link.getAttribute('data-hdri-name');
            if (filteredNames.has(hdriName)) {{
                link.style.display = 'inline-block';
                link.style.opacity = '1';
            }} else {{
                link.style.display = 'none';
                link.style.opacity = '0.3';
            }}
        }});
    }}
}}

// Update Area Intensity Chart - always show original
function updateAreaIntensityChart() {{
    // Per user request: always show the original plot for area intensity
    // No filtering applied to this chart
    return;
}}

// Update all LAB charts with grayed out data
function updateAllLabCharts() {{
    ['global', 'dc', 'dominant'].forEach(chartType => {{
        updateLabChart(chartType);
    }});
}}

// Update individual LAB chart
function updateLabChart(chartType) {{
    const chartId = chartType + 'LabAbChart';
    const chart = window.Chart.getChart(chartId);
    if (!chart) return;
    
    // Check if original data exists
    const originalData = originalVisualizationData.labData[chartType];
    if (!originalData || !originalData.names) {{
        console.warn(`No original data found for ${{chartType}} LAB chart`);
        return;
    }}
    
    // Check if we're showing all data (no filter)
    const isShowingAll = filteredDataExisting.length === allDataExisting.length;
    
    if (isShowingAll) {{
        // Restore original single dataset
        chart.data.datasets = [{{
            label: originalData.label || 'Data',
            data: originalData.points.slice(),
            pointBackgroundColor: originalData.colors.slice(),
            pointBorderColor: '#000',
            pointBorderWidth: 1,
            pointRadius: originalData.Lvals ? originalData.Lvals.map(L => Math.max(4, (L/100)*8 + 2)) : 5
        }}];
        
        // Restore original tooltip callbacks
        if (chart.options.plugins.tooltip.callbacks) {{
            chart.options.plugins.tooltip.callbacks.title = (ctx) => originalData.names[ctx[0].dataIndex];
            chart.options.plugins.tooltip.callbacks.label = (ctx) => {{
                const L = originalData.Lvals ? originalData.Lvals[ctx.dataIndex] : 0;
                return `L*=${{L.toFixed(1)}}, a*=${{ctx.parsed.x.toFixed(1)}}, b*=${{ctx.parsed.y.toFixed(1)}}`;
            }};
        }}
        
        chart.update();
        return;
    }}
    
    const filteredNames = new Set(filteredDataExisting.map(row => row.hdri_name));
    
    // Create two datasets: one for grayed out, one for selected
    const grayedPoints = [];
    const grayedColors = [];
    const grayedRadius = [];
    const selectedPoints = [];
    const selectedColors = [];
    const selectedRadius = [];
    
    // Separate points based on filtering using the names array
    originalData.names.forEach((name, idx) => {{
        if (idx < originalData.points.length) {{
            const point = originalData.points[idx];
            const color = originalData.colors[idx];
            const radius = originalData.Lvals ? Math.max(4, (originalData.Lvals[idx]/100)*8 + 2) : 5;
            
            if (filteredNames.has(name)) {{
                selectedPoints.push(point);
                selectedColors.push(color);
                selectedRadius.push(radius);
            }} else {{
                grayedPoints.push(point);
                // Make color gray/transparent
                grayedColors.push('rgba(200, 200, 200, 0.3)');
                grayedRadius.push(radius * 0.7); // Slightly smaller when grayed
            }}
        }}
    }});
    
    // Update chart with two datasets
    chart.data.datasets = [
        {{
            label: 'Other HDRIs',
            data: grayedPoints,
            pointBackgroundColor: grayedColors,
            pointBorderColor: 'rgba(150, 150, 150, 0.5)',
            pointBorderWidth: 0.5,
            pointRadius: grayedRadius
        }},
        {{
            label: 'Selected HDRIs',
            data: selectedPoints,
            pointBackgroundColor: selectedColors,
            pointBorderColor: '#000',
            pointBorderWidth: 1,
            pointRadius: selectedRadius
        }}
    ];
    
    // Update tooltip callback to show names
    const allNames = [];
    const allLvals = [];
    
    // First add grayed names/Lvals, then selected
    originalData.names.forEach((name, idx) => {{
        if (!filteredNames.has(name)) {{
            allNames.push(name);
            allLvals.push(originalData.Lvals ? originalData.Lvals[idx] : 0);
        }}
    }});
    originalData.names.forEach((name, idx) => {{
        if (filteredNames.has(name)) {{
            allNames.push(name);
            allLvals.push(originalData.Lvals ? originalData.Lvals[idx] : 0);
        }}
    }});
    
    chart.options.plugins.tooltip.callbacks.title = (ctx) => allNames[ctx[0].dataIndex];
    chart.options.plugins.tooltip.callbacks.label = (ctx) => {{
        const L = allLvals[ctx.dataIndex];
        return `L*=${{L.toFixed(1)}}, a*=${{ctx.parsed.x.toFixed(1)}}, b*=${{ctx.parsed.y.toFixed(1)}}`;
    }};
    
    chart.update();
}}

// Update all 3D plots
function updateAll3DCharts() {{
    // Check if Plotly is loaded
    if (typeof Plotly === 'undefined') {{
        console.warn('Plotly not yet loaded, skipping 3D chart updates');
        return;
    }}
    
    // Check if 3D plots are ready
    if (!plotly3dReady) {{
        console.log('3D plots not yet ready, skipping update');
        return;
    }}
    
    // Update RGB 3D plots
    ['globalRgb', 'dcRgb', 'dominantRgb'].forEach(plotType => {{
        update3DRgbPlot(plotType);
    }});
    
    // Update direction 3D plot
    update3DDirectionPlot();
}}

// Update individual 3D RGB plot
function update3DRgbPlot(plotType) {{
    const chartId = plotType + '3dChart';
    const plotDiv = document.getElementById(chartId);
    if (!plotDiv) {{
        console.warn(`3D plot div not found: ${{chartId}}`);
        return;
    }}
    
    // Check if Plotly has rendered the chart
    if (!plotDiv._fullData || plotDiv._fullData.length === 0) {{
        console.warn(`3D plot not yet rendered: ${{chartId}}`);
        return;
    }}
    
    // Store original data if not already stored
    if (!originalVisualizationData.rgb3dData[plotType]) {{
        originalVisualizationData.rgb3dData[plotType] = {{
            x: plotDiv._fullData[0].x.slice(),
            y: plotDiv._fullData[0].y.slice(),
            z: plotDiv._fullData[0].z.slice(),
            text: plotDiv._fullData[0].text.slice(),
            colors: plotDiv._fullData[0].marker.color.slice()
        }};
    }}
    
    // Check if we're showing all data (no filter)
    const isShowingAll = filteredDataExisting.length === allDataExisting.length;
    
    if (isShowingAll) {{
        // Restore original single trace
        const origData = originalVisualizationData.rgb3dData[plotType];
        const trace = {{
            x: origData.x,
            y: origData.y,
            z: origData.z,
            mode: 'markers',
            marker: {{
                size: 8,
                color: origData.colors,
                opacity: 0.8,
                line: {{ color: '#000', width: 1 }}
            }},
            type: 'scatter3d',
            text: origData.text,
            name: 'HDRIs',
            hovertemplate: '<b>%{{text}}</b><br>R: %{{x:.3f}}<br>G: %{{y:.3f}}<br>B: %{{z:.3f}}<br><extra></extra>'
        }};
        
        Plotly.react(chartId, [trace], plotDiv.layout, plotDiv.config);
        return;
    }}
    
    // Get filtered names for quick lookup
    const filteredNames = new Set(filteredDataExisting.map(row => row.hdri_name));
    
    // Get original data from stored
    const originalTrace = originalVisualizationData.rgb3dData[plotType];
    if (!originalTrace) {{
        console.warn(`No original data found for: ${{chartId}}`);
        return;
    }}
    
    // Create two traces: one for grayed out, one for selected
    const grayedTrace = {{
        x: [],
        y: [],
        z: [],
        mode: 'markers',
        marker: {{
            size: 6,
            color: 'rgba(200, 200, 200, 0.3)',
            opacity: 0.3,
            line: {{ color: 'rgba(150, 150, 150, 0.5)', width: 0.5 }}
        }},
        type: 'scatter3d',
        text: [],
        name: 'Other HDRIs',
        hovertemplate: '<b>%{{text}}</b><br>R: %{{x:.3f}}<br>G: %{{y:.3f}}<br>B: %{{z:.3f}}<br><extra></extra>'
    }};
    
    const selectedTrace = {{
        x: [],
        y: [],
        z: [],
        mode: 'markers',
        marker: {{
            size: 8,
            color: [],
            opacity: 0.9,
            line: {{ color: '#000', width: 1 }}
        }},
        type: 'scatter3d',
        text: [],
        name: 'Selected HDRIs',
        hovertemplate: '<b>%{{text}}</b><br>R: %{{x:.3f}}<br>G: %{{y:.3f}}<br>B: %{{z:.3f}}<br><extra></extra>'
    }};
    
    // Separate points based on filtering
    for (let i = 0; i < originalTrace.x.length; i++) {{
        const name = originalTrace.text[i];
        if (filteredNames.has(name)) {{
            selectedTrace.x.push(originalTrace.x[i]);
            selectedTrace.y.push(originalTrace.y[i]);
            selectedTrace.z.push(originalTrace.z[i]);
            selectedTrace.text.push(originalTrace.text[i]);
            selectedTrace.marker.color.push(originalTrace.colors[i]);
        }} else {{
            grayedTrace.x.push(originalTrace.x[i]);
            grayedTrace.y.push(originalTrace.y[i]);
            grayedTrace.z.push(originalTrace.z[i]);
            grayedTrace.text.push(originalTrace.text[i]);
        }}
    }}
    
    // Update plot with both traces
    Plotly.react(chartId, [grayedTrace, selectedTrace], plotDiv.layout, plotDiv.config);
}}

// Update 3D direction plot
function update3DDirectionPlot() {{
    const chartId = 'dominantDirection3dChart';
    const plotDiv = document.getElementById(chartId);
    if (!plotDiv) return;
    
    // Check if Plotly has rendered the chart
    if (!plotDiv._fullData || plotDiv._fullData.length === 0) {{
        console.warn(`3D direction plot not yet rendered`);
        return;
    }}
    
    // Store original data if not already stored
    if (!originalVisualizationData.direction3dData) {{
        originalVisualizationData.direction3dData = {{
            x: plotDiv._fullData[0].x.slice(),
            y: plotDiv._fullData[0].y.slice(),
            z: plotDiv._fullData[0].z.slice(),
            text: plotDiv._fullData[0].text.slice(),
            colors: plotDiv._fullData[0].marker.color.slice()
        }};
    }}
    
    // Check if we're showing all data (no filter)
    const isShowingAll = filteredDataExisting.length === allDataExisting.length;
    
    if (isShowingAll) {{
        // Restore original single trace
        const origData = originalVisualizationData.direction3dData;
        const trace = {{
            x: origData.x,
            y: origData.y,
            z: origData.z,
            mode: 'markers',
            marker: {{
                size: 8,
                color: origData.colors,
                opacity: 0.8,
                line: {{ color: '#000', width: 1 }}
            }},
            type: 'scatter3d',
            text: origData.text,
            name: 'HDRIs',
            hovertemplate: '<b>%{{text}}</b><br>X: %{{x:.3f}}<br>Y: %{{y:.3f}}<br>Z: %{{z:.3f}}<br><extra></extra>'
        }};
        
        Plotly.react(chartId, [trace], plotDiv.layout, plotDiv.config);
        return;
    }}
    
    // Get filtered names for quick lookup
    const filteredNames = new Set(filteredDataExisting.map(row => row.hdri_name));
    
    // Get original data from stored
    const originalTrace = originalVisualizationData.direction3dData;
    if (!originalTrace) {{
        console.warn(`No original data found for direction plot`);
        return;
    }}
    
    // Create two traces similar to RGB plots
    const grayedTrace = {{
        x: [],
        y: [],
        z: [],
        mode: 'markers',
        marker: {{
            size: 6,
            color: 'rgba(200, 200, 200, 0.3)',
            opacity: 0.3,
            line: {{ color: 'rgba(150, 150, 150, 0.5)', width: 0.5 }}
        }},
        type: 'scatter3d',
        text: [],
        name: 'Other HDRIs',
        hovertemplate: '<b>%{{text}}</b><br>X: %{{x:.3f}}<br>Y: %{{y:.3f}}<br>Z: %{{z:.3f}}<br><extra></extra>'
    }};
    
    const selectedTrace = {{
        x: [],
        y: [],
        z: [],
        mode: 'markers',
        marker: {{
            size: 10,
            color: [],
            opacity: 0.9,
            line: {{ color: '#000', width: 1 }}
        }},
        type: 'scatter3d',
        text: [],
        name: 'Selected HDRIs',
        hovertemplate: '<b>%{{text}}</b><br>X: %{{x:.3f}}<br>Y: %{{y:.3f}}<br>Z: %{{z:.3f}}<br><extra></extra>'
    }};
    
    // Separate points based on filtering
    for (let i = 0; i < originalTrace.x.length; i++) {{
        const name = originalTrace.text[i];
        if (filteredNames.has(name)) {{
            selectedTrace.x.push(originalTrace.x[i]);
            selectedTrace.y.push(originalTrace.y[i]);
            selectedTrace.z.push(originalTrace.z[i]);
            selectedTrace.text.push(originalTrace.text[i]);
            selectedTrace.marker.color.push(originalTrace.colors[i]);
        }} else {{
            grayedTrace.x.push(originalTrace.x[i]);
            grayedTrace.y.push(originalTrace.y[i]);
            grayedTrace.z.push(originalTrace.z[i]);
            grayedTrace.text.push(originalTrace.text[i]);
        }}
    }}
    
    // Update plot with both traces
    Plotly.react(chartId, [grayedTrace, selectedTrace], plotDiv.layout, plotDiv.config);
}}

function resetAllFiltersExisting() {{
    console.log('Resetting all filters...');
    
    for (const [filterKey, config] of Object.entries(filterConfigExisting)) {{
        document.getElementById(filterKey + '_min').value = config.default_min;
        document.getElementById(filterKey + '_max').value = config.default_max;
        document.getElementById(filterKey + '_min_val').value = config.default_min.toFixed(3);
        document.getElementById(filterKey + '_max_val').value = config.default_max.toFixed(3);
    }}
    
    applyAllFiltersExisting();
}}

function exportFilteredDataExisting() {{
    console.log('Exporting filtered data...');
    
    const csvContent = 'HDRI_Name,Global_Intensity,Global_R,Global_G,Global_B\\n' + 
                      filteredDataExisting.map(row => 
                          row.hdri_name + ',' + row.global_intensity + ',' + row.global_r + ',' + row.global_g + ',' + row.global_b
                      ).join('\\n');
    
    const blob = new Blob([csvContent], {{ type: 'text/csv' }});
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'filtered_hdris.csv';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    
    alert('Exported ' + filteredDataExisting.length + ' filtered HDRIs to CSV');
}}
'''


def calculate_pandas_stats(df: pd.DataFrame) -> Dict:
    """Calculate comprehensive statistics using pandas."""
    stats = {}
    
    # Basic statistics
    stats['total_hdris'] = len(df)
    
    # Global intensity statistics
    stats['global_intensity'] = {
        'mean': df['global_intensity'].mean(),
        'std': df['global_intensity'].std(),
        'min': df['global_intensity'].min(),
        'max': df['global_intensity'].max(),
        'median': df['global_intensity'].median(),
        'q25': df['global_intensity'].quantile(0.25),
        'q75': df['global_intensity'].quantile(0.75)
    }
    
    # Color statistics for each RGB channel
    color_metrics = ['global', 'dc', 'dominant', 'area']
    for metric in color_metrics:
        r_col, g_col, b_col = f'{metric}_r', f'{metric}_g', f'{metric}_b'
        
        stats[f'{metric}_color'] = {
            'r': {
                'mean': df[r_col].mean(),
                'std': df[r_col].std(),
                'min': df[r_col].min(),
                'max': df[r_col].max()
            },
            'g': {
                'mean': df[g_col].mean(),
                'std': df[g_col].std(),
                'min': df[g_col].min(),
                'max': df[g_col].max()
            },
            'b': {
                'mean': df[b_col].mean(),
                'std': df[b_col].std(),
                'min': df[b_col].min(),
                'max': df[b_col].max()
            }
        }
        
        # Overall brightness for this color metric
        brightness = (df[r_col] + df[g_col] + df[b_col]) / 3
        stats[f'{metric}_brightness'] = {
            'mean': brightness.mean(),
            'std': brightness.std(),
            'min': brightness.min(),
            'max': brightness.max()
        }
    
    # Color correlations
    stats['correlations'] = {
        'global_vs_dc': df[['global_r', 'global_g', 'global_b']].corrwith(
            df[['dc_r', 'dc_g', 'dc_b']]).mean(),
        'global_vs_dominant': df[['global_r', 'global_g', 'global_b']].corrwith(
            df[['dominant_r', 'dominant_g', 'dominant_b']]).mean(),
        'intensity_vs_brightness': df['global_intensity'].corr(
            (df['global_r'] + df['global_g'] + df['global_b']) / 3)
    }
    
    # Top and bottom HDRIs by different metrics
    stats['extremes'] = {
        'brightest_intensity': {
            'name': df.loc[df['global_intensity'].idxmax(), 'hdri_name'],
            'value': df['global_intensity'].max()
        },
        'darkest_intensity': {
            'name': df.loc[df['global_intensity'].idxmin(), 'hdri_name'],
            'value': df['global_intensity'].min()
        },
        'most_colorful': {
            'name': df.loc[((df['global_r'] - df['global_r'].mean())**2 + 
                          (df['global_g'] - df['global_g'].mean())**2 + 
                          (df['global_b'] - df['global_b'].mean())**2).idxmax(), 'hdri_name']
        }
    }
    
    return stats


def create_pandas_histogram(series: pd.Series, bins: int = 20) -> Tuple[List[float], List[int]]:
    """Create histogram data using pandas."""
    counts, bin_edges = np.histogram(series.dropna(), bins=bins)
    bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
    return bin_centers, counts.tolist()


def create_pandas_rgb_histogram(df: pd.DataFrame, rgb_cols: List[str], bins: int = 20) -> Dict[str, Tuple[List[float], List[int]]]:
    """Create RGB histogram data using pandas."""
    result = {}
    colors = ['r', 'g', 'b']
    
    for i, color in enumerate(colors):
        col_name = rgb_cols[i]
        counts, bin_edges = np.histogram(df[col_name].dropna(), bins=bins)
        bin_centers = [(bin_edges[j] + bin_edges[j+1]) / 2 for j in range(len(bin_edges)-1)]
        result[color] = (bin_centers, counts.tolist())
    
    return result


def generate_aggregate_statistics_html(experiment_dir: Path) -> str:
    """Generate an HTML page with aggregate statistics for an experiment."""
    print(f"Starting aggregate statistics generation for: {experiment_dir}")
    
    metrics_data = collect_experiment_metrics(experiment_dir)
    print(f"Raw metrics data collected:")
    for key, values in metrics_data.items():
        print(f"  {key}: {len(values) if isinstance(values, list) else 'N/A'} items")
    
    if not metrics_data['hdri_names']:
        print(f"ERROR: No HDRI data found in experiment directory: {experiment_dir}")
        raise ValueError(f"No HDRI data found in experiment directory: {experiment_dir}")

    experiment_name = experiment_dir.name
    num_hdris = len(metrics_data['hdri_names'])
    print(f"Processing {num_hdris} HDRIs for experiment: {experiment_name}")

    # Create pandas DataFrame for easier analysis
    df = create_dataframe_from_metrics(metrics_data)
    print(f"Created DataFrame with shape: {df.shape}")

    # Use all data for visualization (no sampling)
    print(f"Using all {num_hdris} HDRIs for visualization.")
    viz_data = metrics_data
    viz_df = df

    # Visualizations container
    visualizations: Dict[str, Dict] = {}

    print("Creating visualizations...")
    
    # Intensity histograms using pandas (can handle full dataset)
    print(f"  Creating intensity histograms from {len(df)} intensity values...")
    visualizations['global_intensity'] = create_pandas_histogram(df['global_intensity'])
    visualizations['area_intensity'] = create_pandas_rgb_histogram(df, ['area_r', 'area_g', 'area_b'])
    
    print(f"  Global intensity histogram: {len(visualizations['global_intensity'][0]) if visualizations['global_intensity'][0] else 0} bins")
    print(f"  Area intensity histogram: {len(visualizations['area_intensity']['r'][0]) if visualizations['area_intensity']['r'][0] else 0} bins")

    # Perceptual color scatter plots (CIELAB a*b* maps) - use sampled data
    print(f"  Creating CIELAB visualizations...")
    
    visualizations['global_color_lab'] = create_lab_scatter_data(viz_data['global_color'], viz_data['hdri_names'])
    visualizations['dc_color_lab'] = create_lab_scatter_data(viz_data['dc_color'], viz_data['hdri_names'])
    visualizations['dominant_color_lab'] = create_lab_scatter_data(viz_data['dominant_color'], viz_data['hdri_names'])
    
    print(f"  Global color LAB: {len(visualizations['global_color_lab']['ab'])} points")
    print(f"  DC color LAB: {len(visualizations['dc_color_lab']['ab'])} points")
    print(f"  Dominant color LAB: {len(visualizations['dominant_color_lab']['ab'])} points")

    # 3D RGB scatter plots - use sampled data
    print(f"  Creating 3D RGB visualizations...")
    
    visualizations['global_color_3d'] = create_3d_rgb_scatter_data(viz_data['global_color'], viz_data['hdri_names'])
    visualizations['dc_color_3d'] = create_3d_rgb_scatter_data(viz_data['dc_color'], viz_data['hdri_names'])
    visualizations['dominant_color_3d'] = create_3d_rgb_scatter_data(viz_data['dominant_color'], viz_data['hdri_names'])
    
    print(f"  Global color 3D: {len(visualizations['global_color_3d']['xyz'])} points")
    print(f"  DC color 3D: {len(visualizations['dc_color_3d']['xyz'])} points")
    print(f"  Dominant color 3D: {len(visualizations['dominant_color_3d']['xyz'])} points")

    # 3D Direction scatter plot - use sampled data
    print(f"  Creating 3D dominant direction visualization...")
    
    visualizations['dominant_direction_3d'] = create_3d_direction_scatter_data(
        viz_data['dominant_direction'], 
        viz_data['hdri_names'], 
        viz_data['dominant_color']
    )
    
    print(f"  Dominant direction 3D: {len(visualizations['dominant_direction_3d']['xyz'])} points")

    # Summary statistics using pandas - use full dataset for statistics
    stats = calculate_pandas_stats(df)

    # Create filter configuration for interactive filtering
    filter_config = create_filter_configuration_for_existing(df, stats)
    print(f"  Created filter configuration with {len(filter_config)} filters")

    html_content = _generate_aggregate_html_template(
        experiment_name,
        num_hdris,
        visualizations,
        stats,
        metrics_data['hdri_names'],
        df,  # Pass DataFrame for filtering
        filter_config  # Pass filter configuration
    )

    html_path = experiment_dir / f"{experiment_name}_aggregate_statistics.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Generated aggregate statistics HTML: {html_path}")
    return str(html_path)


# ==============================
# Stats (extended with LAB)
# ==============================

def _channel_stats(values: np.ndarray) -> Dict[str, float]:
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values)),
    }


def calculate_summary_stats(metrics_data: Dict[str, List]) -> Dict:
    stats: Dict[str, Dict] = {}

    # Global intensity stats
    if metrics_data['global_intensity']:
        values = np.array(metrics_data['global_intensity'])
        stats['global_intensity'] = _channel_stats(values)

    # RGB stats for each color metric
    for metric_name in ['global_color', 'dc_color', 'dominant_color', 'area_intensity']:
        if metrics_data[metric_name]:
            rgb_array = np.array(metrics_data[metric_name])
            stats[metric_name] = {}
            for i, channel in enumerate(['r', 'g', 'b']):
                stats[metric_name][channel] = _channel_stats(rgb_array[:, i])

    # LAB stats (L*, a*, b*) for the three actual color metrics
    for metric_name in ['global_color', 'dc_color', 'dominant_color']:
        if metrics_data[metric_name]:
            lab = srgb_array_to_lab(metrics_data[metric_name])
            stats[metric_name + '_lab'] = {
                'L': _channel_stats(lab[:, 0]),
                'a': _channel_stats(lab[:, 1]),
                'b': _channel_stats(lab[:, 2]),
            }

    return stats


# ==============================
# HTML template (adds LAB a*b* canvases + slider + JS conversions)
# ==============================

def _generate_aggregate_html_template(
    experiment_name: str,
    num_hdris: int,
    visualizations: Dict,
    stats: Dict,
    hdri_names: List[str],
    df: Optional[pd.DataFrame] = None,
    filter_config: Optional[Dict] = None
) -> str:
    def to_js(x):
        return str(x).replace("'", '"')

    # Prepare filter data and controls if filtering is enabled
    filter_controls_html = ""
    filter_js = ""
    df_json = "{}"
    
    if df is not None and filter_config is not None:
        # Convert DataFrame to JavaScript object for filtering
        df_json = df.to_json(orient='records')
        
        # Generate filter controls HTML
        filter_controls_html = _generate_filter_controls_html_for_existing(filter_config)
        
        # Generate filter JavaScript
        filter_js = _generate_filter_javascript_for_existing(df_json, filter_config, hdri_names)

    charts_js = ""

    # ------------------ Global Intensity Histogram ------------------
    if visualizations['global_intensity'][0]:
        charts_js += f"""
        // Global Intensity Chart
        var globalIntensityCtx = document.getElementById('globalIntensityChart').getContext('2d');
        globalIntensityChart = new Chart(globalIntensityCtx, {{
            type: 'bar',
            data: {{
                labels: {to_js([f"{x:.3f}" for x in visualizations['global_intensity'][0]])},
                datasets: [{{
                    label: 'Count',
                    data: {to_js(visualizations['global_intensity'][1])},
                    backgroundColor: 'rgba(54, 162, 235, 0.6)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{ display: true, text: 'Global Intensity Distribution' }}
                }},
                scales: {{
                    y: {{ beginAtZero: true, title: {{ display: true, text: 'Count' }} }},
                    x: {{ 
                        title: {{ display: true, text: 'Global Intensity' }},
                        // Keep x-axis fixed during updates
                        beforeUpdate: function(scale) {{
                            if (scale.chart._xAxisFixed) {{
                                scale.options.min = scale._userMin;
                                scale.options.max = scale._userMax;
                            }}
                        }},
                        afterFit: function(scale) {{
                            if (!scale.chart._xAxisFixed) {{
                                scale._userMin = scale.min;
                                scale._userMax = scale.max;
                                scale.chart._xAxisFixed = true;
                            }}
                        }}
                    }}
                }}
            }}
        }});
        allCharts.intensity = globalIntensityChart;
        """

    # ------------------ Area Intensity RGB Histogram ------------------
    if visualizations['area_intensity']['r'][0]:
        charts_js += f"""
        // Area Intensity RGB Histogram
        var areaIntensityCtx = document.getElementById('areaIntensityChart').getContext('2d');
        allCharts.areaIntensity = new Chart(areaIntensityCtx, {{
            type: 'bar',
            data: {{
                labels: {to_js([f"{x:.2f}" for x in visualizations['area_intensity']['r'][0]])},
                datasets: [
                    {{ label: 'Red',   data: {to_js(visualizations['area_intensity']['r'][1])}, backgroundColor: 'rgba(255,99,132,0.6)', borderColor: 'rgba(255,99,132,1)', borderWidth: 1 }},
                    {{ label: 'Green', data: {to_js(visualizations['area_intensity']['g'][1])}, backgroundColor: 'rgba(75,192,192,0.6)', borderColor: 'rgba(75,192,192,1)', borderWidth: 1 }},
                    {{ label: 'Blue',  data: {to_js(visualizations['area_intensity']['b'][1])}, backgroundColor: 'rgba(54,162,235,0.6)', borderColor: 'rgba(54,162,235,1)', borderWidth: 1 }}
                ]
            }},
            options: {{
                responsive: true,
                plugins: {{ title: {{ display: true, text: 'Dominant Area Intensity Distribution (RGB)' }} }},
                scales: {{
                    y: {{ beginAtZero: true, title: {{ display: true, text: 'Count' }} }},
                    x: {{ title: {{ display: true, text: 'Intensity Value' }} }}
                }}
            }}
        }});
        """

    # HSV Color Scatter Charts removed for simplicity

    # ------------------ CIELAB a*b* scatters (perceptual) ------------------
    def lab_scatter(metric_key: str, metric_title: str, chart_id: str, bg_id: str) -> str:
        viz = visualizations[metric_key]
        if not viz['ab'] or len(viz['ab']) == 0:
            print(f"Warning: No CIELAB data for {metric_key}, creating placeholder chart")
            # Create a placeholder chart with minimal setup
            return f"""
            (function() {{
                function initPlaceholderChart() {{
                    console.log('Creating placeholder CIELAB chart for {metric_title}');
                    var bgCanvas = document.getElementById('{bg_id}');
                    if (bgCanvas) {{
                        console.log('Drawing background for placeholder chart {bg_id}');
                        drawLabABBackground(bgCanvas, currentLabL);
                    }} else {{
                        console.error('Background canvas not found for placeholder: {bg_id}');
                    }}
                    var ctx = document.getElementById('{chart_id}');
                    if (ctx) {{
                        console.log('Creating placeholder chart {chart_id}');
                        ctx = ctx.getContext('2d');
                        new Chart(ctx, {{
                            type: 'scatter',
                            data: {{ datasets: [{{
                                label: 'No Data',
                                data: [],
                                pointBackgroundColor: [],
                                pointBorderColor: '#000',
                                pointBorderWidth: 1
                            }}] }},
                            options: {{
                                responsive: true,
                                plugins: {{
                                    title: {{ display: true, text: '{metric_title} — CIELAB a*b* (no data)' }},
                                    legend: {{ display: false }}
                                }},
                                scales: {{
                                    x: {{ min: -50, max: 50, title: {{ display: true, text: 'a*  (green  ↔  red)' }} }},
                                    y: {{ min: -50, max: 50, title: {{ display: true, text: 'b*  (blue   ↔  yellow)' }} }}
                                }},
                                aspectRatio: 1
                            }}
                        }});
                        console.log('Placeholder chart {metric_title} created');
                    }} else {{
                        console.error('Chart canvas not found for placeholder: {chart_id}');
                    }}
                }}
                
                // Try to initialize immediately, or wait for DOM ready
                if (document.readyState === 'loading') {{
                    document.addEventListener('DOMContentLoaded', initPlaceholderChart);
                }} else {{
                    initPlaceholderChart();
                }}
            }})();
            """
        points = [{'x': float(a), 'y': float(b)} for a, b in viz['ab']]
        return f"""
        (function() {{
            function initChart() {{
                console.log('Creating CIELAB chart for {metric_title}');
                console.log('Data points:', {len(points)});
                
                var bgCanvas = document.getElementById('{bg_id}');
                if (bgCanvas) {{
                    console.log('Found background canvas {bg_id}, drawing background...');
                    drawLabABBackground(bgCanvas, currentLabL);
                }} else {{
                    console.error('Background canvas not found: {bg_id}');
                }}
                
                var ctx = document.getElementById('{chart_id}');
                if (!ctx) {{
                    console.error('Chart canvas not found: {chart_id}');
                    return;
                }}
                
                console.log('Found chart canvas {chart_id}, getting 2D context...');
                ctx = ctx.getContext('2d');
                
                var colors = {to_js(viz['colors'])};
                var names = {to_js(viz['names'])};
                var Lvals = {to_js(viz['L'])};
                console.log('Creating chart with', colors.length, 'colors and', names.length, 'names');
                
                // Determine chart type from ID
                var chartType = '{chart_id}' === 'globalLabAbChart' ? 'global' : 
                               '{chart_id}' === 'dcLabAbChart' ? 'dc' : 'dominant';
                
                // Store original data before creating chart
                if (!originalVisualizationData.labData[chartType]) {{
                    originalVisualizationData.labData[chartType] = {{
                        points: {to_js(points)},
                        colors: colors.slice(),
                        names: names.slice(),
                        Lvals: Lvals.slice()
                    }};
                }}
                
                var chart = new Chart(ctx, {{
                type: 'scatter',
                data: {{ datasets: [{{
                    label: '{metric_title}',
                    data: {to_js(points)},
                    pointBackgroundColor: colors,
                    pointBorderColor: '#000',
                    pointBorderWidth: 1,
                    pointRadius: Lvals.map(L => Math.max(4, (L/100)*8 + 2))
                }}] }},
                options: {{
                    responsive: true,
                    plugins: {{
                        title: {{ display: true, text: '{metric_title} — CIELAB a*b* (perceptual)' }},
                        legend: {{ display: false }},
                        tooltip: {{ callbacks: {{
                            title: (ctx) => names[ctx[0].dataIndex],
                            label: (ctx) => `L*=${{Lvals[ctx.dataIndex].toFixed(1)}}, a*=${{ctx.parsed.x.toFixed(1)}}, b*=${{ctx.parsed.y.toFixed(1)}}`
                        }} }}
                    }},
                    scales: {{
                        x: {{ min: -50, max: 50, title: {{ display: true, text: 'a*  (green  ↔  red)' }} }},
                        y: {{ min: -50, max: 50, title: {{ display: true, text: 'b*  (blue   ↔  yellow)' }} }}
                    }},
                    aspectRatio: 1
                }}
                            }});
                
                // Store chart reference
                if ('{chart_id}' === 'globalLabAbChart') {{
                    allCharts.labCharts.global = chart;
                }}
                else if ('{chart_id}' === 'dcLabAbChart') {{
                    allCharts.labCharts.dc = chart;
                }}
                else if ('{chart_id}' === 'dominantLabAbChart') {{
                    allCharts.labCharts.dominant = chart;
                }}
                
                // Chart created successfully
                console.log('Chart {metric_title} created successfully');
            }}
            
            // Try to initialize immediately, or wait for DOM ready
            if (document.readyState === 'loading') {{
                document.addEventListener('DOMContentLoaded', initChart);
            }} else {{
                initChart();
            }}
        }})();
    """

    charts_js += lab_scatter('global_color_lab', 'Global Color', 'globalLabAbChart', 'globalLabAbBg')
    charts_js += lab_scatter('dc_color_lab', 'L=0 Color (DC Term)', 'dcLabAbChart', 'dcLabAbBg')
    charts_js += lab_scatter('dominant_color_lab', 'Dominant Color', 'dominantLabAbChart', 'dominantLabAbBg')

    # ------------------ 3D RGB scatter plots ------------------
    def create_3d_rgb_plot(metric_key: str, metric_title: str, chart_id: str) -> str:
        viz = visualizations[metric_key]
        if not viz['xyz'] or len(viz['xyz']) == 0:
            return f"""
            // No data for {metric_title} 3D plot
            document.addEventListener('DOMContentLoaded', function() {{
                document.getElementById('{chart_id}').innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#666;">No data available</div>';
            }});
            """
        
        # Prepare data for Plotly
        x_vals = [point[0] for point in viz['xyz']]
        y_vals = [point[1] for point in viz['xyz']]
        z_vals = [point[2] for point in viz['xyz']]
        
        return f"""
        document.addEventListener('DOMContentLoaded', function() {{
            var trace = {{
                x: {to_js(x_vals)},
                y: {to_js(y_vals)},
                z: {to_js(z_vals)},
                mode: 'markers',
                marker: {{
                    size: 8,
                    color: {to_js(viz['colors'])},
                    opacity: 0.8,
                    line: {{
                        color: '#000',
                        width: 1
                    }}
                }},
                type: 'scatter3d',
                text: {to_js(viz['names'])},
                hovertemplate: '<b>%{{text}}</b><br>' +
                              'R: %{{x:.3f}}<br>' +
                              'G: %{{y:.3f}}<br>' +
                              'B: %{{z:.3f}}<br>' +
                              '<extra></extra>'
            }};
            
            var layout = {{
                title: {{
                    font: {{ size: 14, color: '#000' }}
                }},
                scene: {{
                    xaxis: {{ title: 'Red (0-1)', range: [0, 1], gridcolor: '#ddd' }},
                    yaxis: {{ title: 'Green (0-1)', range: [0, 1], gridcolor: '#ddd' }},
                    zaxis: {{ title: 'Blue (0-1)', range: [0, 1], gridcolor: '#ddd' }},
                    bgcolor: '#e8f4f8',
                    camera: {{
                        eye: {{ x: 1.5, y: 1.5, z: 1.5 }}
                    }}
                }},
                margin: {{ l: 0, r: 0, t: 30, b: 0 }},
                paper_bgcolor: '#f9f9f9',
                plot_bgcolor: '#f9f9f9',
                showlegend: false
            }};
            
            var config = {{
                displayModeBar: true,
                modeBarButtonsToRemove: ['pan2d', 'select2d', 'lasso2d', 'autoScale2d'],
                displaylogo: false,
                responsive: true
            }};
            
            Plotly.newPlot('{chart_id}', [trace], layout, config);
        }});
        """

    charts_js += create_3d_rgb_plot('global_color_3d', 'Global Color', 'globalRgb3dChart')
    charts_js += create_3d_rgb_plot('dc_color_3d', 'L=0 Color (DC Term)', 'dcRgb3dChart')
    charts_js += create_3d_rgb_plot('dominant_color_3d', 'Dominant Color', 'dominantRgb3dChart')

    # ------------------ 3D Direction scatter plot ------------------
    def create_3d_direction_plot(metric_key: str, metric_title: str, chart_id: str) -> str:
        viz = visualizations[metric_key]
        if not viz['xyz'] or len(viz['xyz']) == 0:
            return f"""
            // No data for {metric_title} 3D direction plot
            document.addEventListener('DOMContentLoaded', function() {{
                document.getElementById('{chart_id}').innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#666;">No data available</div>';
            }});
            """
        
        # Prepare data for Plotly
        x_vals = [point[0] for point in viz['xyz']]
        y_vals = [point[1] for point in viz['xyz']]
        z_vals = [point[2] for point in viz['xyz']]
        
        return f"""
        document.addEventListener('DOMContentLoaded', function() {{
            var trace = {{
                x: {to_js(x_vals)},
                y: {to_js(y_vals)},
                z: {to_js(z_vals)},
                mode: 'markers',
                marker: {{
                    size: 8,
                    color: {to_js(viz['colors'])},
                    opacity: 0.8,
                    line: {{
                        color: '#000',
                        width: 1
                    }}
                }},
                type: 'scatter3d',
                text: {to_js(viz['names'])},
                hovertemplate: '<b>%{{text}}</b><br>' +
                              'X: %{{x:.3f}}<br>' +
                              'Y: %{{y:.3f}}<br>' +
                              'Z: %{{z:.3f}}<br>' +
                              '<extra></extra>'
            }};
            
            var layout = {{
                title: {{
                    font: {{ size: 14, color: '#000' }}
                }},
                scene: {{
                    xaxis: {{ title: 'X Direction', range: [-1, 1], gridcolor: '#ddd' }},
                    yaxis: {{ title: 'Y Direction', range: [-1, 1], gridcolor: '#ddd' }},
                    zaxis: {{ title: 'Z Direction', range: [-1, 1], gridcolor: '#ddd' }},
                    bgcolor: '#e8f4f8',
                    camera: {{
                        eye: {{ x: 1.5, y: 1.5, z: 1.5 }}
                    }}
                }},
                margin: {{ l: 0, r: 0, t: 30, b: 0 }},
                paper_bgcolor: '#f9f9f9',
                plot_bgcolor: '#f9f9f9',
                showlegend: false
            }};
            
            var config = {{
                displayModeBar: true,
                modeBarButtonsToRemove: ['pan2d', 'select2d', 'lasso2d', 'autoScale2d'],
                displaylogo: false,
                responsive: true
            }};
            
            Plotly.newPlot('{chart_id}', [trace], layout, config);
        }});
        """

    charts_js += create_3d_direction_plot('dominant_direction_3d', 'Dominant Direction', 'dominantDirection3dChart')

    # ------------------ JS helpers for LAB background rendering ------------------
    charts_js += """
    // --- CIELAB helpers (D65) for drawing the a*b* background ---
    const Xn = 0.95047, Yn = 1.0, Zn = 1.08883;
    const M_XYZ_to_RGB = [
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ];

    function f_inv(t){
        const eps = 216/24389; // 0.008856
        const kap = 24389/27;  // 903.296
        const t3 = t*t*t;
        return (t3 > eps) ? t3 : (116*t - 16)/kap;
    }

    function labToXYZ(L, a, b){
        const fy = (L + 16)/116;
        const fx = fy + (a/500);
        const fz = fy - (b/200);
        const xr = f_inv(fx);
        const yr = f_inv(fy);
        const zr = f_inv(fz);
        return [xr*Xn, yr*Yn, zr*Zn];
    }

    function xyzToLinearRGB(X, Y, Z){
        const r = M_XYZ_to_RGB[0][0]*X + M_XYZ_to_RGB[0][1]*Y + M_XYZ_to_RGB[0][2]*Z;
        const g = M_XYZ_to_RGB[1][0]*X + M_XYZ_to_RGB[1][1]*Y + M_XYZ_to_RGB[1][2]*Z;
        const b = M_XYZ_to_RGB[2][0]*X + M_XYZ_to_RGB[2][1]*Y + M_XYZ_to_RGB[2][2]*Z;
        return [r, g, b];
    }

    function linearToSRGB(c){
        const a = 0.055;
        return (c <= 0.0031308) ? (12.92*c) : (1.055*Math.pow(c, 1/2.4) - 0.055);
    }

    let currentLabL = 65; // default L* slice for background

    function drawLabABBackground(canvas, L){
        const size = 350; // px (matches CSS)
        canvas.width = size; canvas.height = size;
        const ctx = canvas.getContext('2d');
        const imageData = ctx.createImageData(size, size);
        let i = 0;
        for(let y=0; y<size; y++){
            const bVal = ( (size-1- y) / (size-1) ) * 100 - 50; // map to [-50,50)
            for(let x=0; x<size; x++){
                const aVal = ( x / (size-1) ) * 100 - 50;
                const [X, Y, Z] = labToXYZ(L, aVal, bVal);
                let [lr, lg, lb] = xyzToLinearRGB(X, Y, Z);
                // clamp negative linear before companding
                lr = Math.max(0, lr); lg = Math.max(0, lg); lb = Math.max(0, lb);
                let R = linearToSRGB(lr), G = linearToSRGB(lg), B = linearToSRGB(lb);
                // clamp to displayable range
                R = Math.min(1, Math.max(0, R));
                G = Math.min(1, Math.max(0, G));
                B = Math.min(1, Math.max(0, B));
                imageData.data[i++] = Math.round(R*255);
                imageData.data[i++] = Math.round(G*255);
                imageData.data[i++] = Math.round(B*255);
                imageData.data[i++] = 255;
            }
        }
        ctx.putImageData(imageData, 0, 0);
    }

    function updateLabBackgrounds(L){
        currentLabL = L;
        ['globalLabAbBg','dcLabAbBg','dominantLabAbBg'].forEach(id => {
            const el = document.getElementById(id);
            if (el) drawLabABBackground(el, L);
        });
    }

    // Initialize backgrounds and search when page loads
    document.addEventListener('DOMContentLoaded', function() {
        console.log('DOM loaded, initializing LAB backgrounds...');
        updateLabBackgrounds(currentLabL);
        initializeHDRISearch();
    });

    // HDRI Search functionality
    function initializeHDRISearch() {
        const searchInput = document.getElementById('hdriSearch');
        const linksContainer = document.getElementById('individualLinksContainer');
        const allLinks = linksContainer.querySelectorAll('.individual-link');

        searchInput.addEventListener('input', function(e) {
            const searchTerm = e.target.value.toLowerCase().trim();

            allLinks.forEach(link => {
                const hdriName = link.getAttribute('data-hdri-name').toLowerCase();
                if (hdriName.includes(searchTerm)) {
                    link.style.display = 'inline-block';
                } else {
                    link.style.display = 'none';
                }
            });
        });

        // Clear search on Escape key
        searchInput.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                searchInput.value = '';
                allLinks.forEach(link => {
                    link.style.display = 'inline-block';
                });
            }
        });
    }
    """

    # ------------------ Stats tables ------------------
    stats_tables = ""

    if 'global_intensity' in stats:
        s = stats['global_intensity']
        stats_tables += f"""
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
        """

    # Only area intensity RGB stats (no color statistics)
    if 'area_intensity' in stats:
        s = stats['area_intensity']
        stats_tables += f"""
        <div class="stats-table">
            <h4>Dominant Area Intensity Statistics (RGB channels)</h4>
            <table>
                <tr><th></th><th>Red</th><th>Green</th><th>Blue</th></tr>
                <tr><td>Mean:</td><td>{s['r']['mean']:.1f}</td><td>{s['g']['mean']:.1f}</td><td>{s['b']['mean']:.1f}</td></tr>
                <tr><td>Std Dev:</td><td>{s['r']['std']:.1f}</td><td>{s['g']['std']:.1f}</td><td>{s['b']['std']:.1f}</td></tr>
                <tr><td>Min:</td><td>{s['r']['min']:.1f}</td><td>{s['g']['min']:.1f}</td><td>{s['b']['min']:.1f}</td></tr>
                <tr><td>Max:</td><td>{s['r']['max']:.1f}</td><td>{s['g']['max']:.1f}</td><td>{s['b']['max']:.1f}</td></tr>
                <tr><td>Median:</td><td>{s['r']['median']:.1f}</td><td>{s['g']['median']:.1f}</td><td>{s['b']['median']:.1f}</td></tr>
            </table>
        </div>
        """

    # ------------------ HTML ------------------
    return f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"UTF-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
<title>Aggregate Statistics — {experiment_name}</title>
<script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>
<script src=\"https://cdn.plot.ly/plotly-latest.min.js\"></script>
<style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{ font-family: 'Courier New', monospace; line-height:1.4; color:#000; background:#c0c0c0; }}
    .container {{ max-width: 1400px; margin:0 auto; padding:10px; }}
    .header {{ text-align:center; color:#000; margin-bottom:20px; background:#808080; padding:10px; border:2px inset #c0c0c0; }}
    .header h1 {{ font-size:2rem; margin-bottom:5px; font-weight:bold; }}
    .header p {{ font-size:1.1rem; margin:5px 0; }}
    .content {{ display:grid; grid-template-columns:1fr; gap:20px; }}
    
    /* Visual chart filter styles */
    .chart-filter-container {{ position:relative; }}
    .chart-controls-below {{ margin-top:10px; background:rgba(249,249,249,0.95); padding:10px; border:1px solid #ddd; border-radius:5px; }}
    .range-selector-overlay {{ position:absolute; bottom:50px; left:50px; right:50px; z-index:10; }}
    .range-slider-visual {{ width:100%; height:30px; background:linear-gradient(to right, #e0e0e0 0%, #4CAF50 50%, #e0e0e0 100%); border-radius:15px; position:relative; cursor:pointer; }}
    .range-handle {{ position:absolute; top:-5px; width:20px; height:40px; background:#2196F3; border:2px solid #fff; border-radius:10px; cursor:grab; box-shadow:0 2px 5px rgba(0,0,0,0.3); }}
    .range-handle:active {{ cursor:grabbing; }}
    .range-highlight {{ position:absolute; top:0; height:100%; background:rgba(76,175,80,0.3); border:2px solid #4CAF50; border-radius:15px; pointer-events:none; }}
    .range-label {{ position:absolute; bottom:-25px; font-size:12px; font-weight:bold; color:#333; white-space:nowrap; }}
    
    /* Brush selection styles */
    .chart-brush-overlay {{ position:absolute; top:0; left:0; right:0; bottom:0; z-index:5; pointer-events:auto; cursor:crosshair; }}
    .brush-selection {{ position:absolute; background:rgba(76,175,80,0.2); border:2px solid #4CAF50; border-radius:3px; pointer-events:none; }}
    .brush-handle {{ position:absolute; top:0; bottom:0; width:8px; background:#4CAF50; cursor:ew-resize; }}
    .brush-handle.left {{ left:-4px; }}
    .brush-handle.right {{ right:-4px; }}
    
    /* Dashboard-style components */
    .dashboard-summary {{ background:#fff; padding:20px; border:2px inset #c0c0c0; margin-bottom:20px; grid-column:1 / -1; }}
    .dashboard-summary h2 {{ text-align:center; margin-bottom:20px; color:#000; font-size:1.5rem; }}
    .summary-cards {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:15px; }}
    .summary-card {{ background:#f8f8f8; border:2px outset #c0c0c0; padding:15px; display:flex; align-items:center; gap:15px; }}
    .card-icon {{ font-size:2rem; }}
    .card-content {{ flex:1; }}
    .card-value {{ font-size:1.8rem; font-weight:bold; color:#000; margin-bottom:5px; }}
    .card-label {{ font-size:0.9rem; color:#666; }}
    
    .dashboard-controls {{ background:#fff; padding:15px; border:2px inset #c0c0c0; margin-bottom:20px; grid-column:1 / -1; }}
    .control-row {{ display:flex; gap:10px; justify-content:center; flex-wrap:wrap; }}
    .dashboard-btn {{ background:#e0e0e0; color:#000; border:2px outset #c0c0c0; padding:10px 20px; font-family:inherit; cursor:pointer; font-size:0.9rem; font-weight:bold; }}
    .dashboard-btn:hover {{ background:#d0d0d0; }}
    .dashboard-btn:active {{ border:2px inset #c0c0c0; }}
    .reset-btn {{ background:#ffebee; color:#c62828; }}
    .export-btn {{ background:#e8f5e8; color:#2e7d32; }}
    .toggle-btn {{ background:#fff3e0; color:#ef6c00; }}
    
    .compact-filters {{ background:#fff; padding:15px; border:2px inset #c0c0c0; margin-bottom:20px; grid-column:1 / -1; }}
    .filter-section {{ margin-bottom:15px; }}
    .filter-section h4 {{ margin-bottom:10px; color:#000; background:#e0e0e0; padding:5px; border:1px outset #c0c0c0; }}
    .compact-filter {{ display:flex; align-items:center; gap:10px; margin-bottom:10px; }}
    .compact-filter label {{ min-width:120px; font-size:0.9rem; font-weight:bold; }}
    .compact-range {{ display:flex; align-items:center; gap:8px; flex:1; }}
    .compact-range input[type="range"] {{ flex:1; }}
    .compact-range input[type="number"] {{ width:70px; padding:3px; font-size:0.8rem; border:1px inset #c0c0c0; }}
    .charts-section, .stats-section {{ background:#fff; padding:15px; border:2px inset #c0c0c0; }}
    .intensity-section {{ background:#fff; padding:15px; border:2px inset #c0c0c0; margin-bottom:20px; grid-column:1 / -1; }}

    .cielab-section {{ background:#fff; padding:15px; border:2px inset #c0c0c0; margin-bottom:20px; grid-column:1 / -1; }}
    .lab-grid {{ display:grid; grid-template-columns:repeat(3, 1fr); gap:20px; align-items:center; justify-items:center; }}
    .stats-section {{ /* removed max-height and overflow */ }}
    .section h2 {{ color:#000; margin:0 0 15px 0; font-size:1.3rem; background:#c0c0c0; padding:8px; border:1px outset #c0c0c0; font-weight:bold; }}
    .chart-container {{ margin-bottom:30px; background:#f8f8f8; padding:15px; border:1px inset #c0c0c0; }}
    .chart-container canvas {{ max-height:300px; }}

    .color-chart-container, .lab-chart-container {{ margin-bottom:30px; background:#f8f8f8; padding:15px; border:1px inset #c0c0c0; display:flex; justify-content:center; align-items:center; position:relative; }}
    .color-chart-container canvas {{ max-width:350px; max-height:350px; width:350px; height:350px; }}

    .color-wheel-bg {{ position:relative; width:350px; height:350px; border-radius:50%; background:conic-gradient(from 0deg, #ff0000, #ffff00, #00ff00, #00ffff, #0000ff, #ff00ff, #ff0000); opacity:0.3; }}
    .color-wheel-bg canvas {{ position:absolute; top:0; left:0; }}

    /* LAB a*b* map */
    .lab-chart-container {{ position:relative; width:350px; height:350px; margin:0 auto; border:1px solid #ccc; }}
    .lab-chart-container canvas {{ position:absolute; top:0; left:0; width:350px !important; height:350px !important; }}
    #globalLabAbBg, #dcLabAbBg, #dominantLabAbBg {{ z-index:0; }}
    #globalLabAbChart, #dcLabAbChart, #dominantLabAbChart {{ z-index:1; }}

    .lab-controls {{ display:flex; gap:10px; align-items:center; margin-bottom:10px; }}
    .lab-controls label {{ font-weight:bold; }}

    /* 3D RGB plot styles */
    .rgb3d-section {{ background:#fff; padding:15px; border:2px inset #c0c0c0; margin-bottom:20px; grid-column:1 / -1; }}
    .rgb3d-grid {{ display:grid; grid-template-columns:repeat(3, 1fr); gap:20px; align-items:center; justify-items:center; }}
    .rgb3d-chart-container {{ position:relative; width:400px; height:400px; margin:0 auto; border:1px solid #ccc; background:#f9f9f9; }}
    .rgb3d-chart-container h4 {{ color:#000; margin:0 0 10px 0; font-size:1rem; background:#e0e0e0; padding:5px; border:1px outset #c0c0c0; font-weight:bold; text-align:center; }}

    /* 3D Direction plot styles */
    .direction3d-section {{ background:#fff; padding:15px; border:2px inset #c0c0c0; margin-bottom:20px; grid-column:1 / -1; }}
    .direction3d-container {{ display:flex; justify-content:center; align-items:center; }}
    .direction3d-chart-container {{ position:relative; width:600px; height:500px; margin:0 auto; border:1px solid #ccc; background:#f9f9f9; }}
    .direction3d-chart-container h4 {{ color:#000; margin:0 0 10px 0; font-size:1rem; background:#e0e0e0; padding:5px; border:1px outset #c0c0c0; font-weight:bold; text-align:center; }}

    .stats-table {{ margin-bottom:20px; background:#f8f8f8; padding:10px; border:1px inset #c0c0c0; }}
    .stats-table h4 {{ color:#000; margin:0 0 10px 0; font-size:1rem; background:#e0e0e0; padding:5px; border:1px outset #c0c0c0; font-weight:bold; }}
    .stats-table table {{ width:100%; border-collapse:collapse; font-size:0.9rem; }}
    .stats-table th, .stats-table td {{ padding:4px 8px; border:1px solid #808080; text-align:left; }}
    .stats-table th {{ background:#d0d0d0; font-weight:bold; }}
    .stats-table td:first-child {{ background:#e8e8e8; font-weight:bold; }}

    .navigation-section {{ background:#fff; padding:15px; border:2px inset #c0c0c0; margin-bottom:20px; }}
    .individual-links {{ display:flex; flex-wrap:wrap; gap:5px; margin-top:10px; max-height:100px; overflow-y:auto; }}
    .individual-link {{ 
        background:#f0f0f0; 
        color:#000; 
        text-decoration:none; 
        padding:3px 6px; 
        border:1px solid #ccc; 
        font-family:'Courier New',monospace; 
        font-size:0.7rem;
        border-radius:2px;
        transition:background-color 0.2s;
        white-space:nowrap;
    }}
    .individual-link:hover {{ background:#e0e0e0; }}
    .individual-link:active {{ background:#d0d0d0; }}

    /* Search functionality styles */
    .search-container input {{
        font-size: 14px;
        border-radius: 0;
        outline: none;
    }}
    .search-container input:focus {{
        border: 2px inset #808080;
    }}
    .individual-links {{
        max-height: 150px;
        overflow-y: auto;
        border: 1px solid #c0c0c0;
        padding: 8px;
        background: #f8f8f8;
    }}

    .overview-section {{ background:#fff; padding:15px; border:2px inset #c0c0c0; margin-bottom:20px; }}
    .overview-stats {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:15px; margin-top:10px; }}
    .overview-item {{ background:#f0f0f0; padding:10px; border:1px inset #c0c0c0; text-align:center; }}
    .overview-item h4 {{ color:#000; margin-bottom:5px; font-size:0.9rem; font-weight:bold; }}
    .overview-item .value {{ font-size:1.5rem; font-weight:bold; color:#000; }}

    .footer {{ text-align:center; color:#000; margin-top:20px; background:#c0c0c0; padding:10px; border:1px inset #c0c0c0; font-size:0.9rem; grid-column:1 / -1; }}

    @media (max-width: 1024px) {{
        .content {{ grid-template-columns: 1fr; }}
        .filter-panel-existing {{ max-height:none; margin-bottom:20px; }}
    }}
</style>
</head>
<body>
<div class=\"container\">
    <div class=\"header\">
        <h1>Aggregate Lighting Analysis Statistics</h1>
        <p>Experiment: {experiment_name}</p>
        <p>Dataset: {num_hdris} HDRI files</p>
    </div>

    <div class=\"navigation-section\">
        <h2>Individual Reports</h2>
        <div class=\"search-container\" style=\"margin-bottom: 15px;\">
            <input type=\"text\" id=\"hdriSearch\" placeholder=\"Search HDRIs...\" style=\"width: 100%; padding: 8px; font-family: 'Courier New', monospace; border: 2px inset #c0c0c0; background: #ffffff;\">
        </div>
        <div class=\"individual-links\" id=\"individualLinksContainer\">
            {''.join([f'<a href="{name}/{name}_report.html" class="individual-link" data-hdri-name="{name}">{name}</a>' for name in hdri_names])}
        </div>
    </div>

    <div class=\"overview-section\">
        <h2>Dataset Overview</h2>
        <div class=\"overview-stats\">
            <div class=\"overview-item\"><h4>Total HDRIs</h4><div class=\"value\">{num_hdris}</div></div>
            <div class=\"overview-item\"><h4>Experiment</h4><div class=\"value\">{experiment_name}</div></div>
            <div class=\"overview-item\"><h4>Analysis Type</h4><div class=\"value\">Lighting Distribution</div></div>
        </div>
    </div>

    {filter_controls_html}

    <div class=\"intensity-section\">
        <h2>📊 Intensity Distribution & Interactive Filtering</h2>
        <div style=\"display:grid; grid-template-columns:1fr 1fr; gap:20px;\">
            <div class=\"chart-filter-container\">
                <div>
                    <div class=\"chart-container\" style=\"position:relative;\">
                        <canvas id=\"globalIntensityChart\"></canvas>
                        <div class=\"chart-brush-overlay\" id=\"intensity-brush-overlay\"></div>
                    </div>
                    <div class=\"chart-controls-below\">
                        <div style=\"display:flex; align-items:center; gap:10px; font-size:0.9rem;\">
                            <span style=\"font-weight:bold;\">🎯 Click & Drag to Filter:</span>
                            <span id=\"selection-range\" style=\"color:#4CAF50; font-weight:bold;\">Full Range</span>
                            <button onclick=\"clearBrushSelection()\" style=\"background:#ff9800; color:white; border:none; padding:3px 8px; border-radius:3px; cursor:pointer; font-size:0.8rem;\">Clear</button>
                            <span id=\"chart-status\" style=\"font-size:0.8rem; color:#666; margin-left:10px;\">Loading...</span>
                        </div>
                    </div>
                </div>
            </div>
            <div class=\"chart-container\"><canvas id=\"areaIntensityChart\"></canvas></div>
        </div>
    </div>



    <div class=\"cielab-section\">
        <h2>Perceptual Color Maps (CIELAB)</h2>

        <div class=\"lab-controls\">
            <label for=\"labLevel\">L* slice:</label>
            <input id=\"labLevel\" type=\"range\" min=\"0\" max=\"100\" value=\"65\" oninput=\"document.getElementById('labLevelVal').textContent=this.value; updateLabBackgrounds(+this.value);\" />
            <span id=\"labLevelVal\">65</span>
        </div>
        <div class=\"lab-grid\">
            <div class=\"lab-chart-container\" style=\"width:350px;height:350px;\">
                <canvas id=\"globalLabAbBg\"></canvas>
                <canvas id=\"globalLabAbChart\"></canvas>
            </div>
            <div class=\"lab-chart-container\" style=\"width:350px;height:350px;\">
                <canvas id=\"dcLabAbBg\"></canvas>
                <canvas id=\"dcLabAbChart\"></canvas>
            </div>
            <div class=\"lab-chart-container\" style=\"width:350px;height:350px;\">
                <canvas id=\"dominantLabAbBg\"></canvas>
                <canvas id=\"dominantLabAbChart\"></canvas>
            </div>
        </div>
    </div>

    <div class=\"rgb3d-section\">
        <h2>3D RGB Color Space</h2>
        <p style=\"text-align:center; margin-bottom:15px; color:#666; font-size:0.9rem;\">Interactive 3D plots showing RGB values as coordinates. Click and drag to rotate, scroll to zoom.</p>
        <div class=\"plot3d-legend\" style=\"text-align:center; margin-bottom:20px; font-size:0.9rem;\">
            <span style=\"display:inline-block; margin-right:20px;\">
                <span style=\"display:inline-block; width:12px; height:12px; background:rgba(200,200,200,0.3); border:1px solid rgba(150,150,150,0.5); vertical-align:middle;\"></span> Other HDRIs
            </span>
            <span style=\"display:inline-block;\">
                <span style=\"display:inline-block; width:12px; height:12px; background:linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1); border:1px solid #000; vertical-align:middle;\"></span> Selected HDRIs (original colors)
            </span>
        </div>
        <div class=\"rgb3d-grid\">
            <div class=\"rgb3d-chart-container\">
                <h4>Global Color</h4>
                <div id=\"globalRgb3dChart\" style=\"width:100%; height:350px;\"></div>
            </div>
            <div class=\"rgb3d-chart-container\">
                <h4>L=0 Color (DC Term)</h4>
                <div id=\"dcRgb3dChart\" style=\"width:100%; height:350px;\"></div>
            </div>
            <div class=\"rgb3d-chart-container\">
                <h4>Dominant Color</h4>
                <div id=\"dominantRgb3dChart\" style=\"width:100%; height:350px;\"></div>
            </div>
        </div>
    </div>

    <div class=\"direction3d-section\">
        <h2>3D Dominant Direction Analysis</h2>
        <p style=\"text-align:left; margin-bottom:15px; color:#666; font-size:0.9rem;\">3D visualization of dominant lighting directions as unit vectors. Each point represents the primary direction of illumination for an HDRI, with colors showing the actual dominant color.</p>
        <div class=\"plot3d-legend\" style=\"text-align:center; margin-bottom:20px; font-size:0.9rem;\">
            <span style=\"display:inline-block; margin-right:20px;\">
                <span style=\"display:inline-block; width:12px; height:12px; background:rgba(200,200,200,0.3); border:1px solid rgba(150,150,150,0.5); vertical-align:middle;\"></span> Other HDRIs
            </span>
            <span style=\"display:inline-block;\">
                <span style=\"display:inline-block; width:12px; height:12px; background:linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1); border:1px solid #000; vertical-align:middle;\"></span> Selected HDRIs (original colors)
            </span>
        </div>
        <div class=\"direction3d-container\">
            <div class=\"direction3d-chart-container\">
                <h4>Dominant Direction Vectors</h4>
                <div id=\"dominantDirection3dChart\" style=\"width:100%; height:450px;\"></div>
            </div>
        </div>
    </div>

    <div class=\"stats-section\">
        <h2>Summary Statistics</h2>
        {stats_tables}
    </div>

    <div class=\"footer\">
        <p>Generated by LightingStudio Aggregate Analysis Pipeline</p>
        <p>HDRI files: {', '.join(hdri_names[:5])}{'...' if len(hdri_names) > 5 else ''}</p>
    </div>
</div>
<script>
// Initialize global variables that charts will use
let allCharts = {{
    intensity: null,
    areaIntensity: null,
    labCharts: {{
        global: null,
        dc: null,
        dominant: null
    }},
    plotly3d: {{
        globalRgb: null,
        dcRgb: null,
        dominantRgb: null,
        direction: null
    }}
}};

let globalIntensityChart = null;
let intensityRange = {{ min: 0, max: 1 }};
let originalVisualizationData = {{
    labData: {{
        global: null,
        dc: null,
        dominant: null
    }},
    rgb3dData: {{
        global: null,
        dc: null,
        dominant: null
    }},
    direction3dData: null
}};

{charts_js}

{filter_js}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate aggregate statistics for an experiment (with CIELAB maps)")
    parser.add_argument("experiment_dir", type=str, help="Path to experiment directory")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists() or not experiment_dir.is_dir():
        print(f"Error: Experiment directory does not exist: {experiment_dir}")
        exit(1)

    try:
        html_path = generate_aggregate_statistics_html(experiment_dir)
        print(f"Successfully generated aggregate statistics: {html_path}")
    except Exception as e:
        print(f"Error generating aggregate statistics: {e}")
        exit(1)
