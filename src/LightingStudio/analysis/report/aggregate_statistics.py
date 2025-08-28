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
from typing import Dict, List, Tuple
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
        'area_intensity': []
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
    
    return pd.DataFrame(data_dict)


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

    # Apply data sampling for performance with large datasets
    MAX_VISUALIZATION_POINTS = 500  # Limit to 500 points for performance
    if num_hdris > MAX_VISUALIZATION_POINTS:
        print(f"Large dataset detected ({num_hdris} HDRIs). Sampling {MAX_VISUALIZATION_POINTS} points for visualization performance.")
        
        # Sample DataFrame
        viz_df = df.sample(n=MAX_VISUALIZATION_POINTS, random_state=42).sort_index()
        print(f"Sampled DataFrame with shape: {viz_df.shape}")
        
        # Convert back to metrics_data format for visualization
        viz_data = {
            'hdri_names': viz_df['hdri_name'].tolist(),
            'global_color': viz_df[['global_r', 'global_g', 'global_b']].values.tolist(),
            'global_intensity': viz_df['global_intensity'].tolist(),
            'dc_color': viz_df[['dc_r', 'dc_g', 'dc_b']].values.tolist(),
            'dominant_color': viz_df[['dominant_r', 'dominant_g', 'dominant_b']].values.tolist(),
            'area_intensity': viz_df[['area_r', 'area_g', 'area_b']].values.tolist()
        }
    else:
        print(f"Small dataset ({num_hdris} HDRIs). Using all data for visualization.")
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

    # Summary statistics using pandas - use full dataset for statistics
    stats = calculate_pandas_stats(df)

    # Add sampling info to stats if applicable
    if num_hdris > MAX_VISUALIZATION_POINTS:
        stats['sampling_info'] = {
            'total_hdris': num_hdris,
            'visualized_hdris': len(viz_data['hdri_names']),
            'sampling_note': f"Visualizations show a random sample of {len(viz_data['hdri_names'])} HDRIs for performance. Statistics are calculated from all {num_hdris} HDRIs."
        }

    html_content = _generate_aggregate_html_template(
        experiment_name,
        num_hdris,
        visualizations,
        stats,
        metrics_data['hdri_names']
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
    hdri_names: List[str]
) -> str:
    def to_js(x):
        return str(x).replace("'", '"')

    charts_js = ""

    # ------------------ Global Intensity Histogram ------------------
    if visualizations['global_intensity'][0]:
        charts_js += f"""
        // Global Intensity Chart
        var globalIntensityCtx = document.getElementById('globalIntensityChart').getContext('2d');
        new Chart(globalIntensityCtx, {{
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
                    x: {{ title: {{ display: true, text: 'Global Intensity' }} }}
                }}
            }}
        }});
        """

    # ------------------ Area Intensity RGB Histogram ------------------
    if visualizations['area_intensity']['r'][0]:
        charts_js += f"""
        // Area Intensity RGB Histogram
        var areaIntensityCtx = document.getElementById('areaIntensityChart').getContext('2d');
        new Chart(areaIntensityCtx, {{
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

    // Initialize backgrounds when page loads
    document.addEventListener('DOMContentLoaded', function() {
        console.log('DOM loaded, initializing LAB backgrounds...');
        updateLabBackgrounds(currentLabL);
    });
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
<style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{ font-family: 'Courier New', monospace; line-height:1.4; color:#000; background:#c0c0c0; }}
    .container {{ max-width: 1400px; margin:0 auto; padding:10px; }}
    .header {{ text-align:center; color:#000; margin-bottom:20px; background:#808080; padding:10px; border:2px inset #c0c0c0; }}
    .header h1 {{ font-size:2rem; margin-bottom:5px; font-weight:bold; }}
    .header p {{ font-size:1.1rem; margin:5px 0; }}
    .content {{ display:grid; grid-template-columns:2fr 1fr; gap:20px; }}
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

    .overview-section {{ background:#fff; padding:15px; border:2px inset #c0c0c0; margin-bottom:20px; }}
    .overview-stats {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:15px; margin-top:10px; }}
    .overview-item {{ background:#f0f0f0; padding:10px; border:1px inset #c0c0c0; text-align:center; }}
    .overview-item h4 {{ color:#000; margin-bottom:5px; font-size:0.9rem; font-weight:bold; }}
    .overview-item .value {{ font-size:1.5rem; font-weight:bold; color:#000; }}

    .footer {{ text-align:center; color:#000; margin-top:20px; background:#c0c0c0; padding:10px; border:1px inset #c0c0c0; font-size:0.9rem; grid-column:1 / -1; }}

    @media (max-width: 1024px) {{
        .content {{ grid-template-columns: 1fr; }}
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
        <div class=\"individual-links\">
            {', '.join([f'<a href="{name}/{name}_report.html" class="individual-link">{name}</a>' for name in hdri_names])}
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

    <div class=\"intensity-section\">
        <h2>Intensity Distribution Plots</h2>
        <div style=\"display:grid; grid-template-columns:1fr 1fr; gap:20px;\">
            <div class=\"chart-container\"><canvas id=\"globalIntensityChart\"></canvas></div>
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
{charts_js}
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
