"""
HTML Report Generator for HDRI Analysis
Generates beautiful HTML reports displaying analysis results.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from src.LightingStudio.analysis.utils.io import exr_to_png_tensor, exr_to_png


def generate_html_report(
    hdri_output_dir: Path,
    hdri_name: str,
    hdri_list: Optional[List[str]] = None
) -> str:
    """
    Generate an HTML report for a single HDRI analysis.
    
    :param hdri_output_dir: Directory containing the analysis outputs
    :param hdri_name: Name of the HDRI (stem)
    :param hdri_list: Optional list of all HDRI names for navigation
    :return: Path to the generated HTML file
    """
    
    # Create web assets directory
    web_dir = hdri_output_dir / "web"
    web_dir.mkdir(exist_ok=True)
    
    # Define the PNG files that should already exist (created during processing)
    png_files = {}
    png_names = [
        "original", "median_cut", "density_map_fast", "sph_metrics",
        "reconstructed_1", "reconstructed_2", "reconstructed_3"
    ]
    
    for name in png_names:
        png_path = web_dir / f"{hdri_name}_{name}.png"
        if png_path.exists():
            png_files[name] = f"web/{png_path.name}"
            print(f"Found PNG: {png_path.name}")
        else:
            print(f"Warning: {png_path.name} not found")
            png_files[name] = None
    
    # Load analysis metrics if available
    metrics = {}
    
    # Load naive metrics
    naive_metrics_path = hdri_output_dir / f"{hdri_name}_naive_metrics.json"
    if naive_metrics_path.exists():
        with open(naive_metrics_path, 'r') as f:
            metrics['naive'] = json.load(f)
    
    # Load SPH metrics  
    sph_metrics_path = hdri_output_dir / f"{hdri_name}_sph_metrics.json"
    if sph_metrics_path.exists():
        with open(sph_metrics_path, 'r') as f:
            metrics['sph'] = json.load(f)
    
    # Generate HTML content
    html_content = _generate_html_template(hdri_name, png_files, metrics, hdri_list, hdri_output_dir)
    
    # Write HTML file
    html_path = hdri_output_dir / f"{hdri_name}_report.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Generated HTML report: {html_path}")
    return str(html_path)


def _generate_html_template(
    hdri_name: str, 
    png_files: Dict[str, Optional[str]], 
    metrics: Dict,
    hdri_list: Optional[List[str]] = None,
    hdri_output_dir: Optional[Path] = None
) -> str:
    """Generate the HTML template with the analysis results."""
    
    # Build the image gallery with 3 rows
    
    # Row 1: Original HDRI (really big)
    original_row = ""
    if png_files.get("original"):
        original_row = f"""
        <div class="original-container">
            <h3>Original HDRI</h3>
            <img src="{png_files['original']}" alt="Original HDRI" onclick="openModal(this)">
            <p class="image-description">The original high dynamic range image</p>
        </div>
        """
    else:
        original_row = f"""
        <div class="original-container missing">
            <h3>Original HDRI</h3>
            <div class="missing-image">Image not available</div>
            <p class="image-description">The original high dynamic range image</p>
        </div>
        """
    
    # Row 2: Median Cut, Density Map, SPH Metrics
    analysis_row = ""
    analysis_configs = [
        ("median_cut", "Median Cut Sampling", "Visualization of median cut sampling points"),
        ("density_map_fast", "Density Map (Fast)", "Fast density estimation map"),
        ("sph_metrics", "Dominant Direction and Color", "Spherical harmonic derived dominant direction and color")
    ]
    
    for img_key, title, description in analysis_configs:
        if png_files.get(img_key):
            analysis_row += f"""
            <div class="analysis-container">
                <h4>{title}</h4>
                <img src="{png_files[img_key]}" alt="{title}" onclick="openModal(this)">
                <p class="image-description">{description}</p>
            </div>
            """
        else:
            analysis_row += f"""
            <div class="analysis-container missing">
                <h4>{title}</h4>
                <div class="missing-image">Image not available</div>
                <p class="image-description">{description}</p>
            </div>
            """
    
    # Row 3: Spherical Harmonic Reconstructions (L=1, L=2, L=3)
    sph_gallery = ""
    sph_configs = [
        ("reconstructed_1", "L=1", "Spherical harmonic reconstruction using bands 0:1"),
        ("reconstructed_2", "L=2", "Spherical harmonic reconstruction using bands 0:2"), 
        ("reconstructed_3", "L=3", "Spherical harmonic reconstruction using bands 0:3")
    ]
    
    for img_key, title, description in sph_configs:
        if png_files.get(img_key):
            sph_gallery += f"""
            <div class="sph-image-container">
                <h4>{title}</h4>
                <img src="{png_files[img_key]}" alt="{title}" onclick="openModal(this)">
                <p class="image-description">{description}</p>
            </div>
            """
        else:
            sph_gallery += f"""
            <div class="sph-image-container missing">
                <h4>{title}</h4>
                <div class="missing-image">Image not available</div>
                <p class="image-description">{description}</p>
            </div>
            """
    
    # Build navigation bar
    navigation_bar = ""
    if hdri_list and len(hdri_list) > 1:
        current_index = hdri_list.index(hdri_name) if hdri_name in hdri_list else 0
        prev_hdri = hdri_list[current_index - 1] if current_index > 0 else None
        next_hdri = hdri_list[current_index + 1] if current_index < len(hdri_list) - 1 else None
        
        # Determine the experiment name for aggregate link (parent directory name)
        experiment_name = hdri_output_dir.parent.name if hdri_output_dir else "experiment"
        aggregate_file = f"../{experiment_name}_aggregate_statistics.html"
        
        navigation_bar = f"""
        <div class="navigation-bar">
            <button class="nav-button" onclick="navigateToHdri('{prev_hdri}')" {'disabled' if prev_hdri is None else ''}>
                ← Previous
            </button>
            <span class="nav-counter">{current_index + 1} of {len(hdri_list)}</span>
            <button class="nav-button" onclick="navigateToHdri('{next_hdri}')" {'disabled' if next_hdri is None else ''}>
                Next →
            </button>
            <button class="nav-button aggregate-button" onclick="window.location.href='{aggregate_file}'">
                📊 Aggregate Statistics
            </button>
        </div>
        """
    else:
        # Even for single HDRI, show link to aggregate if multiple HDRIs exist
        experiment_name = hdri_output_dir.parent.name if hdri_output_dir else "experiment"
        aggregate_file = f"../{experiment_name}_aggregate_statistics.html"
        navigation_bar = f"""
        <div class="navigation-bar">
            <span class="nav-counter">Individual Report</span>
            <button class="nav-button aggregate-button" onclick="window.location.href='{aggregate_file}'">
                📊 Aggregate Statistics
            </button>
        </div>
        """
    
    # Build metrics section
    metrics_section = ""
    
    # Helper function to convert RGB list to hex color
    def rgb_to_hex(rgb_list):
        if isinstance(rgb_list, list) and len(rgb_list) >= 3:
            # Values are already in 0-255 range, just clamp and convert to int
            r = max(0, min(255, int(rgb_list[0])))
            g = max(0, min(255, int(rgb_list[1])))
            b = max(0, min(255, int(rgb_list[2])))
            return f"#{r:02x}{g:02x}{b:02x}"
        return "#000000"
    
    if metrics.get('naive'):
        naive = metrics['naive']
        global_color = naive.get('global_color', [0, 0, 0])
        global_intensity = naive.get('global_intensity', 'N/A')
        
        global_intensity_str = f"{global_intensity:.4f}" if isinstance(global_intensity, (int, float)) else str(global_intensity)
        color_hex = rgb_to_hex(global_color)
        
        metrics_section += f"""
        <div class="metrics-group">
            <h3>Naive Metrics</h3>
            <div class="metrics-grid">
                <div class="metric">
                    <span class="metric-label">Global Color:</span>
                    <div class="color-with-values">
                        <div class="color-display" style="background-color: {color_hex};" title="RGB: {global_color}"></div>
                        <span class="color-values">[{global_color[0]:.0f}, {global_color[1]:.0f}, {global_color[2]:.0f}]</span>
                    </div>
                </div>
                <div class="metric">
                    <span class="metric-label">Global Intensity:</span>
                    <span class="metric-value">{global_intensity_str}</span>
                </div>
            </div>
        </div>
        """
    
    if metrics.get('sph'):
        sph = metrics['sph']
        dominant_color = sph.get('dominant_color', [0, 0, 0])
        dc_color = sph.get('dc_color', [0, 0, 0])
        area_intensity = sph.get('area_intensity', [0, 0, 0])
        
        # Show area intensity as RGB vector
        area_intensity_str = "N/A"
        if isinstance(area_intensity, list) and len(area_intensity) >= 3:
            # Format as RGB vector with 4 decimal places
            r, g, b = area_intensity[0], area_intensity[1], area_intensity[2]
            area_intensity_str = f"[{r:.4f}, {g:.4f}, {b:.4f}]"
        
        dominant_color_hex = rgb_to_hex(dominant_color)
        dc_color_hex = rgb_to_hex(dc_color)
        
        metrics_section += f"""
        <div class="metrics-group">
            <h3>Spherical Harmonic Metrics</h3>
            <div class="metrics-grid">
                <div class="metric">
                    <span class="metric-label">L=0 Color (DC Term):</span>
                    <div class="color-with-values">
                        <div class="color-display" style="background-color: {dc_color_hex};" title="RGB: {dc_color}"></div>
                        <span class="color-values">[{dc_color[0]:.0f}, {dc_color[1]:.0f}, {dc_color[2]:.0f}]</span>
                    </div>
                </div>
                <div class="metric">
                    <span class="metric-label">Dominant Color:</span>
                    <div class="color-with-values">
                        <div class="color-display" style="background-color: {dominant_color_hex};" title="RGB: {dominant_color}"></div>
                        <span class="color-values">[{dominant_color[0]:.0f}, {dominant_color[1]:.0f}, {dominant_color[2]:.0f}]</span>
                    </div>
                </div>
                <div class="metric">
                    <span class="metric-label">Dominant Area Intensity:</span>
                    <span class="metric-value">{area_intensity_str}</span>
                </div>
            </div>
        </div>
        """
    
    # Timing information section removed
    
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HDRI Analysis Report - {hdri_name}</title>
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
            padding: 5px;
        }}
        
        .header {{
            text-align: center;
            color: #000;
            margin-bottom: 5px;
            background: #808080;
            padding: 5px;
            border: 2px inset #c0c0c0;
        }}
        
        .header h1 {{
            font-size: 1.4rem;
            margin: 0;
            font-weight: bold;
        }}
        
        .header p {{
            font-size: 1.0rem;
            margin: 0;
        }}
        
        .navigation-bar {{
            display: flex;
            justify-content: center;
            align-items: center;
            background: #808080;
            padding: 3px;
            border: 1px inset #c0c0c0;
            margin-bottom: 3px;
            gap: 10px;
        }}
        
        .nav-button {{
            background: #c0c0c0;
            border: 1px outset #c0c0c0;
            padding: 2px 8px;
            cursor: pointer;
            font-family: 'Courier New', monospace;
            font-size: 0.8rem;
            color: #000;
        }}
        
        .nav-button:hover:not(:disabled) {{
            background: #e0e0e0;
        }}
        
        .nav-button:active:not(:disabled) {{
            border: 1px inset #c0c0c0;
        }}
        
        .nav-button:disabled {{
            background: #a0a0a0;
            color: #606060;
            cursor: not-allowed;
        }}
        
        .aggregate-button {{
            background: #c0c0c0 !important;
            color: #000 !important;
            border: 1px outset #c0c0c0 !important;
            font-weight: bold;
        }}
        
        .aggregate-button:hover {{
            background: #e0e0e0 !important;
        }}
        
        .aggregate-button:active {{
            border: 1px inset #c0c0c0 !important;
        }}
        
        .nav-counter {{
            font-family: 'Courier New', monospace;
            font-size: 0.8rem;
            color: #000;
            font-weight: bold;
            padding: 0 10px;
        }}
        
        .content {{
            display: block;
        }}
        
        .section {{
            background: #ffffff;
            padding: 2px;
            border: 2px inset #c0c0c0;
            margin-bottom: 3px;
        }}
        
        .section h2 {{
            color: #000;
            margin: 0 0 3px 0;
            font-size: 1.1rem;
            background: #c0c0c0;
            padding: 2px 5px;
            border: 1px outset #c0c0c0;
            font-weight: bold;
        }}
        
        .original-row {{
            display: flex;
            gap: 3px;
            margin: 3px 0;
        }}
        
        .original-container {{
            text-align: center;
            background: #ffffff;
            padding: 1px;
            border: 1px inset #c0c0c0;
            flex: 3;
        }}
        
        .metrics-sidebar {{
            background: #ffffff;
            padding: 2px;
            border: 1px inset #c0c0c0;
            flex: 2;
            min-width: 200px;
            max-width: 400px;
        }}
        
        .metrics-sidebar h2 {{
            color: #000;
            margin: 0 0 3px 0;
            font-size: 0.9rem;
            background: #c0c0c0;
            padding: 2px 5px;
            border: 1px outset #c0c0c0;
            font-weight: bold;
        }}
        
        .original-container h3 {{
            color: #000;
            margin: 0 0 2px 0;
            font-size: 1.0rem;
            font-weight: bold;
            background: #c0c0c0;
            padding: 2px 5px;
        }}
        
        .original-container img {{
            max-width: 100%;
            height: auto;
            cursor: pointer;
            border: 1px solid #808080;
        }}
        
        .analysis-row {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 3px;
            margin: 3px 0;
        }}
        
        .analysis-container {{
            text-align: center;
            background: #ffffff;
            padding: 1px;
            border: 1px inset #c0c0c0;
        }}
        
        .analysis-container h4 {{
            color: #000;
            margin: 0 0 2px 0;
            font-size: 0.8rem;
            font-weight: bold;
            background: #c0c0c0;
            padding: 1px;
        }}
        
        .analysis-container img {{
            max-width: 100%;
            height: auto;
            cursor: pointer;
            border: 1px solid #808080;
        }}
        
        .analysis-container:hover {{
            background: #e0e0e0;
        }}
        
        .analysis-container img:hover {{
            border: 1px solid #000;
        }}
        
        .sph-row {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 3px;
            margin-top: 3px;
        }}
        
        .sph-image-container {{
            text-align: center;
            background: #ffffff;
            padding: 1px;
            border: 1px inset #c0c0c0;
        }}
        
        .sph-image-container:hover {{
            background: #e0e0e0;
        }}
        
        .sph-image-container h4 {{
            color: #000;
            margin: 0 0 2px 0;
            font-size: 0.8rem;
            font-weight: bold;
            background: #c0c0c0;
            padding: 1px;
        }}
        
        .sph-image-container img {{
            max-width: 100%;
            height: auto;
            cursor: pointer;
            border: 1px solid #808080;
        }}
        
        .sph-image-container img:hover {{
            border: 1px solid #000;
        }}
        
        .sph-image-container.missing {{
            border: 1px dashed #808080;
            background: #e0e0e0;
        }}
        
        .image-container {{
            text-align: center;
            background: #ffffff;
            padding: 1px;
            border: 1px inset #c0c0c0;
        }}
        
        .image-container:hover {{
            background: #e0e0e0;
        }}
        
        .image-container h3 {{
            color: #000;
            margin: 0 0 2px 0;
            font-size: 0.9rem;
            font-weight: bold;
            background: #c0c0c0;
            padding: 1px 3px;
        }}
        
        .image-container img {{
            max-width: 100%;
            height: auto;
            cursor: pointer;
            border: 1px solid #808080;
        }}
        
        .image-container img:hover {{
            border: 1px solid #000;
        }}
        
        .image-description {{
            margin: 2px 0 0 0;
            color: #404040;
            font-size: 0.7rem;
            padding: 1px;
        }}
        
        .missing-image {{
            height: 150px;
            background: #e0e0e0;
            border: 1px dashed #808080;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #404040;
            font-size: 0.8rem;
        }}
        
        .metrics-group {{
            margin-bottom: 5px;
        }}
        
        .metrics-group h3 {{
            color: #000;
            margin: 0 0 2px 0;
            font-size: 0.9rem;
            background: #c0c0c0;
            padding: 2px 5px;
            border: 1px outset #c0c0c0;
            font-weight: bold;
        }}
        
        .metrics-grid {{
            display: block;
        }}
        
        .metric {{
            background: #ffffff;
            padding: 2px 5px;
            border: 1px inset #c0c0c0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1px;
        }}
        
        .metric-label {{
            font-weight: bold;
            color: #000;
            font-size: 0.8rem;
        }}
        
        .metric-value {{
            font-family: 'Courier New', monospace;
            font-weight: normal;
            color: #000;
            background: #f0f0f0;
            padding: 1px 3px;
            border: 1px inset #c0c0c0;
            font-size: 0.8rem;
        }}
        
        .color-with-values {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        
        .color-display {{
            width: 40px;
            height: 20px;
            border: 1px solid #000;
            cursor: pointer;
            flex-shrink: 0;
        }}
        
        .color-display:hover {{
            border: 2px solid #000;
        }}
        
        .color-values {{
            font-family: 'Courier New', monospace;
            font-size: 0.7rem;
            color: #000;
            background: #f0f0f0;
            padding: 1px 3px;
            border: 1px inset #c0c0c0;
        }}
        
        /* Modal styles */
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
        }}
        
        .modal-content {{
            margin: auto;
            display: block;
            width: 90%;
            max-width: 1200px;
            max-height: 90%;
            object-fit: contain;
        }}
        
        .modal-content {{
            animation: zoom 0.3s;
        }}
        
        @keyframes zoom {{
            from {{transform:scale(0)}}
            to {{transform:scale(1)}}
        }}
        
        .close {{
            position: absolute;
            top: 15px;
            right: 35px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            transition: 0.3s;
            cursor: pointer;
        }}
        
        .close:hover,
        .close:focus {{
            color: #bbb;
        }}
        
        .footer {{
            text-align: center;
            color: #000;
            margin-top: 5px;
            background: #c0c0c0;
            padding: 2px;
            border: 1px inset #c0c0c0;
            font-size: 0.7rem;
        }}
        
        @media (max-width: 768px) {{
            .original-row {{
                flex-direction: column;
            }}
            
            .metrics-sidebar {{
                min-width: auto;
                max-width: none;
            }}
            
            .analysis-row {{
                grid-template-columns: 1fr;
                gap: 2px;
            }}
            
            .sph-row {{
                grid-template-columns: 1fr;
                gap: 2px;
            }}
            
            .header h1 {{
                font-size: 1.2rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>HDRI Analysis Report</h1>
            <p>{hdri_name}</p>
        </div>
        
        {navigation_bar}
        
        <div class="content">
            <div class="section">
                <h2>Analysis Visualizations</h2>
                
                <!-- Row 1: Original HDRI with Metrics Sidebar -->
                <div class="original-row">
                    {original_row}
                    <div class="metrics-sidebar">
                        <h2>Analysis Metrics</h2>
                        {metrics_section}
                    </div>
                </div>
                
                <!-- Row 2: Analysis Images -->
                <h3>Analysis Results</h3>
                <div class="analysis-row">
                    {analysis_row}
                </div>
                
                <!-- Row 3: SPH Reconstructions -->
                <h3>SPH Reconstructions</h3>
                <div class="sph-row">
                    {sph_gallery}
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated by LightingStudio Analysis Pipeline</p>
        </div>
    </div>
    
    <!-- Modal for image viewing -->
    <div id="imageModal" class="modal">
        <span class="close">&times;</span>
        <img class="modal-content" id="modalImage">
    </div>
    
    <script>
        function openModal(img) {{
            var modal = document.getElementById("imageModal");
            var modalImg = document.getElementById("modalImage");
            modal.style.display = "flex";
            modal.style.alignItems = "center";
            modal.style.justifyContent = "center";
            modalImg.src = img.src;
        }}
        
        // Close modal when clicking on close button or outside the image
        document.getElementsByClassName("close")[0].onclick = function() {{
            document.getElementById("imageModal").style.display = "none";
        }}
        
        document.getElementById("imageModal").onclick = function(event) {{
            if (event.target === this) {{
                this.style.display = "none";
            }}
        }}
        
        // Close modal with Escape key
        document.addEventListener('keydown', function(event) {{
            if (event.key === 'Escape') {{
                document.getElementById("imageModal").style.display = "none";
            }}
        }});
        
        // Navigation function
        function navigateToHdri(hdriName) {{
            if (hdriName && hdriName !== 'None') {{
                // Navigate to the corresponding HDRI report in its subdirectory
                window.location.href = '../' + hdriName + '/' + hdriName + '_report.html';
            }}
        }}
        
        // Keyboard navigation
        document.addEventListener('keydown', function(event) {{
            if (event.key === 'ArrowLeft') {{
                const prevButton = document.querySelector('.nav-button:first-child');
                if (prevButton && !prevButton.disabled) {{
                    prevButton.click();
                }}
            }} else if (event.key === 'ArrowRight') {{
                const nextButton = document.querySelector('.nav-button:last-child');
                if (nextButton && !nextButton.disabled) {{
                    nextButton.click();
                }}
            }}
        }});
    </script>
</body>
</html>
    """
