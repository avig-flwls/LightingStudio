"""
Simple Blender rendering functions for LightingStudio analysis pipeline.
"""

import shutil
import subprocess
import numpy as np
import cv2
import json
from pathlib import Path


# Global configuration
BLENDER_PATH = r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe"
PACKED_BLEND_FILE_PATH = Path(r"C:\Users\AviGoyal\Documents\LightingStudio\tmp\source\render_assets\packed_blend_file.blend")
PREPARE_SCRIPT_PATH = Path(r"C:\Users\AviGoyal\Documents\LightingStudio\src\LightingStudio\analysis\blender_interface\prepare_blend_file.py")


def copy_render_assets(destination_dir: Path) -> bool:
    """Copy the packed blend file into the render_assets directory with its original name."""
    render_assets_dest = destination_dir / "render_assets"
    if render_assets_dest.exists():
        shutil.rmtree(render_assets_dest)
    render_assets_dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(PACKED_BLEND_FILE_PATH, render_assets_dest / PACKED_BLEND_FILE_PATH.name)
    return True


def prepare_blend_file(output_dir: Path, hdri_path: Path) -> bool:
    """Prepare blend file with HDRI using Blender script."""
    base_blend_file = output_dir / "render_assets" / PACKED_BLEND_FILE_PATH.name
    
    cmd = [
        BLENDER_PATH,
        str(base_blend_file),
        "--background",
        "--python", str(PREPARE_SCRIPT_PATH),
        "--",
        "--experiment_folder", str(output_dir),
        "--hdri_path", str(hdri_path)
    ]
    
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return True


def render_blend_file(output_dir: Path) -> bool:
    """Render the prepared blend file."""
    blend_file = output_dir / "render_assets" / PACKED_BLEND_FILE_PATH.name
    cmd = [BLENDER_PATH, str(blend_file), "--background", "-a"]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return True


def compute_render_metrics(output_dir: Path) -> bool:
    """Compute render metrics."""
    render_metrics_dir = output_dir / "blender_renders"
    metrics = {}

    # Process individual files
    for file in render_metrics_dir.iterdir():
        if file.is_file():
            img = cv2.imread(str(file), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            metrics[f"{file.stem}_intensity"] = float(np.mean(img))

    # Compute person averages
    person_0001_values = [v for k, v in metrics.items() if k.startswith("person_0001")]
    person_0002_values = [v for k, v in metrics.items() if k.startswith("person_0002")]
    
    metrics["person_0001_intensity"] = float(np.mean(person_0001_values))
    metrics["person_0002_intensity"] = float(np.mean(person_0002_values))

    # Save metrics
    metrics_path = output_dir / "blender_renders" / "render_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    return True


def process_hdri_with_blender(hdri_path: Path, output_dir: Path) -> bool:
    """Complete Blender pipeline: copy assets, prepare, and render."""
    copy_render_assets(output_dir)
    prepare_blend_file(output_dir, hdri_path)
    print("Rendering with Blender...") 
    render_blend_file(output_dir)
    compute_render_metrics(output_dir)
    return True
