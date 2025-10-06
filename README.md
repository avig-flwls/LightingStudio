# LightingStudio

A toolkit for lighting analysis HDRI

## Setup

### Python Environment

```bash
# Clone the repository
git clone <repo-url>
cd LightingStudio

# Install dependencies using uv
uv sync --all-extras

# Activate virtual environment
# Linux/macOS:
source .venv/bin/activate
# Windows:
.\.venv\Scripts\activate

uv pip install -e .
```

### Nuke License Configuration

```batch
set foundry_LICENSE=4101@license.us-west-2.root.flwls-infra.net
& "C:\Program Files\Nuke13.2v9\Nuke13.2.exe"
```

## Usage

### Download HDRIs from Polyhaven

```bash
cd LightingStudio
python src/LightingStudio/ingest/scrape_polyhaven.py --max-downloads 5 --max-resolution 1k --root-output-dir .\tmp\source\
```

### Analysis and Report Generation

```bash
# Usage examples:

# Process a single HDRI file (saves both EXR and PNG):
python -m src.LightingStudio.analysis.report.generate_report_parallel --hdri "path/to/file.exr" --n_samples 1024 --l_max 3

# Process a single HDRI file (PNG only for faster processing):
python -m src.LightingStudio.analysis.report.generate_report_parallel --hdri "path/to/file.exr" --n_samples 1024 --l_max 3 --png-only

# Process all HDRI files in a folder with parallel processing (auto-detects CPU cores):
python -m src.LightingStudio.analysis.report.generate_report_parallel --folder "path/to/hdris" --n_samples 1024 --l_max 3 --png-only
python -m src.LightingStudio.analysis.report.generate_report_parallel --folder "C:\Users\AviGoyal\Documents\LightingStudio\tmp\source\1k_small" --n_samples 1024 --l_max 3 --png-only

python -m src.LightingStudio.analysis.report.generate_report_parallel --folder "C:\Users\AviGoyal\Documents\LightingStudio\tmp\source\1k_small_med" --n_samples 2 --l_max 3 --png-only --processes 4

# Process with specific number of parallel processes:
python -m src.LightingStudio.analysis.report.generate_report_parallel --folder "path/to/hdris" --n_samples 1024 --l_max 3 --png-only --processes 4

# Process sequentially (single-threaded):
python -m src.LightingStudio.analysis.report.generate_report_parallel --folder "path/to/hdris" --n_samples 1024 --l_max 3 --png-only --processes 1

# Only run analysis on the following HDRIs:
python -m src.LightingStudio.analysis.report.aggregate_statistics  "C:\Users\AviGoyal\Documents\LightingStudio\tmp\experiments\dainty-flounder"

```

### Running Tests

```bash
python -m src.LightingStudio.analysis.unit_tests.spherical_harmonic_test -H 1024 -W 2048 --l-max 4
python -m src.LightingStudio.analysis.unit_tests.transforms_test  -H 1024 -W 2048
python -m src.LightingStudio.analysis.unit_tests.view_solid_angle_test
```