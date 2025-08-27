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

tests
```
python -m src.LightingStudio.analysis.unit_tests.spherical_harmonic_test -H 1024 -W 2048 --l-max 4
python -m src.LightingStudio.analysis.unit_tests.transforms_test  -H 1024 -W 2048
python -m src.LightingStudio.analysis.unit_tests.view_solid_angle_test
```