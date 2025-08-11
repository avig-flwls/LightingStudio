


Nuke License
```
set foundry_LICENSE=4101@license.us-west-2.root.flwls-infra.net
& "C:\Program Files\Nuke13.2v9\Nuke13.2.exe"
``


Download HDRI's
```
cd LightingStudio
python src/LightingStudio/ingest/scrape_polyhaven.py --max-downloads 5 --max-resolution 1k --root-output-dir .\tmp\source\
```