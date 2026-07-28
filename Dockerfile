FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends procps && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .

# Extract the KEGG module graphs now, while the filesystem is still writable.
# Singularity mounts the image read-only at run time, so leaving this to first
# use would fail on HPC. Also fails the build if the data files did not make it
# into the installed package.
RUN python -c "\
from BLIMMP_Scripts.module_detection import ensure_module_graphs; \
print('module graphs:', ensure_module_graphs())" \
 && python -c "\
import BLIMMP_Scripts, pathlib, sys; \
root = pathlib.Path(BLIMMP_Scripts.__file__).parent; \
required = ['Data_Dependencies/kegg_bacteria_modules.json', \
            'Data_Dependencies/ko_list.txt', \
            'Data_Dependencies/module_freq.txt', \
            'Graph_Dependencies/KEGG_Module_Equations_Jan26.json', \
            'Graph_Dependencies/module_ko_reaction.json']; \
missing = [r for r in required if not (root / r).exists()]; \
sys.exit('Missing packaged data files: ' + repr(missing)) if missing else print('packaged data OK')"
