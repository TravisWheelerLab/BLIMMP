# BLIMMP

conda create -n BLIMMP-test

conda activate BLIMMP-test

git clone this repo

cd BLIMMP

pip install .

python BLIMMP_Scripts/module_detection.py -h

python META_DAWG/module_detection.py ./Examples/example.domtblout \
	--format domtblout \
    -c 0.5 \
    --output example_name
