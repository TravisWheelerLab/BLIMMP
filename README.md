# BLIMMP

conda create -n BLIMMP-test python=3.10 -y

conda activate BLIMMP-test

git clone this repo

cd BLIMMP

pip install .

python BLIMMP_Scripts/module_detection.py -h

cd ..

time python BLIMMP/BLIMMP_Scripts/module_detection.py BLIMMP/Examples/example.domtblout -f domtblout --sigma 1.0 --output example_name -l
