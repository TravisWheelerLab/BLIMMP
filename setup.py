from setuptools import setup, find_packages

setup(
    name="BLIMMP",
    version="0.1.3",
    author="Neha Sontakke",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5",
        "numpy>=1.23",
        "numba",
        "scipy",
    ],
    package_data={
        "BLIMMP_Scripts": [
            # Module graphs, shipped zipped and extracted on first use.
            # module_detection.py hard-fails if this is missing, so a rename
            # here can no longer silently produce all-zero results.
            "Graph_Dependencies/KEGG_Graphs_Generated_March26.zip",
            "Graph_Dependencies/*.json",
            "Graph_Dependencies/MODULE_ALL_NEIGHBOR_DATA/*.json",
            "Data_Dependencies/*.txt",
            "Data_Dependencies/*.json",
            "Data_Dependencies/ATB_Taxonomy_Frequency/*.tsv",
        ]
    },
    include_package_data=True,
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "BLIMMP=BLIMMP_Scripts.module_detection:main",
        ],
    },
)
