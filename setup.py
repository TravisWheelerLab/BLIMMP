from setuptools import setup, find_packages

setup(
    name="BLIMMP",
    version="0.1.0",
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
            "Graph_Dependencies/KEGG_Module_Graphs.zip",
            "Graph_Dependencies/KEGG_Module_Equations_Jan26.json",
            "Graph_Dependencies/module_ko_reaction.json",
            "Graph_Dependencies/MODULE_ALL_NEIGHBOR_DATA/*",
            "Data_Dependencies/ATB_Taxonomy_Frequency/*",
            "Data_Dependencies/ko_list.txt",
            "Data_Dependencies/module_freq.txt",
        ]
    },
    include_package_data=True,
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "BLIMMP=BLIMMP_Scripts.module_detection:main",
        ],
    },
)
