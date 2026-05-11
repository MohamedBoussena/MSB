from setuptools import setup, find_packages

setup(
    name="msb_survival",
    version="0.1.0",
    author="Your Name",
    description="Multimodality Stacking with Blockwise Missing Values for Survival Analysis",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "scikit-survival",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)