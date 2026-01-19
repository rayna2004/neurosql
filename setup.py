"""
Setup script for the NeuroSQL package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read version
__version__ = "1.0.0"

# Read README if exists
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "NeuroSQL - A Knowledge Graph System with Multiple Abstraction Layers"

# Core dependencies
install_requires = [
    "networkx>=3.0",
    "matplotlib>=3.5",
    "flask>=2.0",
    "requests>=2.28",
]

# Optional dependencies
extras_require = {
    "data": [
        "pandas>=1.3.0",
        "numpy>=1.21.0",
    ],
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=22.0.0",
        "flake8>=5.0.0",
    ],
    "full": [
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "plotly>=5.0.0",
        "pytest>=7.0.0",
    ]
}

setup(
    name="neurosql",
    version=__version__,
    author="NeuroSQL Team",
    description="A Knowledge Graph System with Multiple Abstraction Layers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    package_data={
        "": ["templates/*"],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Database",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    keywords=[
        "knowledge-graph",
        "graph-database",
        "semantic-web",
        "artificial-intelligence",
    ],
)