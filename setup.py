"""Setup script for content-vector."""

from setuptools import setup, find_packages

setup(
    name="content-vector",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "openai-whisper>=20231117",
        "torch>=2.0.0",
        "chromadb>=0.4.22",
        "openai>=1.0.0",
        "anthropic>=0.18.0",
        "click>=8.1.0",
        "rich>=13.0.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
    ],
    entry_points={
        "console_scripts": [
            "cvector=src.cli:main",
        ],
    },
    python_requires=">=3.9",
)
