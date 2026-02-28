"""Setup script for Sharingan package."""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Sharingan: Semantic Video Understanding with Temporal Reasoning"

setup(
    name="sharingan-core",
    version="3.0.0",
    author="S Khavin",
    author_email="",
    description="Semantic Video Understanding with Temporal Reasoning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/skhavindev/sharingan",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="video understanding, computer vision, temporal reasoning, VLM, CLIP, semantic search",
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "Pillow>=8.0.0",
        "tqdm>=4.60.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "smolvlm": [
            "transformers>=4.35.0",
            "accelerate>=0.20.0",
        ],
        "chat": [
            "transformers>=4.35.0",
            "bitsandbytes>=0.41.0",
            "accelerate>=0.20.0",
        ],
        "all": [
            "transformers>=4.35.0",
            "bitsandbytes>=0.41.0",
            "accelerate>=0.20.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sharingan-core=sharingan.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
