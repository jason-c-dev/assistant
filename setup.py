#!/usr/bin/env python3
"""
Setup script for Personal Assistant CLI package.
"""

from setuptools import setup, find_packages
import os
import sys

# Ensure we're using Python 3.10+
if sys.version_info < (3, 10):
    sys.exit("Python 3.10+ is required for this package.")

# Read version from package
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'assistant', '__init__.py')
    with open(version_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    raise RuntimeError("Unable to find version string.")

# Read long description from README
def get_long_description():
    readme_file = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_file):
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements from requirements.txt
def get_requirements():
    requirements_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = []
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith('#'):
                # Skip development dependencies for production install
                if not any(dev_dep in line for dev_dep in ['pytest', 'black', 'flake8', 'mypy']):
                    requirements.append(line)
        return requirements

setup(
    name="personal-assistant-cli",
    version=get_version(),
    author="Assistant Development Team",
    author_email="dev@assistant.local",
    description="A memory-enabled AI assistant CLI using AWS Strands Agent SDK and MCP",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/personal-assistant-cli",
    packages=find_packages(exclude=['tests*']),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Utilities",
    ],
    python_requires=">=3.10",
    install_requires=get_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0", 
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "assistant=assistant.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "assistant": ["*.yaml", "*.yml"],
    },
    zip_safe=False,
    keywords="ai assistant cli memory mcp strands",
    project_urls={
        "Bug Reports": "https://github.com/your-org/personal-assistant-cli/issues",
        "Source": "https://github.com/your-org/personal-assistant-cli",
        "Documentation": "https://github.com/your-org/personal-assistant-cli#readme",
    },
) 