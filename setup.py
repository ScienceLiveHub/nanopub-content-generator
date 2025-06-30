from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nanopub-content-generator",
    version="1.0.0",
    author="Science Live Team",
    author_email="support@sciencelive.com",
    description="AI-powered content generation engine for nanopublications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ScienceLiveHub/nanopub-content-generator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Text Processing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    
    # package data configuration
    package_data={
        "": ["templates/*.json"],  # Include templates from root
    },
    
    # ALTERNATIVE: Use MANIFEST.in approach
    # Create a MANIFEST.in file with: recursive-include templates *.json
    
    entry_points={
        "console_scripts": [
            "nanopub-generate=nanopub_content_generator.cli:main",
        ],
    },
)
