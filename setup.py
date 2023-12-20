from pathlib import Path

from setuptools import setup, find_packages #type:ignore


# read the contents of your README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(
    name="StressAnalysisTool",
    version="0.0.1",
    author="A.H",
    description="A Stress Analysis Tool for a Laminated Composites",
    long_description=long_description,
    ext_modules=[],
    zip_safe=False,
    python_requires=">=3.8.0",
    packages=find_packages('.'),
    package_dir={'': '.'},
    long_description_content_type="text/markdown",
    entry_points={
        'console_scripts': ['stressAnalysisTool=main:run']
                         },
    install_requires=["streamlit==1.28.2", "numpy==1.26.2"],
)