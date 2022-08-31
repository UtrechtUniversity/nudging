# based on https://github.com/pypa/sampleproject - MIT License
from pathlib import Path
from setuptools import setup, find_packages

readme_path = Path(__file__).parent.absolute() / "README.md"
with open(readme_path, "r", encoding="utf-8") as f:
    long_description = f.read()


setup(
    name='nudging',
    version='0.1.0',
    author='Utrecht University - Research Engineering team',
    description='Precision nudging package',
    long_description=long_description,
    packages=find_packages(exclude=['data', 'docs', 'tests', 'examples']),
    python_requires='~=3.7',
    install_requires=[
        "pandas",
        "numpy",
        "sklearn",
        "scipy",
        "seaborn",
        "pyreadstat",
        "causalinference",
        "tqdm",
        "PyYAML",
    ]
)
