from setuptools import setup, find_packages
from typing import List

def read_requirements(file_path: str) -> List[str]:
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

setup(
    name="mlops",
    version="0.1",
    packages=find_packages(),
    install_requires=read_requirements("requirements.txt"),
)