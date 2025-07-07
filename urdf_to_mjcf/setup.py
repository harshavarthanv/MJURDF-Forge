from setuptools import setup, find_packages

setup(
    name="urdf_to_mjcf",
    version="0.1.0",
    description="Convert URDF robot descriptions to MuJoCo MJCF XML format",
    author="Harshavarthan Varatharajan",
    packages=find_packages(),
    entry_points={"console_scripts": ["urdf_to_mjcf=urdf_to_mjcf.cli:main"]},
    python_requires=">=3.7",
)
