from setuptools import setup, find_packages

setup(
    name="urdf_to_mjcf",
    version="0.1.0",
    author="Harshavarthan Varatharajan",
    author_email="youremail@example.com",
    url="https://github.com/harshavarthanv/MJURDF-Forge",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "trimesh",
    ],
    entry_points={
        "console_scripts": [
            "urdf_to_mjcf=urdf_to_mjcf.cli:main",
            "mjcf_viewer=urdf_to_mjcf.launch_mujoco:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
    ],
    python_requires=">=3.7",
    include_package_data=True,
    zip_safe=False,
)
