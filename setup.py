from setuptools import setup, find_packages

setup(
    name='astroca',
    version='0.1.0',
    packages=find_packages(include=["astroca", "astroca.*"]),
    install_requires=[
        "numpy>=1.20",
        "tifffile",
        "scikit-image",
        "napari[all]",
        "joblib",
        "numba",
        "scipy",
        "matplotlib",
        "configparser",
    ],
    author='Audigier matteo',
    description='Segmentation et analyse d\'astrocytes en 3D+temps',
    license='MIT',
)
