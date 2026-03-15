from setuptools import setup, find_packages

setup(
    name="yam_teleop",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "pyzmq",
        "opencv-python",
        "numpy",
        "pyyaml",
    ],
)
