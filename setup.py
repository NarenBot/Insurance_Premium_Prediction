from setuptools import setup, find_packages
import os


def get_requirements(file_path):
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements


setup(
    name="Insurance_Premium_Prediction",
    version="0.02",
    description="Build a solution that should able to predict the premium of the personal for health insurance.",
    author="NarenBot",
    author_email="narendas10@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
