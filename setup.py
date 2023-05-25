from setuptools import setup, find_packages

# Guide here: https://godatadriven.com/blog/a-practical-guide-to-using-setup-py/

with open("requirements.txt", "r") as fh:
    requirements_txt = fh.read().splitlines()

setup(
    name='dss_toolkit',
    version='0.1.0',
    description='Library for commonly used code for Data Science Experiments',
    author='Joseph Benjamin Del Mundo',
    author_email='jbdelmundo@gmail.com',
    packages=find_packages(include=['dss_toolkit', 'dss_toolkit.*']),
    
    install_requires=requirements_txt # Specify required libraries: pkg, pkg ==version or pkg>=version
    # tests
)