from setuptools import setup, find_packages

setup(
    name='bdtpy',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'ipywidgets',
        'sdeint'
    ]
)