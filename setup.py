from setuptools import setup, find_packages

setup(
    name='acas-v2',
    version='0.1.0',
    description='A Gymnasium environment package of Acas-Xu',
    author='Walid Abouir',
    author_email='walid.abouir@onera.fr',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
    ],
)
