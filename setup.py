import os

from setuptools import find_packages, setup
from CCAugmentation import __title__ as TITLE, __version__ as VERSION, __author__ as AUTHOR, __license__ as LICENSE


def _parse_requirements():
    lines = [line.strip() for line in open('requirements.txt')]
    return [line for line in lines if line and not line.startswith('#')]


REQUIREMENTS = _parse_requirements()

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(DIR_PATH, 'README.md'), 'r') as f:
    README = f.read()

setup(
    name=TITLE.lower(),
    version=VERSION,
    packages=find_packages(),
    url='https://github.com/pijuszczyk/CCAugmentation',
    license=LICENSE,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    author=AUTHOR,
    author_email='piotr.a.juszczyk@gmail.com',
    description='Data preprocessing & augmentation framework, designed for '
                'working with crowd counting datasets and ML/DL '
                'framework-independent. Supports multitude of simple '
                'as well as advanced transformations, outputs and loaders, '
                'all of them to be combined using pipelines.',
    long_description=README,
    long_description_content_type='text/markdown',
    install_requires=REQUIREMENTS
)
