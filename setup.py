import os

from setuptools import setup

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(DIR_PATH, "README.md"), 'r') as f:
    README = f.read()

setup(
    name='ccaugmentation',
    version='0.1.0',
    packages=['CCAugmentation', 'CCAugmentation.examples'],
    url='https://github.com/pijuszczyk/CCAugmentation',
    license='MIT',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    author='Piotr Juszczyk',
    author_email='piotr.a.juszczyk@gmail.com',
    description='Data preprocessing & augmentation framework, designed for working with crowd counting datasets and ML/DL framework-independent. Supports multitude of simple as well as advanced transformations, outputs and loaders, all of them to be combined using pipelines.',
    long_description=README,
    long_description_content_type='text/markdown',
    install_requires=['numpy==1.19.3', 'opencv-python', 'matplotlib', 'scipy', 'tqdm']
)
