from setuptools import setup

setup(
    name='ccaugmentation',
    version='0.1.0',
    packages=['CCAugmentation', 'CCAugmentation.examples'],
    url='',
    license='MIT',
    author='Piotr Juszczyk',
    author_email='piotr.a.juszczyk@gmail.com',
    description='Data preprocessing & augmentation framework, designed for working with crowd counting datasets and ML/DL framework-independent. Supports multitude of simple as well as advanced transformations, outputs and loaders, all of them to be combined using pipelines.',
    install_requires=['numpy==1.19.3', 'opencv-python', 'matplotlib', 'scipy', 'tqdm']
)
