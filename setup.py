from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(
    name='dse-dsmi',
    version='1.0',
    license='MIT',
    author='Chen Liu, Danqi Liao',
    author_email='chen.liu.cl2482@yale.edu',
    packages={'dse_dsmi'},
    description='Diffusion spectral entropy and diffusion spectral mutual information (https://arxiv.org/abs/2312.04823).',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ChenLiu-1996/DiffusionSpectralEntropy',
    keywords='entropy, mutual information, neural network entropy, deep learning entropy',
    install_requires=['numpy', 'scipy', 'matplotlib', 'scikit-learn'],
    classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'License :: Other/Proprietary License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    ],
)