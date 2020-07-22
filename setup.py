from setuptools import setup, find_packages
from setuptools.command.install import install

setup(name='voduct',
      packages=find_packages(),
      version="0.1.0",
      description='A sequence reduction system',
      author='Satchel Grant',
      author_email='grantsrb@stanford.edu',
      url='https://github.com/grantsrb/voduct.git',
      install_requires= ["numpy",
                         "matplotlib",
                         "torch",
                         "tqdm",
                         "psutil"],
      py_modules=['voduct'],
      long_description='''
          This is a package to create and use a combination method of
          transformers with convolutions to reduce sequences to single
          embeddings
          ''',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: MacOS :: MacOS X :: Ubuntu',
          'Topic :: Scientific/Engineering :: Information Analysis'],
      )
