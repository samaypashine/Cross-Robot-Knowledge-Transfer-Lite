from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='paper5',
      version='1.0',
      packages=find_packages(),
      install_requires=required,
      author='Gyan Tatiya',
      author_email='Gyan.Tatiya@tufts.edu',
      description='Tool dataset experiments',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/gtatiya/paper5',
      license='MIT License',
      python_requires='>=3.6',
      )
