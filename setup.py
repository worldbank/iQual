from setuptools import setup, find_packages
# Read requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()
# Read README.md
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

# Setup
setup(
    name='iQual',
    version='0.1.0',
    description='Iterative Qualitative Analysis - with Human Coding',
    long_description=long_description,
    author='Aditya Karan Chhabra',
    author_email='aditya0chhabra@gmail.com',
    url='https://github.com/worldbank/iQual',
    maintainer='Aditya Karan Chhabra',
    maintainer_email='aditya0chhabra@gmail.com',
    packages=find_packages(where='src'),
    exclude_package_data={'': ['data','notebooks','docs','tests']},
    package_dir={'': 'src'},
    install_requires=required,
    python_requires='>=3.7',
    keywords='natural-language-processing qualitative-analysis world-bank human-coding interpretative-coding qualitative-data-analysis qualitative-research nlp',
   classifiers=[
        'Development Status :: 3 - Alpha',
        "Intended Audience :: Science/Research",
        "Topic :: Sociology",
        "Topic :: Economics",
        "Topic :: Utilities",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Information Analysis",
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
