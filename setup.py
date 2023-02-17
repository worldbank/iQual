from setuptools import setup, find_packages
# Read requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()

# Setup
setup(
    name='iQual',
    version='0.1.0',
    description='Iterative Qualitative Analysis - with Human Coding',
    long_description='A Python package that allows qualitative analysis of open-ended interviews at scale, by extending a small set of interpretative human-codes to a much larger set of documents using natural language processing.',    
    author='Aditya Karan Chhabra',
    author_email='aditya0chhabra@gmail.com',
    url='https://github.com/worldbank/iQual',
    packages=find_packages(where='src'),
    exclude_package_data={'': ['data','notebooks','docs','tests']},
    package_dir={'': 'src'},
    install_requires=required,
    python_requires='>=3.7',
    keywords='natural-language-processing qualitative-analysis world-bank human-coding interpretative-coding qualitative-data-analysis qualitative-research nlp',
   classifiers=[
        'Development Status :: 3 - Alpha',
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Sociology",
        "Topic :: Economics",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Utilities",
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
