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
    version='0.1.1',
    description='iQual is a package that leverages natural language processing to scale up interpretative qualitative analysis. It also provides methods to assess the bias, interpretability and efficiency of the machine-enhanced codes.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Aditya Karan Chhabra',
    author_email='aditya0chhabra@gmail.com',
    url='https://github.com/worldbank/iQual',
    maintainer='Aditya Karan Chhabra',
    maintainer_email='aditya0chhabra@gmail.com',
    packages=find_packages(where='src'),
    exclude_package_data={'': ['data','notebooks','docs','tests']},
    license='MIT',
    license_files=('LICENSE.md',),
    package_dir={'': 'src'},
    install_requires=required,
    python_requires='>=3.7',
    keywords='nlp natural-language-processing qualitative-analysis human-coding qualitative-research',
    classifiers=[
        'Development Status :: 3 - Alpha',
        "Intended Audience :: Science/Research",
        "Topic :: Sociology",
        "Topic :: Utilities",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Information Analysis",
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
