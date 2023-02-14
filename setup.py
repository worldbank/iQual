from setuptools import setup, find_packages

setup(
    name='iQual',
    version='0.1.0',
    description='A package for scaling-up qualitative analysis',
    author='Aditya Karan Chhabra',
    author_email='aditya0chhabra@gmail.com',
    url='https://github.com/worldbank/iQual',
    packages=find_packages(where='src'),
    exclude_package_data={'': ['data','notebooks','docs','tests']},
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'sentence-transformers',
        'spacy',        
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Operating System :: OS Independent'
    ],
)
