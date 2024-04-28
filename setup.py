from setuptools import setup, find_packages

setup(
    
    name = 'AUTOGRAD',
    version = '0.1.0',
    packages = find_packages(),
    install_requires = [
        'numpy>=1.18.0',
        'requests>=2.25.1',
        'matplotlib>=3.3.0',
        'pandas>=1.0.0',
        'graphviz',
        'scikit-learn'
    ],
    author='Ramin Mohammadi',
    author_email='r.mohammadi@northeastern.edu',
    description='This package is ',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/your_package',
    classifiers=[
        'Programming Language :: Python :: 3',
        # Add more classifiers as needed
    ],
    
    
)