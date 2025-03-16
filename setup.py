from setuptools import setup, find_packages

setup(
    
    name = 'gradientblueprint',
    version = '1.0.0',
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
    author_email='rey.mhmmd@gmail.com',
    description='This is a comprehensive package designed to demystify the concepts of gradients and their applications across various machine learning algorithms. With this release, developers and researchers gain access to a powerful toolkit that simplifies the understanding and implementation of gradient-based optimization techniques.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/raminmohammadi/GredientBlueprint',
    classifiers=[
        'Programming Language :: Python :: 3',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.9',
)
