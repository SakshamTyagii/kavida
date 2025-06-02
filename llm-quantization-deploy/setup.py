from setuptools import setup, find_packages

setup(
    name='llm-quantization-deploy',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A project for quantizing and deploying large language models locally.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # List your project dependencies here
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)