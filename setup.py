# setup.py

import io
import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))


def read_requirements(filename):
    with io.open(os.path.join(here, filename), encoding='utf-8') as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith('#')
        ]


requirements = read_requirements('requirements.txt')
dev_requirements = read_requirements('requirements_developper.txt')
dev_only_requirements = [req for req in dev_requirements if req not in requirements]

setup(
    name='seimei',
    version='0.1.0',
    description='Search-Enhanced Interface for Multi-Expertise Integration',
    #long_description=io.open(os.path.join(here, 'README.md'), encoding='utf-8').read(),
    #long_description_content_type='text/markdown',
    author='Kentaro Seki',
    author_email='seki.kentaro@kyotoai.org',
    url='https://github.com/kyotoai/SEIMEI',
    packages=find_packages(exclude=('tests',)),
    install_requires=requirements,
    extras_require={
        'dev': dev_only_requirements,
    },
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    entry_points={
        'console_scripts': [
            'seimei = seimei.cli:main',
        ],
    },
)
