try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'DeepScite - A Simple Convolutional-based Recommendation Model',
    'author': 'Noon van der Silk',
    'author_email': 'noonsilk+deepscite@gmail.com',
    'version': '0.2',
    'packages': ['deepscite'],
    'scripts': [],
    'name': 'deepscite'
}

setup(**config)
