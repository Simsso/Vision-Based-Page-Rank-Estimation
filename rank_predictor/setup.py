"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

from codecs import open
from setuptools import setup


def get_version():
    with open('version.txt') as ver_file:
        version_str = ver_file.readline().rstrip()
    return version_str


def get_install_requires():
    with open('requirements.txt') as reqs_file:
        reqs = [line.rstrip() for line in reqs_file.readlines()]
    return reqs


setup(name="rank_predictor",
      version=get_version(),
      packages=['rank_predictor', 'rank_predictor.data', 'rank_predictor.data.v1', 'rank_predictor.data.v2',
                'rank_predictor.model', 'rank_predictor.trainer', 'rank_predictor.trainer.ranking'],
      description="Pagerank prediction project",
      install_requires=get_install_requires(),
      author="Timo Denk, Samed Güner",
      zip_safe=False)
