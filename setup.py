#!/usr/bin/env python

from setuptools import setup
import subprocess
import os


def write_version_file(version):
    """ Writes a file with version information to be used at run time

    Parameters
    ----------
    version: str
        A string containing the current version information

    Returns
    -------
    version_file: str
        A path to the version file

    """
    try:
        git_log = subprocess.check_output(
            ['git', 'log', '-1', '--pretty=%h %ai']).decode('utf-8')
        git_diff = (subprocess.check_output(['git', 'diff', '.']) +
                    subprocess.check_output(
                        ['git', 'diff', '--cached', '.'])).decode('utf-8')
        if git_diff == '':
            git_status = '(CLEAN) ' + git_log
        else:
            git_status = '(UNCLEAN) ' + git_log
    except Exception as e:
        print("Unable to obtain git version information, exception: {}"
              .format(e))
        git_status = ''

    version_file = '.version'
    if os.path.isfile(version_file) is False:
        with open('src/' + version_file, 'w+') as f:
            f.write('{}: {}'.format(version, git_status))

    return version_file


# get version info from __init__.py
def readfile(filename):
    with open(filename) as fp:
        filecontents = fp.read()
    return filecontents


VERSION = '0.0.1'
version_file = write_version_file(VERSION)
long_description = ''

setup(name='frodo',
      description='',
      long_description=long_description,
      url='',
      author='Greg Ashton',
      author_email='greg.ashton@monash.edu',
      license="",
      version=VERSION,
      packages=['frodo'],
      package_dir={'frodo': 'src'},
      package_data={'frodo': [version_file]},
      install_requires=[
          'future',
          'dynesty',
          'corner',
          'numpy>=1.9',
          'matplotlib>=2.0',
          'pandas',
          'scipy',
          'bilby'],
      entry_points={'console_scripts':
                    ['frodo_single_pulse=frodo.single_pulse:main']
                    },
      classifiers=[
          "Programming Language :: Python :: 3.7",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent"])
