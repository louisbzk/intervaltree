#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
intervaltree: A mutable, self-balancing interval tree for Python 2 and 3.
Queries may be by point, by range overlap, or by range envelopment.

Distribution logic

Note that "python setup.py test" invokes pytest on the package. With appropriately
configured setup.cfg, this will check both xxx_test modules and docstrings.

Copyright 2013-2023 Chaim Leib Halbert

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import io
import os
import subprocess
from sys import exit

from setuptools import setup
from setuptools.command.test import test as TestCommand


def prereleaser_set_commit_msg(data):
    """Override the default commit message for prerelease."""
    data["commit_msg"] = "[^] Preparing release: %(new_version)s"


def postrealeaser_set_commit_msg(data):
    """Override the default commit message for postrelease."""
    data["commit_msg"] = "[^] Back to development: %(new_version)s"


## CONFIG
target_version = "3.2.0"


def version_info(target_version):
    is_dev_version = "PYPI" in os.environ and os.environ["PYPI"] == "pypitest"
    if is_dev_version:
        p = subprocess.Popen("git describe --tag".split(), stdout=subprocess.PIPE)
        git_describe = str(p.communicate()[0]).strip()
        release, build, commitish = git_describe.split("-")
        version = "{0}a{1}".format(target_version, build)
    else:  # This is a RELEASE version
        version = target_version
    return {
        "is_dev_version": is_dev_version,
        "version": version,
        "target_version": target_version,
    }


vinfo = version_info(target_version)
if vinfo["is_dev_version"]:
    pass
else:
    pass

with io.open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


## PyTest
# This is a plug-in for setuptools that will invoke py.test
# when you run python setup.py test
class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest  # import here, because outside the required eggs aren't loaded yet

        exit(pytest.main(self.test_args))


## Run setuptools
setup(
    long_description=long_description,
    long_description_content_type="text/markdown",
    download_url="https://github.com/chaimleib/intervaltree/tarball/{version}".format(
        **vinfo
    ),
    zip_safe=True,
)
