# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="RISynG",
    version="1.0.1s",
    author="xfurna",
    author_email="architdwivedi.off@gmail.com",
    description="---",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xfurna/RISynG/",
    packages=["risyng"],
    license="GPLv3",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    entry_points={"console_scripts": ["risyng=risyng.__main__:main",]},
)
