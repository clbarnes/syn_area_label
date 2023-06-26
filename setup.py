from pathlib import Path

from extreqs import parse_requirement_files
from setuptools import find_packages, setup

with open(Path(__file__).resolve().parent / "README.md") as f:
    readme = f.read()

install_requires, extras_require = parse_requirement_files(
    Path(__file__).resolve().parent / "requirements.txt"
)

setup(
    name="syn_area_label",
    url="https://github.com/clbarnes/syn_area_label",
    author="Chris L. Barnes",
    description="Preparing data for labelling synaptic areas using CATMAID annotations",
    long_description=readme,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src", include=["syn_area_label*"]),
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.9, <4.0",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
)
