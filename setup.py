from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pytides-py3",
    version="0.0.6",
    author="Sangkon Han",
    author_email="sigmadream@gmail.com",
    description="Tidal analysis and prediction library for Python >= 3.10.x",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sigmadream/pytides-py3",
    packages=find_packages(),
    install_requires=["numpy>=1.26", "scipy>=1.14"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    license="MIT",
)
