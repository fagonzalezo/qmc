import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="qmc",
    packages=setuptools.find_packages(),
    version='0.0.1',
    description="Implementation of some machine learning techniques based on density matrices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Fabio A. González et al.",
    author_email="fagonzalezo@unal.edu.co",
    license="GNUv3",
    install_requires=["scipy", "numpy >= 1.19.2", "scikit-learn", "tensorflow >= 2.2.0", "typeguard"],
    python_requires='>=3.6'
)
