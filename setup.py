import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="qmc",
    packages=setuptools.find_packages(),
    version='0.0.1',
    description="Machine learning techniques based on density matrices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Fabio A. González et al.",
    author_email="fagonzalezo@unal.edu.co",
    license="GNUv3",
    install_requires=["numpy", "scipy", "scikit-learn", "tensorflow >= 2.2.0", "typeguard"],
    python_requires='>=3.6'
)