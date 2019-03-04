import setuptools

REQUIRED_PACKAGES = ['fire', 'pandas', 'nltk', 'spacy', 'sklearn']

setuptools.setup(
    name='spooky_author_identification',
    version='0.0.1',
    description='kaggle competition: spooky author identification',
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages()
)
