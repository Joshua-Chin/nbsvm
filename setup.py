from distutils.core import setup

setup(
    name='NBSVM',
    version='1.0',
    url='https://github.com/Joshua-Chin/nbsvm.git',
    author='Joshua Chin',
    packages = 'nbsvm',
    install_requires = ['scikit-learn', 'numpy']
)
