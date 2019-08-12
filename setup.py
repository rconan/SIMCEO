from distutils.core import setup

setup(
    author = 'R. Conan',
    author_email = 'rconan@gmto.org',
    url ='https://github.com/rconan/SIMCEO/tree/python-client',
    name='DOS',
    description='GMT Dynamic Optical Simulation Client/Server application',
    version='0.1dev',
    packages=['dos',],
    long_description=open('README.md').read(),
)
