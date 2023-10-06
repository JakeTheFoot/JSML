from setuptools import setup, find_packages

VERSION = '0.2.1'
DESCRIPTION = 'Barebones file structure for a Python package'

# Setting up
setup(
    name="JSML",
    version=VERSION,
    author="Jake Silberstein",
    author_email="<jake.silberstein8@gmail.com>",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=[],
    keywords=['python', 'Neural Networks', 'AI', 'CNN',
              'RNN', 'DQN', 'LSTM', 'GRU', 'Transformers', 'Beyesian Optimization'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
