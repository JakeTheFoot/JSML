from setuptools import setup, find_packages

VERSION = '0.2.15'
DESCRIPTION = 'Non-Functional Machine Learning Library'

# Setting up
setup(
    name="JSML",
    version=VERSION,
    author="Jake Silberstein",
    author_email="jake.silberstein8@gmail.com",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'matplotlib',
        'joblib',
        'opencv-python',
        'gym',
        'python-abc',
        'scikit-optimize',
        'scikit-learn'
    ],
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
