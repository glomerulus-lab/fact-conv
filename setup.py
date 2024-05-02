from setuptools import setup, find_packages


import warnings
warnings.filterwarnings("ignore", "The setuptools-version-command package is deprecated")
warnings.filterwarnings("ignore", "easy_install command is deprecated")
warnings.filterwarnings("ignore", "setup.py install is deprecated")


setup(
    name                 = 'Glomerulus',
    version              = '0.0.0.dev0',
    author               = 'Vivian White',
    author_email         = 'vivian.white@mila.quebec',
    classifiers          = [
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    ext_modules          = [],
    python_requires      = '>=3.8.1',
    install_requires     = [
        'torch>=2.0.1',
    ],
    extras_require       = {
        "test": [
            "pytest",
        ],
    },
    packages             = find_packages(include=['glomerulus', 'glomerulus.*']),
)
