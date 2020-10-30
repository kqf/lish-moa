from setuptools import setup, find_packages

setup(
    name="lish-moa",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'lish-moa=mlp.baseine:main',
        ],
    },
)
