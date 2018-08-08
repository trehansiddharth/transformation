from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name="transformation",
	version="0.1",
	description="3D transformations library for working with different coordinate systems",
    long_description=readme(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
	url="https://github.com/trehansiddharth/transformation",
	author="Siddharth Trehan",
    author_email="trehans@mit.edu",
	license="MIT",
	packages=["transformation"],
	install_requires=[
            "numpy",
            "numpy-quaternion",
            "scipy"
	],
	zip_safe=False)
