from setuptools import setup,find_packages,Extension


setup(
    name='GBMApp',
    version='0.1.0',
    packages=find_packages(),
    # Uncomment the following line when C++ files are added
    # ext_modules=ext_modules,
    entry_points={
        'console_scripts': ['gbmapp=src.gui.mainframe:main']
    },
)