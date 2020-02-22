from setuptools import setup
import os


setup(
   name='beat_pd',
   version='1.0',
   description='Utils module used in BeatPD Competition',
   author='Hung V. Tran',
   author_email='tvhung@selab.hcmus.edu.vn',
   packages=['beat_pd'],  #same as name
   install_requires=[
       'pandas', 
        'numpy',
        'torch',
        
    ], #external packages as dependencies
)