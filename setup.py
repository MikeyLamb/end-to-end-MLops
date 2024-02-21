from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    this function returns the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj: #open file path with temporary object
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements] #replace the \n line break with a blank

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements


setup(
name='end-to-end-mlops',
version='0.0.1',
author='MikeyLamb',
author_email='mslambrecht@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')

)