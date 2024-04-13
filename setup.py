# Responsible for creating my application as a package. Can also deploy on PyPi
# automatically find packages from directory
from setuptools import find_packages, setup
from typing import List

hyphen_e_dot = '-e .'

def get_requirements(file_path: str)->List[str]:
    """
    This function will return the list of requirements
    """
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        [req.replace ('\n', '')for req in requirements]
        
        if hyphen_e_dot in requirements:
            requirements.remove(hyphen_e_dot)
    
    return requirements


# Create the params of the project:
setup(
name='CPT Rent Price Prediction',
version='0.0.1',
author='Mikey Lamb',
packages=find_packages(),
install_requires=get_requirements('requirements.txt'),
)