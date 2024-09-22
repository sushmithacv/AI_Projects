from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    HYPHEN_E_DOT = '-e .'
    requirements = []
    
    # Open the file and read its contents
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
    
        # Remove '-e .' if present
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)  # Fix the removal of '-e .'
    
    return requirements

# Setup configuration
setup(
    name="AI_project",  # Name of your package
    version="0.1.0",  # Initial release version
    author="Sushmitha",
    author_email="cvsushi14@gmail.com",
    
    packages=find_packages(),  # Automatically find packages
    
    install_requires=get_requirements('requirements.txt')
)
