# __init__.py

# This file indicates that the directory should be treated as a package.

# You can import necessary modules or packages here
# For example:
# from .module_name import ClassName

# You can also define package-level variables or functions
__version__ = "1.0.0"
__package_name__ = "constants"

# Initialize package-level resources if needed
def initialize_package():
    print(f"Initializing Package: {__package_name__} v{__version__}")
    

## Automatic initialization
initialize_package()