from src.features_extraction.services import DataService, CalculatorService

"""
Welcome to the feature calculator!
This component is responsible for extracting and calculating features_extraction from the data.
Multiple cmd parameters are optional in order to determine the calculator mode:
    * Blinks    - calculate only blinks related features_extraction
    * Fixations - calculate only fixations related features_extraction
    * Saccades  - calculate only saccades related features_extraction
    * RoI       - calculate only RoI related features_extraction
    * Distance  - calculate only distance related features_extraction
    * All       - calculate all the features_extraction
    
(Important! Directories and other key parameters are set in the config file)
"""


def run(args):
    DataService.init()

    CalculatorService.calculate(args, write=True)


run(['all'])
