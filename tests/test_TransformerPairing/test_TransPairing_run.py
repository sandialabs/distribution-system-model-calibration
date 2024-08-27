# Python Library Imports
import unittest
from pathlib import Path
import os

# Package Code
from sdsmc.MeterTransformerPairing import TransformerPairing


# Test Transformer Pairing funct.

class TestingSDSMC( unittest.TestCase ):

    def test_transformerPairing_run( self ):
        useTrueLabels = True

        # Relative pathing setup
        currentDirectory = Path(__file__).parent.resolve()
        sampleDataDirectory = Path( currentDirectory.parent.parent, 'sdsmc/SampleData')
        saveResultsPath = currentDirectory

        # Get sample data
        voltageInputPathCSV = Path( sampleDataDirectory, "voltageData_AMI.csv")
        realPowerInputPathCSV = Path( sampleDataDirectory, "realPowerData_AMI.csv")
        custIDInputPathCSV = Path( sampleDataDirectory, "CustomerIDs_AMI.csv")
        reactivePowerInputPathCSV = Path( sampleDataDirectory, "reactivePowerData_AMI.csv")
        transformerLabelsErrorsPathCSV = Path( sampleDataDirectory, 'TransformerLabelsErrors_AMI.csv')

        if useTrueLabels:
            transformerLabelsTruePath = Path(sampleDataDirectory, 'TransformerLabelsTrue_AMI.csv' )
        TransformerPairing.run( voltageInputPathCSV, realPowerInputPathCSV, reactivePowerInputPathCSV, custIDInputPathCSV, transformerLabelsErrorsPathCSV, transformerLabelsTruePath, saveResultsPath, useTrueLabels )

        # Test all necessary files are there and under the right name.

        self.assertTrue( Path( currentDirectory, "outputs_ChangedCustomers_M2T.csv").exists() )
        self.assertTrue( Path( currentDirectory, "outputs_ImprovementStats.csv").exists() )

        # Test some values as a sanity check to ensure the calculations were correct.

if __name__ == '__main__':
    unittest.main()