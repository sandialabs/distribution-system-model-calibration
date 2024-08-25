# Python Library Imports
import unittest
from pathlib import Path
import os

# Package Code
from sdsmc.PhaseIdentification import PhaseIdentification_CAEnsemble


# Test Phase Identification funct.

class TestingSDSMC( unittest.TestCase ):

		def test_phaseID_run( self ):
			useTrueLabels = True

			# Relative pathing setup
			currentDirectory = Path(__file__).parent.resolve()
			sampleDataDirectory = Path( currentDirectory.parent.parent, 'sdsmc/SampleData')
			saveResultsPath = currentDirectory

			# Get sample data
			sampleInputPathCSV = Path( sampleDataDirectory, 'phaseID_singleFile_AMI.csv')
															
			phaseLabelsTrue = Path( sampleDataDirectory, 'PhaseLabelsTrue_AMI.csv')
			numPhases = Path( sampleDataDirectory, 'NumPhases.csv')

			PhaseIdentification_CAEnsemble.run( sampleInputPathCSV, phaseLabelsTrue_csv=phaseLabelsTrue, numPhases_csv=numPhases, saveResultsPath=Path(currentDirectory, 'outputs_phaseID_test.csv') )
			# Test all necessary files are there and under the right name.

			self.assertTrue( Path( currentDirectory, "ModifiedSC_HIST.png").exists() )
			self.assertTrue( Path( currentDirectory, "outputs_phaseID_test.csv").exists() )

			# Test some values as a sanity check to ensure the calculations were correct.

if __name__ == '__main__':
    unittest.main()