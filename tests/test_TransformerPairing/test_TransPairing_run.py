# Python Library Imports
import pytest
from pathlib import Path
import os

# Package Code
from sdsmc.MeterTransformerPairing import TransformerPairing


# Test Transformer Pairing funct.

def test_transformerPairing_run():
    useTrueLabels = True

    sampleDataDirectory = Path( "../../sdsmc/SampleData/")
    saveResultsPath = Path("./")
    voltageInputPathCSV = Path( sampleDataDirectory, "voltageData_AMI.csv")
    realPowerInputPathCSV = Path( sampleDataDirectory, "realPowerData_AMI.csv")
    custIDInputPathCSV = Path( sampleDataDirectory, "CustomerIDs_AMI.csv")
    reactivePowerInputPathCSV = Path( sampleDataDirectory, "reactivePowerData_AMI.csv")
    transformerLabelsErrorsPathCSV = Path( sampleDataDirectory, 'TransformerLabelsErrors_AMI.csv')

    if useTrueLabels:
        transformerLabelsTruePath = Path(sampleDataDirectory, 'TransformerLabelsTrue_AMI.csv' )
    TransformerPairing.run( voltageInputPathCSV, realPowerInputPathCSV, reactivePowerInputPathCSV, custIDInputPathCSV, transformerLabelsErrorsPathCSV, transformerLabelsTruePath, saveResultsPath, useTrueLabels )

    assert os.path.exists("outputs_ChangedCustomers_M2T.csv") == True
    assert os.path.exists("outputs_ImprovementStats.csv") == True
