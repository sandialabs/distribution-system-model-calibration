# Distribution System Model Calibration Algorithms

### Description

The code in this library was developed by Sandia National Laboratories under funding provided by the U.S. Department of Energy Solar Energy Technologies Office as part of the project titled “Physics-Based Data-Driven grid Modelling to Accelerate Accurate PV Integration” Agreement Number 34226. More information about the project can be found at [https://www.osti.gov/biblio/1855058.](https://www.osti.gov/biblio/1855058)

Five distribution system model calibration algorithms are included in this release. There are two algorithms for performing phase identification: one based on an ensemble spectral clustering approach and one based on leveraging additional sensors placed on the medium voltage. Start with the CA_Ensemble_SampleScripts.py file and the SensorMethod_SampleScript.py file, respectively. The third and fourth algorithms identify the connection between service transformers and low-voltage customers (meter to transformer pairing algorithm).  The algorithms have different data requirements, P/Q/V/Labels and P/V/Labels/Distance respectively. Start with the MeterToTransPairingScripts.py file.  The fifth algorithm provides an online customer phase change detection algorithm.  Start with CreateTimeDurationCurve_Script.py.

There is also a sample dataset included to facilitate the use of the code. The dataset will load automatically when using one of the sample scripts provided. For more details, please see Section IV.

For more detailed documentation, please see the DistributionSystemModelCalbrationAlgorithmManual.pdf included in the repository.

This code and data are released without any guarantee of robustness under conditions different from the ones tested during development.  Contributors include Logan Blakely, Matthew J. Reno, and Bethany D. Peña.

Questions or inquiries can be directed to Logan Blakely (Sandia National Laboratories) at lblakel@sandia.gov.

### Installation and Usage

To install:

```sh
python -m pip install git+https://github.com/sandialabs/distribution-system-model-calibration
```

To execute, for example, the Meter Transformer Pairing analytic:

```python
import sdsmc

from sdsmc.TransformerPairing.MeterTransformerPairing import transformerPairing

transformerPairing(
    voltageData_AMI,
    realPowerData_AMI,
    reactivePowerData_AMI,
    customerIDs_AMI,
    transLabelsErrors_csv,
    transLabelsTrue_csv,
    saveResultsPath,
    useTrueLabels=True)
```

The first 4 arguments require AMI data formatted as a .CSV file. The AMI data of each row is aligned to the customers row in the customerIDs_AMI CSV file.
Ex: customer_0 in row_0 of customerIDs_AMI
    row_0 in voltageData_AMI, realPowerData_AMI, and reactivePowerData_AMI include comma separate data values.
    
Note: If useTrueLabels is true, there will be an additional output file that shows improvement statistics

For more information, check out the test data in the OMF for working examples: https://github.com/dpinney/omf/tree/master/omf/static/testFiles/transformerPairing

Please also for additional clarification of the formatting of the first 4 arguments containing CSV AMI data, check out the Open Modeling Framework Wikipedia page that uses this function here: https://github.com/dpinney/omf/wiki/Models-~-transformerPairing
