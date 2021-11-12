# Distribution System Model Calibration Algorithms

The code in this library was developed by Sandia National Laboratories under funding provided by the U.S. Department of Energy Solar Energy Technologies Office as part of the project titled “Physics-Based Data-Driven grid Modelling to Accelerate Accurate PV Integration” Agreement Number 34226. More information about the project and resulting publications can be found at https://www.researchgate.net/project/Intelligent-Model-Fidelity-IMoFi.

Three distribution system model calibration algorithms are included in this release. There are two algorithms for performing phase identification: one based on an ensemble spectral clustering approach and one based on leveraging additional sensors placed on the medium voltage. Start with the CA_Ensemble_SampleScripts.py file and the SensorMethod_SampleScript.py file, respectively. The third algorithm identifies the connection between service transformers and low-voltage customers (meter to transformer pairing algorithm). Start with the MeterToTransPairingScripts.py file.

There is also a sample dataset included to facilitate the use of the code. The dataset will load automatically when using one of the sample scripts provided. For more details, please see Section IV.

For more detailed documentation, please see the DistributionSystemModelCalbrationAlgorithmManual.pdf included in the repository.


This code and data are released without any guarantee of robustness under conditions different from the ones tested during development.
Questions or inquiries can be directed to Logan Blakely (Sandia National Laboratories) at lblakel@sandia.gov.






