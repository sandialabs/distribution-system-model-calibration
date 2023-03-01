# -*- coding: utf-8 -*-
"""
BSD 3-Clause License

Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


 OnlineChangepoint_Script.py
    
Publications related to this method:
    
B. D. Peña, L. Blakely, and M. J. Reno, “Online Data-Driven Detection of Phase Changes in Evolving Distribution Systems,” presented at the ISGT, 2023.
B. D. Peña, L. Blakely, M. J. Reno, “Data-Driven Detection of Phase Changes in Evolving Distribution Systems,” presented at TPEC, 2022.

    
"""


# This script is to run the online phase change detection algorithm on the available data

###############################################################
#   Import Python Libraries

from pathlib import Path
import numpy as np
import pandas as pd

# Import Custom Libraries
import ChangepointUtils as CPUtils
import OnlineChangepointFunctions as OCF

#           End of Imports
###############################################################################

###############################################################################
#
#              Load Data and Set Path Names
#                   
currentDirectory = Path.cwd()
filePath = Path(currentDirectory.parent,'SampleData')
filename = Path(filePath,'ChangePoint_CustIDs.npy')
groundtruthIDs = list(np.load(filename))  # The ID list for customers with true changepoints
filename = Path(filePath,'Changepoint_PhasesTimesteps.npy')
groundtruthTimesteps = np.array(np.load(filename),dtype=int) # The timestep and new phase of the true changepoints
filename = Path(filePath,'Changepoint_voltageData.npy')
voltageInput = np.load(filename) # AMI voltage timeseries for each customer
filename = Path(filePath,'ChangePoint_AllCustIDs.npy')
custIDsInput = list(np.load(filename)) # The customer IDs for all customers
filename = Path(filePath,'Changepoint_phaseLabels.npy')
phaseLabelsInput = np.load(filename) # The original, true phase labels for each customer

# Format ground truth labels as dictionaries
realEventsIDs = {}
realEventsPhases = {}
for custCtr in range(0,len(groundtruthIDs)):
    currCust = groundtruthIDs[custCtr]
    realEventsIDs[currCust] = [groundtruthTimesteps[custCtr,0],]
    realEventsPhases[currCust] = groundtruthTimesteps[custCtr,1]
custIDs = []
for custCtr in range(0,len(custIDsInput)):
    custIDs.append(str(custIDsInput[custCtr]))
saveResultsBasePath = currentDirectory


# Load Time Duration Curve Parameters from file - This would have been generated using the CreateTimeDurationCurve_Script.py
td_curve_path = currentDirectory

# Try to use a newly generated time duration curve parameters (td_curve_params.npy), else use the provided parameters generated using the sample data (td_curve_params_SAMPLE.npy)
try:
    filePath = str(td_curve_path) + '\\td_curve_params.npy'
    td_curve_params = np.load(filePath)    
except:
    filePath = str(td_curve_path) + '\\td_curve_params_SAMPLE.npy'
    td_curve_params = np.load(filePath)     

# Time Duration Curve Definition - using the generated parameters
tdFlatlineCutoff = 0.5
funct = lambda x, a, b, c: a * np.exp(-b * x) + c
td_curve = lambda x : 0.99 if x == 2 else ( tdFlatlineCutoff if funct(x-1, *td_curve_params) <= tdFlatlineCutoff else funct(x-1, *td_curve_params))

###############################################################################
#
#
#      Define Algorithm Parameters
#                   
#               
#

loadDataFlag = False # Flag to load data from a previous run
addNoiseFlag = True # Flag to add measurement noise or not
perNoiseCustomers = 100 # percentage of customers to inject measurement noise into - This should probably be 100
perSTDNoise = 0.07 # percent standard deviation of the injected gaussian measurement nosie
meanValue = 240 # Mean voltage value for the AMI data - used to inject measurement noise 
perMislabeledPhases = 1  # Percentage of customers with injected incorrect phase labels (unrelated to events)
percentMissing = 0.05 #0.1 # Percentage of data per customer which be changed to missing 0.05 is a reasonable default value
minMissingDataInterval = 4 # minimum contiguous missing values 
maxMissingDataInterval = 96 # max contiguous missing values 
kFinal = 6
kVector = [3,6,12,15,30]
windowSize = 384
numInitialWindows = 3
###############################################################################

###############################################################################
#
#                   Run the Online Algorithm
#                   

# Add realistic data concerns to the synthetic sample data
if addNoiseFlag:
    voltageNoise = CPUtils.AddGaussianNoise(voltageInput, perSTDNoise, meanValue, perNoiseCustomers)
    print('Added Gaussian measurement noise to the sample data')
else:
    voltageNoise = deepcopy(voltageInput)

missVoltage = CPUtils.MissingData_VarInt(voltageNoise, percentMissing=percentMissing, minmissingDataInterval=minMissingDataInterval, maxmissingDataInterval=maxMissingDataInterval)
print('Added missing data to the sample data')
newVoltage = CPUtils.ConvertToPerUnit_Voltage(missVoltage)
newVoltage = CPUtils.CalcDeltaVoltage(newVoltage)
print('Converted data into per-unit and delta voltage representation')

if perMislabeledPhases != 0:
    phaseLabelErrors = CPUtils.AddMisLabeledPhases(phaseLabelsInput,perMislabeledPhases)
    print('Injected ' + str(perMislabeledPhases) + ' % mislabeled phases')
else:
    phaseLabelErrors = phaseLabelsInput

print('Beginning phase change detection algorithm')
# Call primary changepoint function
predictionsOverTime, aggMatAll, possibleChangePointsAll, \
    confidenceScoresAll, predictedPhasesAll = OCF.OnlineChangepointFunc(custIDs,newVoltage,phaseLabelErrors,td_curve,)
print('Algorithm Complete')
#Save CSV results file
lastPred = predictionsOverTime[-1]
filePath = str(saveResultsBasePath) + '\\results.csv'
lastPred.to_csv(filePath)
print('Saved results from the last available window into results.csv in the working directory')

#
#               End of Online Changepoint Algorithm
##############################################################################
    
################################################################################
#
#               Results Analysis Section
#   
        
## Analyze results over time
resultsOverTime = OCF.getFP_FN_TP_Analysis(predictionsOverTime, realEventsIDs, windowSize)
timeToFlagged, timeToDecided = OCF.getTime_To_Detection_TP(predictionsOverTime, realEventsIDs,realEventsPhases, windowSize)

# Plot results figures
OCF.PlotFPOverTime_LINE(resultsOverTime,savePath=saveResultsBasePath)

OCF.plotTimeToFlaggedDecided_HIST(timeToFlagged,timeToDecided,savePath=saveResultsBasePath)

OCF.plotTP_Flagged_Decided_LINE(realEventsIDs,predictedPhasesAll.shape[1],windowSize,timeToFlagged,timeToDecided,savePath=saveResultsBasePath)















