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


 CreateTimeDurationCurve_Script.py

Publications related to this method:
    
B. D. Peña, L. Blakely, and M. J. Reno, “Online Data-Driven Detection of Phase Changes in Evolving Distribution Systems,” presented at the ISGT, 2023.
B. D. Peña, L. Blakely, M. J. Reno, “Data-Driven Detection of Phase Changes in Evolving Distribution Systems,” presented at TPEC, 2022.

    
"""


###############################################################
#   Import Python Libraries

from pathlib import Path
import numpy as np

# Import Custom Libraries
if __package__ in [None, '']:
    import ChangepointUtils as CPUtils
    import OnlineChangepointFunctions as OCF
else:
    from . import ChangepointUtils as CPUtils
    from . import OnlineChangepointFunctions as OCF

###############################################################################
#
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
voltageInputAll = np.load(filename) # AMI voltage timeseries for each customer
filename = Path(filePath,'ChangePoint_AllCustIDs.npy')
custIDs = list(np.load(filename)) # The customer IDs for all customers
filename = Path(filePath,'Changepoint_phaseLabels.npy')
phaseLabelsInputAll = np.load(filename) # The original, true phase labels for each customer

# Format ground truth labels as dictionaries
realEventsIDs = {}
realEventsPhases = {}
for custCtr in range(0,len(groundtruthIDs)):
    currCust = groundtruthIDs[custCtr]
    realEventsIDs[currCust] = groundtruthTimesteps[custCtr,0]
    realEventsPhases[currCust] = groundtruthTimesteps[custCtr,1]

custIDInputAll = []
for custCtr in range(0,len(custIDs)):
    custIDInputAll.append(str(custIDs[custCtr]))

# For creation of the time duration curve we omit the customers with changepoints
custIndices = []
for custCtr in range(0,len(groundtruthIDs)):
    index = custIDInputAll.index(groundtruthIDs[custCtr])
    custIndices.append(index)
    
voltageInput = np.delete(voltageInputAll,custIndices,axis=1)
phaseLabelsInput=np.delete(phaseLabelsInputAll,custIndices,axis=1)
custIDInput = list(np.delete(np.array(custIDInputAll),custIndices))

saveResultsBasePath = currentDirectory

###################################################################################
#
#
#  Monte Carlo with Measurement Noise, Missing data, and Incorrect Phase Labels Added
#
#

## Run Monte Carlo Simulation
mislabeledCustsAll = []
cTPPOverTime_NoiseAll = []
numSims = 5
numSamples = voltageInput.shape[0]
windowSize = 384
numWindows = int(numSamples/windowSize)

print('Beginning Monte Carlo Simulations to Create Time Duration Curve')
for i in range(numSims):
    print("################ Simulation " + str(i+1) + "/" + str(numSims) +" ###################")
    
    ### Inject missing labels
    phaseLabelErrors = CPUtils.AddMisLabeledPhases(phaseLabelsInput, 10) #10% mislabeled
    misLabeledCusts = CPUtils.identifyMislabeledCusts(phaseLabelsInput, phaseLabelErrors)    
    noiseVoltage =  CPUtils.AddGaussianNoise(voltageInput, 0.07, 240, 100)
    missVoltage = CPUtils.MissingData_VarInt(noiseVoltage, percentMissing=0.1, minmissingDataInterval=8, maxmissingDataInterval=96)
    # Use shortened voltage timeseries for computation time - the first 3 months are enough to generate the full TDC    
    newVoltage = CPUtils.ConvertToPerUnit_Voltage(missVoltage)
    newVoltage = CPUtils.CalcDeltaVoltage(newVoltage)
 
    predictedPhasesAll, aggMatAll, possibleChangePointsAll, rankedPredictionsOverTime, misLabeledCusts, cTPPOverTime_Noise = OCF.run_TDCMonteCarlo(phaseLabelsInput,phaseLabelErrors,newVoltage,custIDInput,misLabeledCusts,savePath=saveResultsBasePath)
    mislabeledCustsAll.append(misLabeledCusts)
    cTPPOverTime_NoiseAll.append(cTPPOverTime_Noise)
print('Monte Carlo Simulatin Complete')
################################################################################
#
#                  Create TD Curve from Noise Window Scores

# Fit curve parameters and save parameters
td_curve_params,noise_events_max = OCF.useMCResultsToFitTDCurve(cTPPOverTime_NoiseAll,savePath=str(saveResultsBasePath))
print('Saved Time Duration Curve Parameters to the working directory as td_curve_params.npy')
# Specify low confidence score truncation point
tdFlatLineCutoff = 0.5

# Create purely exponential curve to serve as the baseline for the Time Duration Curve
funct  = lambda x, a, b, c: a * np.exp(-b * x) + c
td_curveEXP = lambda x : funct(x , *td_curve_params)

# Conservatively shift and truncate the exponential curve to create the Time Duration Curve
# This is the curve used in the phase changepoint algorithm!
td_curve = lambda x : 0.99 if x == 2 else ( tdFlatLineCutoff if funct(x-1, *td_curve_params) <= tdFlatLineCutoff else funct(x-1, *td_curve_params))

# Plot TDC and exponential curve with noise points
#   The plot considers the event window to be window 0 as is in the paper.  i.e. easier to discuss in terms of 'event detected 2 windows after occurence. . . '
OCF.PlotTDC_SCATTERPLOT_EventWin2(noise_events_max,numWindows,numSims,td_curve,td_curveEXP,cTPPOverTime_NoiseAll,tdFlatLineCutoff,str(saveResultsBasePath))




