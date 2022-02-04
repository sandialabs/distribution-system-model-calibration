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



# Input data

voltageInputCust: numpy array of float (measurements,customers) - the AMI
   voltage timeseries for each customer.  Each column is the timeseries for 
   a single customer.  The units should be in volts.  
   
voltageInputSens: numpy array of float (measurements, sensor datastreams) - the
   sensor voltage timeseries.  Each column should be a sensor datastream with
   measurements in volts.  The length of the measurements dimension should
   match in length and timestamp with the voltageInputCust field.  Each data
   stream will correspond to one measurement field and phase for a particular
   sensor.  Currently our work is using the average voltage field from the
   sensors.  For example, if there are 10 sensors in the system, and the 
   average voltage field is used, each sensor will have measurements from each
   of the three phases A,B,C.  Thus the length of the sensor datastreams 
   dimension will be 30.  Our work utilized IntelliRupters which take 
   measurements on either side of the device, thus it might be expedient to 
   downselect to one of the two datastreams, as the two datastreams will 
   either be repeated (closed devices) or measuring two different sections of 
   the grid (open devices).  

sensPhases: numpy array of int (1,sensor datastreams) - the phase labels of
    the sensors in integer form.  1 - Phase A, 2 - Phase B, 3 - Phase C.  The
    number of sensor datastreams should match the dimensions and indexing of 
    axis 1 in voltageInputSens
    
sensIDs: list of str - the IDs for the sensors in string form.  The length of 
    this list should match the length and indexing of the sensor datastreams
    dimension of voltageInputSens and sensPhases.  (The IDs will be repeated
    for the different phase datastreams of the same sensor)
    
phaseLabelsTrue: numpy array of int (1,customers) - the ground truth phase 
    labels of each customer in integer form.  1 - Phase A, 2 - Phase B, 
    3 - Phase C.  The length of the array should match in length and indexing 
    with the customer dimension of voltageInputCust

phaseLabelsErrors: numpy array of int (1,customers) - the original utility phase 
    labels of each customer in integer form.  1 - Phase A, 2 - Phase B, 
    3 - Phase C.  The length of the array should match in length and indexing 
    with the customer dimension of voltageInputCust.  These labels are assumed
    to have erros in the labeling.  In the sampel data 18 customers have 
    labels which are different from the labels in phaseLabelsTrue
    
custIDInput: list of str (customers) - the customer IDs in string form for
    each customer.  This list should correspond in length and indexing with
    the customer dimension in voltageInputCust and phaseLabelsInput

"""

##############################################################################
#
#           Import Statements

# Standard Libraries
import sys
import numpy as np
from pathlib import Path
import pandas as pd


# Custom Libraries
import PhaseIdent_Utils as PIUtils
import SensorMethod_Funcs as SensMethod

##############################################################################

##############################################################################
#                    Load Sample Data

currentDirectory = Path.cwd()
filePath = Path(currentDirectory.parent,'SampleData')

filename = Path(filePath,'VoltageData_AMI.npy')
voltageInputCust = np.load(filename)
filename = Path(filePath,'VoltageData_Sensor.npy')
voltageInputSens = np.load(filename)
filename = Path(filePath,'PhaseLabels_Sensor.npy')
sensPhases = np.load(filename)
filename = Path(filePath,'SensorIDs.npy')
sensIDs = list(np.load(filename))
filename = Path(filePath,'PhaseLabelsTrue_AMI.npy')
phaseLabelsTrue = np.load(filename)
filename = Path(filePath,'PhaseLabelsErrors_AMI.npy')
phaseLabelsErrors = np.load(filename)
filename = Path(filePath,'CustomerIDs_AMI.npy')
custIDInput = list(np.load(filename))



##############################################################################
#



# Data pre-processing steps
# This converts the original voltage timeseries (assumed to be in volts) into per-unit representation
vNorm = PIUtils.ConvertToPerUnit_Voltage(voltageInputCust)
# This takes the difference between adjacent measurements, converting the timeseries into a per-unit, change in voltage timeseries
vNDV = PIUtils.CalcDeltaVoltage(vNorm)
vSensNorm = PIUtils.ConvertToPerUnit_Voltage(voltageInputSens)
sensNDV = PIUtils.CalcDeltaVoltage(vSensNorm)



# windowSize is the number of measurments used in each window for calculating 
#   the median correlation coefficient value between each customer and all 
#   sensor datastream
windowSize = 96 

# ccDropFilter is a threshold value.  If the correlation coefficient 
#   separation value (the difference between the highest and next highest CC 
#   for a single sensor), then those correlation coefficients are dropped
ccDropFilter = 0.06 
ccDropFlag = True # Flag to use the correlation coefficient separation filtering or not

# Run Sensor-based Phase Identification Method - see function documentation in SensorMethod_Funcs.py
predictedPhaseLabels,custIDFound,noVotesIndex,noVotesIDs, omittedCust,\
confScoreCombined,sensVotesConfScore, ccSeparation,\
winVotesConfScore,custWindowCounts  = SensMethod.AssignPhasesUsingSensors(vNDV,sensNDV,
                                                               custIDInput, sensIDs, 
                                                               sensPhases,windowSize,
                                                               dropLowCCSepFlag=ccDropFlag,
                                                               numVotes=5,ccSepThresh=ccDropFilter)


# Remove customers which were omitted from analysis due to missing data
if predictedPhaseLabels.shape[1] != phaseLabelsTrue.shape[1]:
    phaseLabelsFound = np.delete(phaseLabelsTrue,noVotesIndex,axis=1)
    phaseLabelsErrorsFound = np.delete(phaseLabelsErrors,noVotesIndex,axis=1)
else:
    phaseLabelsFound = phaseLabelsTrue
    phaseLabelsErrorsFound = phaseLabelsErrors


# Determine customers with different predicted phase labels than predicted 
diffIndices = np.where(predictedPhaseLabels!=phaseLabelsErrors)[1]
diffIDs = list(np.array(custIDFound)[diffIndices])
newPhaseLabels = np.expand_dims(predictedPhaseLabels[0,diffIndices],axis=0)
orgDiffPhaseLabels = np.expand_dims(phaseLabelsErrors[0,diffIndices],axis=0)
totalPredicted = len(diffIDs)


#Filter Results by confidence scores
filDiff,filNPL, filOrgDiff = PIUtils.FilterPredictedCustomersByConf(diffIDs,custIDFound,newPhaseLabels,orgDiffPhaseLabels,winVotesConfScore=winVotesConfScore,
                                       ccSeparation=ccSeparation,sensVotesConfScore=sensVotesConfScore,combConfScore=confScoreCombined,winVotesThresh=0.75, 
                                       ccSepThresh=-1,sensVotesThresh=0.75,combConfThresh=-1)        


# Compare predicted results to the ground-truth phase labels
orgDiff = np.where(predictedPhaseLabels!=phaseLabelsFound)[1]
accuracy = (((phaseLabelsFound.shape[1])-len(orgDiff)) / phaseLabelsFound.shape[1]) * 100


print('Sensor-based Phase Identification Results')
print('')
print('Results compared to the original phase labels:')
print('There were ' + str(len(diffIDs)) + ' customers whose predicted phase labels are different from the original phase labels')
print('Afer filtering using confidence scores, there are ' + str(len(filDiff)) + ' customers with different phase labels')

print('')

print('Results compared to the ground truth phase labels:')
print('There are ' + str(len(orgDiff)) + ' customers with incorrect phase labels')
print('The accuracy of the predicted labels compared to the ground truth is ' + str(accuracy) + '%')

# Write outputs to csv file
df = pd.DataFrame()
df['customer ID'] = custIDFound
df['Original Phase Labels (with errors)'] = phaseLabelsErrorsFound[0,:]
df['Predicted Phase Labels'] = predictedPhaseLabels[0,:]
df['Actual Phase Labels'] = phaseLabelsFound[0,:]
df.to_csv('outputs_SensorMethod.csv')
print('Predicted phase labels written to outputs_SensorMethod.csv')

# Confidence Score Plots
PIUtils.PlotHistogramOfWinVotesConfScore(winVotesConfScore)
PIUtils.PlotHistogramOfCombinedConfScore(confScoreCombined)
PIUtils.PlotHistogramOfSensVotesConfScore(sensVotesConfScore)
PIUtils.PlotHistogramOfCCSeparation(ccSeparation,xLim=-1)



##############################################################################















