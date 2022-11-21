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



This file contains a sample script for running the Co-Association Matrix
Ensemble Phase Identfication algorithm which uses customer advanced metering 
infrastructure (AMI) data with an ensemble spectral clustering algorithm and 
co-association matrix to cluster customers by their service phase. 

This script also requires functions that are in CA_Ensemble_Funcs.py and
PhaseIdent_Utils.py

Input Data Formatting:
    voltageInput: numpy array of float - This matrix contains the voltage 
        measurements (in volts) for all customers under consideration.  The
        matrix should be in the form (measurements,customers) where each 
        column represents one customer AMI timeseries.  It is recommended that
        the timeseries interval be at least 15-minute sampling, although the
        algorithm will still function using 30-minute or 60-minute interval
        sampling
    custIDInput: list of str - This is a list of customer IDs as strings.  The
        length of this list should match the customer dimension of voltageInput
    phaseLabelsErrors: numpy array of int - This contains the phase labels for
        each customer in integer form (i.e. 1 - Phase A, 2 - Phase B,
        3 - Phase C).  Any integer notation may be used for this field; it is 
        only used to assigned the final phase predictions.  In practice, this
        field could even be omitted, the phase identification results from 
        CAEnsemble will still be grouped by phase, and the assignment of final
        phase labels could be left for a post-processing step by the utility.
        The dimensions of this matrix should be (1, customers).
        These are assumed to be the original, utility labels, which
        may contain some number of errors.  The sample data included with these
        scripts has ~9% of phase labels injected with errors.  This can be 
        seen by comparing this field with the entries in phaseLabelsTrue which
        contains the ground-truth phase labels
    phaseLabelsTrue: numpy array of int (1,customers) - This contains the 
        ground-truth phase labels for each customer, if available.  Note that,
        in practice this may not be available, but for testing purposes this 
        is provided along with functions to evaluate the phase identifcation
        accuracy against the ground-truth labels.
        
The indexing of each of the input data must match, i.e. voltageInput[:,1] must
    represent the same customer as custIDInput[1], phaseLabelErrors[0,1] and
    phaseLabelsTrue[0,1]

"""

##############################################################################

# Import - Python Libraries
import numpy as np
from pathlib import Path
import pandas as pd

# Import - Custom Libraries
import CA_Ensemble_Funcs as CAE
import PhaseIdent_Utils as PIUtils



##############################################################################
#                    Load Sample Data

currentDirectory = Path.cwd()
filePath = Path(currentDirectory.parent,'SampleData')

filename = Path(filePath,'VoltageData_AMI.npy')
voltageInputCust = np.load(filename)
filename = Path(filePath,'PhaseLabelsTrue_AMI.npy')
phaseLabelsTrue = np.load(filename)
filename = Path(filePath,'PhaseLabelsErrors_AMI.npy')
phaseLabelsErrors = np.load(filename)
filename = Path(filePath,'CustomerIDs_AMI.npy')
custIDInput = list(np.load(filename))



##############################################################################
###############################################################################
#
#
#         Co-Association Matrix Ensemble Phase Identification
#                   
#


# Data pre-processing steps
# This converts the original voltage timeseries (assumed to be in volts) into per-unit representation
vNorm = PIUtils.ConvertToPerUnit_Voltage(voltageInputCust)
# This takes the difference between adjacent measurements, converting the timeseries into a per-unit, change in voltage timeseries
vNDV = PIUtils.CalcDeltaVoltage(vNorm)


# kFinal is the number of final clusters produced by the algorithm.  Each 
#   cluster will represent a phase grouping of customers.  Ideally, this value
#   could be 3, however in practice usually a larger value is required.  Issues
#   such as customers located on adjancent feeders present in the data, voltage
#   regulators, or other topology issues may require tuning of this parameter.
#   This could be done using silhouette score analysis.  7 is likely a good
#   place to start with this parameter.
kFinal=7

# kVector is the number of clusters used internally by the algorithm in each 
#   window.  
kVector =[6,12,15,30]
# windowSize is the number of datapoints used in each window of the ensemble
windowSize = 384

# This is the primary phase identification function - See documentation in CA_Ensemble_Funcs.py for details on the inputs/outputs
finalClusterLabels,noVotesIndex,noVotesIDs,clusteredIDs,caMatrix,custWindowCounts = CAE.CAEnsemble(vNDV,kVector,kFinal,custIDInput,windowSize)

# Remove any omitted customers from the list of phase labels
if len(noVotesIndex) != 0:
    clusteredPhaseLabels = np.delete(phaseLabelsErrors,noVotesIndex,axis=1)
    clusteredTruePhaseLabels = np.delete(phaseLabelsTrue,noVotesIndex,axis=1)
    custIDFound = list(np.delete(np.array(custIDInput),noVotesIndex))
else:
    clusteredPhaseLabels = phaseLabelsErrors
    clusteredTruePhaseLabels = phaseLabelsTrue
    custIDFound = custIDInput

    
# Use the phase labels to assign final phase predictions based on the majority vote in the final clusters
# This assumes that phase labels are both available and believed to be reasonably accurate.
# In the case where phase labels are unavailable or believed to be highly innacurate, some other method of final phase prediction must be used.
predictedPhases = PIUtils.CalcPredictedPhaseNoLabels(finalClusterLabels, clusteredPhaseLabels,clusteredIDs)

# This shows how many of the predicted phase labels are different from the original phase labels
diffIndices = np.where(predictedPhases!=clusteredPhaseLabels)[1]

# If the ground-truth labels are available, this will calculate a true accuracy
accuracy, incorrectCustCount = PIUtils.CalcAccuracyPredwGroundTruth(predictedPhases, clusteredTruePhaseLabels,clusteredIDs)
accuracy = accuracy*100


print('Spectral Clustering Ensemble Phase Identification Results')
print('There are ' + str(diffIndices.shape[0]) + ' customers with different phase labels compared to the original phase labeling.')
print('')

print('The accuracy of the predicted phase is ' + str(accuracy) + '% after comparing to the ground truth phase labels')
print('There are '+ str(incorrectCustCount) + ' incorrectly predicted customers')
print('There are ' + str(len(noVotesIndex)) + ' customers not predicted due to missing data')


# Calculate and Plot the confidence scores - Modified Silhouette Coefficients
allSC = PIUtils.Calculate_ModifiedSilhouetteCoefficients(caMatrix,clusteredIDs,finalClusterLabels,predictedPhases,kFinal)
PIUtils.Plot_ModifiedSilhouetteCoefficients(allSC)

# Create output list which includes any customers omitted from the analysis due to missing data 
# Those customers will be at the end of the list and have a predicted phase and silhouette coefficient of -99 to indicate that they were not included in the analysis
phaseLabelsOrg_FullList, phaseLabelsPred_FullList,allFinalClusterLabels, phaseLabelsTrue_FullList,custID_FullList, allSC_FullList = PIUtils.CreateFullListCustomerResults_CAEns(clusteredPhaseLabels,phaseLabelsErrors,finalClusterLabels,clusteredIDs,custIDInput,noVotesIDs,predictedPhases,allSC,phaseLabelsTrue=phaseLabelsTrue)
 

# Write outputs to csv file
df = pd.DataFrame()
df['customer ID'] = custID_FullList
df['Original Phase Labels (with errors)'] = phaseLabelsOrg_FullList[0,:]
df['Predicted Phase Labels'] = phaseLabelsPred_FullList[0,:]
df['Actual Phase Labels'] = phaseLabelsTrue_FullList[0,:]
df['Confidence Score'] = allSC_FullList
df['Final Cluster Label'] = allFinalClusterLabels
df.to_csv('outputs_CAEnsMethod.csv')
print('')
print('Predicted phase labels written to outputs_CAEnsMethod.csv')



































