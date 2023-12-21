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


 ChangepointUtils.py

This file contains helper functions for the online phase changepoint detection task
These functions perform tasks adjacent to the main algorithm
   Function List
     - AddGaussianNoise
     - MissingData_VarInt
     - FindNewInterval
     - ConvertToPerUnit_Voltage
     - CalcDeltaVoltage
     - AddMisLabeledPhases
     - identifyMislabeledCusts
     - CreateAggWeightMatrix
     - CleanVoltWindow
     - UpdateCustWindowCounts
     - UpdateAggKM
     - AnalyzeClustersLabels
     - getModeClusterAssignment2
     - updateWindowCtr

    
Publications related to this method:
    
B. D. Peña, L. Blakely, and M. J. Reno, “Online Data-Driven Detection of Phase Changes in Evolving Distribution Systems,” presented at the ISGT, 2023.
B. D. Peña, L. Blakely, M. J. Reno, “Data-Driven Detection of Phase Changes in Evolving Distribution Systems,” presented at TPEC, 2022.

    
"""



# This file contains helper functions for the online phase changepoint detection task
#    These functions perform tasks adjacent to the main algorithm

### Function List
#  - AddGaussianNoise
#  - MissingData_VarInt
#  - FindNewInterval
#  - ConvertToPerUnit_Voltage
#  - CalcDeltaVoltage
#  - AddMisLabeledPhases
#  - identifyMislabeledCusts
#  - CreateAggWeightMatrix
#  - CleanVoltWindow
#  - UpdateCustWindowCounts
#  - UpdateAggKM
#  - AnalyzeClustersLabels
#  - getModeClusterAssignment2
#  - updateWindowCtr


##############################################################################
# Import Python Libraries
import random
import numpy as np
from copy import deepcopy
from scipy.stats import mode
import pandas as pd



def AddGaussianNoise(voltageArray, meanValue,stdPercentValue, percentNoisyMeters):
    ''' This function takes the original array of voltage time-series and 
        adds gaussian noise to each measurement, specified by the mean and 
        standard deviation for the percentage of meters specified.  If the
        percentNoisyMeters is not 100, then the meters with noise are sampled
        uniformly at random up to the specified percentage.
        
        Parameters
        ---------
            voltageArray: Numpy array (measurments,customers) of the original voltage time series
            meanValue: float - the mean value for the gaussian distribution.
                Make sure the units of this match the units in voltageArray,
                i.e. either Volts or per-unit
            stdPercentValue: float - the standard deviation of the gaussian distribution
                specified as a percentage of the meanValue.  i.e. 0.07% of 
                the meanValue.  THIS PARAMETER MUST BE A PERCENTAGE!
            percentNoisyMeters: percentage of meters to bias, 100 = all customers
            idealMean: the ideal mean of the values (240 for the original synthetic dataset)
            
        Returns
        -------
            newVoltageArray: The new voltage time series with gaussian noise 
            added to the measurements
        '''
        
    
    if stdPercentValue == 0:
        return voltageArray
    else:
        stdValueDecimal = stdPercentValue / 100
        stdInUnits = meanValue * stdValueDecimal
        newVoltageArray = deepcopy(voltageArray)
        noisyMetersDecimal = percentNoisyMeters / 100
        custIndices = random.sample(range(voltageArray.shape[1]),int(voltageArray.shape[1]*noisyMetersDecimal))
        
        for custCtr in range(0,len(custIndices)):
            #print('Starting' + str(custCtr) + '/' +str(len(custIndices)))
            noise = np.random.normal(meanValue,stdInUnits,voltageArray.shape[0])
            newVoltageArray[:,custIndices[custCtr]] = newVoltageArray[:,custIndices[custCtr]] + noise
        
        return newVoltageArray    
# End of AddGaussianNoise



def MissingData_VarInt(VoltageArray, percentMissing, minmissingDataInterval, maxmissingDataInterval):
    ''' This function takes the parameters of a time-series and 
        creates a specified percentage of randomly distributed missing data to the
        time series by uniformly at random selecting data to replace with nan
        for each customer. So each customer has the specified percentage of missing
        data randomly distributed in their time series.  All missing data instances
        will be of  a length in the range between minmissingDataInterval and 
        maxmissingDataInterval. The onus is on the user to choose
        reasonable values for the minmissingDataInterval, maxmissingDataInterval,
        and the percentMissing. If one instance of the minmissingDataInterval
        for each customer is larger than the percentMissing, the program raises an
        error. 
        
        ---------
            VoltageArray: Numpy array of float (measurments,customers)  
                The original voltage time series.
                
            percentMissing: float 
                 Percentage of data to remove from the time series.
                 
            minmissingDataInterval : int
                A parameter that identifies the minimal length of missing data 
                being inserted into the dataset. For example if the paramter is 
                equal to 2, that means that the lowest length of missing data that 
                can be inserted at a certain instance is 2 measurements.
        
            maxmissingDataInterval : int
                A parameter that identifies the maximal length of missing data 
                being inserted into the dataset. For example if the paramter is 
                equal to 50, that means that the longest length of missing data that 
                can be inserted at a certain instance is 50 measurements.
            
        Returns
        -------
            NewVoltageArray: numpy array of int (time,customers) 
                This array has nan's in the places with missing data.
        '''
    # Lists created to be used for graphing and checking purposes.
    percentages = []
    randintervallength = []
    # Range of the intervals.
    intervalrange = maxmissingDataInterval - minmissingDataInterval + 1
    # Creates a copy of VoltageArray so that it can be changed.
    NewVoltageArray = deepcopy(VoltageArray)
    # Finds the number of data points that need to be missing for each customer.
    numIndicesTotal = int(np.round((NewVoltageArray.shape[0]*percentMissing)/100, 0))
    
    # Test cases that check for certain errors.
    if (minmissingDataInterval <= 0) or (maxmissingDataInterval <= 0):
        print("The value of the interval has to be greater than or equal to 0.")
        return -1
    if minmissingDataInterval >= maxmissingDataInterval:
        print("The minimum data interval must be smaller the maximum data interval.")
        return -1
    if (percentMissing < 0) or (percentMissing > 100):
        print("Percent missing has to be in the range from 0 to 100 inclusive.")
        return -1
    if percentMissing == 0:
        print("Percent of the data missing is 0, array is unchanged.")
        return VoltageArray
    if minmissingDataInterval > numIndicesTotal:
        print("The minimum interval is greater than the percent data missing.")
        return -1
    # For loop in order to loop through each customer to add randomized missing data.
    for custCtr in range(0, NewVoltageArray.shape[1]):
        # Counter to count the number of NaN's being added.
        NaNCount = 0 
        # Checks to see that the number of NaN's being added is less than the total needed.
        while NaNCount < numIndicesTotal:
            # Creates a random point in the datalength for an instance of missing data to be inserted.
            indices = random.sample(range(NewVoltageArray.shape[0]), 1)
            index = indices[0]
            # Creates a length for an interval.
            vectorlength = FindNewInterval(minmissingDataInterval, maxmissingDataInterval)
            # Adds the length of the NaN vector to the list randintervallength.
            randintervallength.append(vectorlength)
            # Creates a bound that takes the min value between the end of datalength and the index + vectorlength.
            bound = min(index + vectorlength, NewVoltageArray.shape[0])
            # Checks to see if the index is less than or equal to the bound and if the number of NaN's is less than the total needed.
            while index < bound and NaNCount < numIndicesTotal:
                # If the point being checked doesn't have a NaN, add 1 to NaNcount and make it a NaN.
                if not np.isnan(NewVoltageArray[index, custCtr]):
                    NaNCount += 1
                    NewVoltageArray[index,custCtr] = float('NaN')
                # Add 1 to the index to check the next point.
                index += 1
        # Appends to a list of missing data percentages.
        percentage = (NaNCount/NewVoltageArray.shape[0])*100
        percentages.append(percentage)
    return NewVoltageArray
# End of MissingData_VarInt


def FindNewInterval(minmissingDataInterval,maxmissingDataInterval):
    ''' This function takes the parameters of minmissingDataInterval and 
        maxmissingDataInterval and finds a random interval in the range 
        between the two given parameters. It is used in the function
        MissingData3_OutputMask() in order to randomize the lengths of 
        the missing data being inserted into each customer.
    
    Parameters
    ----------
    minmissingDataInterval : int
        A parameter that identifies the minimal length of missing data 
        being inserted into the dataset. For example if the paramter is 
        equal to 2, that means that the lowest length of missing data that 
        can be inserted at a certain instance is 2 measurements.
        
    maxmissingDataInterval : int
        A parameter that identifies the maximal length of missing data 
        being inserted into the dataset. For example if the paramter is 
        equal to 50, that means that the longest length of missing data that 
        can be inserted at a certain instance is 50 measurements.

    Returns
    -------
    missingIntervalLength : int 
        an interval length chosen at random in the range of the two parameters,
        minmissingDataInterval and maxmissingDataInterval.

    '''
    missingIntervalLength = random.randint(minmissingDataInterval,maxmissingDataInterval)
    return missingIntervalLength
# End of FindNewInterval() function


###############################################################################
#
# ConvertToPerUnit_Voltage
#
def ConvertToPerUnit_Voltage(timeseries):
    ''' This function takes a voltage timeseries and converts it into a per
            unit representation.  This function looks at each customer's 
            timeseries individual, rounds the mean of the measurements and 
            compares that to a list of known base voltages.  Voltage levels may
            need to be added to that list as we use more voltages levels.  
            This allows for the case where some customers run at 240V and some
            run at 120V in the same dataset (This occurs in the EPB data). The
            function will print a warning if some customers have all NaN 
            values, but it will complete successfully in that case with that
            customer having all NaN values in the per-unit timeseries as well.

        Parameters
        ----------
            timeseries: numpy array (measurements,customers), the raw voltage
                measurements

        Returns:
            voltagePU: numpy array (measurements,customers), the voltage 
                timeseries converted into per unit representation
        '''
    
    voltageMismatchThresh = .8
    voltageLevels = np.array([120,240,7200])
    voltagePU = np.zeros((timeseries.shape),dtype=float)
    dataLength = timeseries.shape[0]
    
    for custCtr in range(0, timeseries.shape[1]):
        currentCust = timeseries[:,custCtr]
        # Check for the case a customer only has NaN values in the timeseries
        if np.sum(np.isnan(currentCust)) == dataLength:
            print('Warning!  Customer index ' + str(custCtr) + ' only had NaN values in the timeseries. ')
            voltagePU[:,custCtr] = currentCust
            continue
        else:
            meanValue = np.round(np.nanmean(currentCust),decimals=0)
            vDiff = np.abs(voltageLevels - meanValue)
            index = np.argmin(vDiff)
            if np.abs(vDiff[index]) > (voltageMismatchThresh*voltageLevels[index]):
                print('Error!  Customer# ' + str(custCtr) + 'has a mean voltage value of ' + str(meanValue) + '.  This voltage level is not supported in the function.  Please add this voltage level to the source code of the function')
                return (-1)
            voltagePU[:,custCtr] = np.divide(currentCust, voltageLevels[index])
    return voltagePU
# End of ConvertToPerUnit_Voltage




##############################################################################
#
# CalcDeltaVoltage
#
def CalcDeltaVoltage(voltage):
    ''' This function takes a voltage timeseries and converts a change in 
    voltage representation by taking the difference between adjacent 
    timesteps

        Parameters
        ----------
            voltage: numpy array (measurements,customers), the voltage
                measurements

        Returns:
            deltaVoltage: numpy array (measurements,customers), the voltage 
                timeseries converted into difference representation
        '''   
        
    deltaVoltage = np.diff(voltage, n=1, axis=0)
    return deltaVoltage
# End of CalcDelta Voltage
    



def AddMisLabeledPhases(phaseLabels, percentMislabeled):
    ''' This function takes the original, ground truth, phase labels and injects
        errors into the phase labeling according to the specified percentage.  
        The function ensures that the errors are distributed among the three phases 
        mostly equally.
        
        Parameters
        ---------
            phaseLabels: List of original, ground-truth phase labels
            percentMisLabeled: The desired percentage of errors to inject
            
        Returns
        -------
            newPhaseLabels: the new list of phase labels with errors injected
        '''
    
    newPhaseLabels = deepcopy(phaseLabels)    
    percentMislabeled = percentMislabeled / 100
    aIndices = np.where(newPhaseLabels==1)[1]
    bIndices = np.where(newPhaseLabels==2)[1]
    cIndices = np.where(newPhaseLabels==3)[1]
    phaseAIndices = random.sample(range(len(aIndices)),int(len(aIndices)*percentMislabeled))
    phaseBIndices = random.sample(range(len(bIndices)),int(len(bIndices)*percentMislabeled))
    phaseCIndices = random.sample(range(len(cIndices)),int(len(cIndices)*percentMislabeled))
    for errorCtr in range(0,len(phaseAIndices)):
        newPhaseLabels[0,aIndices[phaseAIndices[errorCtr]]] = 2    
    #End errorCtr for loop - A Indices
    
    for errorCtr in range(0,len(phaseBIndices)):
        newPhaseLabels[0,bIndices[phaseBIndices[errorCtr]]] = 3      
    #End errorCtr for loop - A Indices
    
    for errorCtr in range(0,len(phaseCIndices)):
        newPhaseLabels[0,cIndices[phaseCIndices[errorCtr]]] = 1
    #End errorCtr for loop - A Indices
    return newPhaseLabels  
# End of AddMislabeledPhases


def identifyMislabeledCusts(phaseLabelsInput, phaseLabelErrors):
    ''' This function takes finds the indices of the intentionally mislabeled
    customers
        
        Parameters
        ---------
            phaseLabelsInput: ndarray of int -  List of original, ground-truth 
                phase labels
            phaseLabelErrors: ndarray of int - the list of phase labels with
                some customers intentionally mislabeled
            
        Returns
        -------
            incies: list of int - the list of customer indices where the phase
                label was changed.
        '''
    matches = phaseLabelsInput[:,] == phaseLabelErrors[:,]
    indices = np.where(matches == False)[1]

    return indices
# End of identifyMislabeledCusts


                    

##############################################################################
#
#       CreateAggWeightMatrix
#
def CreateAggWeightMatrix(custID):
    """ This function takes list of customer IDs and returns an emtpy (all zero)
        weight matrix for the phase identification case where existing phase
        labels are not used.  It also calculates the weight increment size using
        the number of windows.
            
            Parameters
            ---------
                custID: list of string containing the IDs of erach customer
                
                
            Returns
            -------
                aggWM: ndarray of float the aggregated weight matrix initialized
                    to all zeros.  This will update with weights from each window,
                    tracking paired/unpaired customer information
            """
    aggWM = np.zeros((len(custID),len(custID)),dtype=float)
    return aggWM
# End of CreateAggWeightMatrix
    


##############################################################################
#
# CleanVoltWindow
#
def CleanVoltWindow(voltWindow,currentCustIDs,currentPhaseLabels):
    """ This function takes a window of voltage time series and removes customers
        which contain missing data during that window.  This function is for 
        the phase identification task and does not include transformer labels.
        Use CleanVoltWindowTrans for the transformer pairing task
            
            Parameters
            ---------
                voltWindow: numpy array of float (measurements,customers) time 
                    series voltage measurements
                currentCustIDs: numpy array of strings (customers) with the 
                    IDs of the customers currently in use
                currentPhaseLabels: numpy array of int (1,customers) with the
                    phase labels of the current customers in use

            Returns
            -------
                voltWindow: numpy array of float (measurements, customers) the
                    same volt window with the customers which had missing data
                    during the window removed
                currentCustIDs: numpy array of strings (customers) same list
                    of customers without the 'cleaned' customers
                currentPhaseLabels: numpy array of in (1,customers) same list
                    of phase labels without the 'cleaned customers
            """
    badIndices = []
    for custCtr in range(0,voltWindow.shape[1]):
        temp = voltWindow[:,custCtr]
        indices = np.where(np.isnan(temp))
        if len(indices[0]) != 0:
            badIndices.append(custCtr)
    voltWindow = np.array(voltWindow)
    voltWindow = np.delete(voltWindow,badIndices,axis=1)
    currentCustIDs = np.delete(currentCustIDs,badIndices)
    currentPhaseLabels = np.delete(currentPhaseLabels,badIndices,axis=1)
    voltWindow = pd.DataFrame(voltWindow)
    return voltWindow,currentCustIDs,currentPhaseLabels
# End of CleanVoltWindow


##############################################################################
#
#       UpdateCustWindowCounts
#
def UpdateCustWindowCounts(custWindowCounts,currentIDs,custIDInput):
    """ This function updates the number of windows that each customer was 
            included in the analysis, i.e. not excluded due to missing data. 
            This function assumes that all entries in currentIDs are contained
            in the list of IDs in custIDInput and does not do error check for
            that fact.
            
            Parameters
            ---------
                custWindowCounts: numpy array of int (total number of customers)- the
                    window count for each customer
                currentIDs: list of str (current number of customers) - the list of 
                    customers IDs included in the current window to be added
                    to the custWindowCounts
                custIDInput: list of str (total number of customers) - the 
                    complete list of customer IDs. The length of this should
                    match the length/indexing of custWindowCounts
            Returns
            -------
                custWindowCounts: numpy array of int (total number of customers)
                    the window count for each customer that has been updated
                    with the new information in currentIDs
            """
    
    for custCtr in range(0,len(currentIDs)):
        index = np.where(np.array(custIDInput)==currentIDs[custCtr])[0][0]
        custWindowCounts[index] = custWindowCounts[index] + 1
    return custWindowCounts
# End of UpdateCustWindowCounts function
                    


def UpdateAggKM(clusterLabels,custID,currentIDs,aggKM):
    """ This function takes cluster labels resulting from the k-means
        of a window, the existing weight matrix and the customer ids to update
        the weights based on the current k-means results.  Paired customers' 
        weights are incremented. 
        
        This is a modification of the UpdateAggWM function used to create an
        aggregate weight matrix for the k-vector k-means results for a single 
        window of data. 
            
            Parameters
            ---------
                clusterLabels: ndarray of int representing the cluster labeling
                    of each customer from the spectral clustering algorithm
                custID: list of string containing the IDs of each customer
                currentCustID: list of strings containing the IDs of each customer
                    clustered in the current window (does not include customers
                    with missing data in this window)
                aggKM: ndarray of float, shape (customers,customers) the 
                    aggregated weight matrix previously initialized
            Returns
            -------
                aggKM: ndarray of float the aggregated weight matrix previously
                    initialized and updated with the new informaiton from this window.
            """
    
    allIndices = []
    for custCtr in range(0,len(currentIDs)):
        custIDStr = np.array(custID,dtype=str)
        custIndex = np.where(currentIDs[custCtr]==custIDStr)[0][0]
        allIndices.append(custIndex)
        updateIndices = np.where(clusterLabels==clusterLabels[custCtr])[0]
        updateIndicesTrue = np.in1d(custIDStr,currentIDs[updateIndices])
        updateIndicesTrue = np.where(updateIndicesTrue==True)[0]
        aggKM[custIndex,updateIndicesTrue] = aggKM[custIndex,updateIndicesTrue] + 1
    # End of custCtr for loop
    return aggKM
# End of UpdateAggWM function


###############################################################################
#
#                           AnalyzeClusterLabels
#
# 
def AnalyzeClustersLabels(clusterLabels, currentPhaseLabels, k):
    ''' This function takes the utility labels and the labels for which 
        customers were in which cluster and creates a matrix showing the number of 
        each utility label in each cluster and also creates a clustering prediction 
    
        Parameters
        ---------
            clusterLabels: numpy array of int (customers) - the number for 
                which cluster a customer ended up in during the clustering 
            currentPhaseLabels: numpy array of int (1,customers) - the utility 
                labels for the customers in this clustering run
            k: int - number of clusters
        
        Returns
        -------
            clusterResults - (number of clusters, 4) counts for each of the 4 
                phases in each cluster
            predictedPhase - the majority vote of the cluster that each 
                customers was labeled in
        '''
    
    clusterResults = np.zeros((k,4), dtype=float)
    predictedPhase = np.zeros((currentPhaseLabels.shape),dtype=int)
    for clustCtr in range(0,(k)):
        indices = np.where(clusterLabels==clustCtr)
        for ctr in np.nditer(indices):
            clusterResults[clustCtr,currentPhaseLabels[0,ctr]] = clusterResults[clustCtr,currentPhaseLabels[0,ctr]] + 1
    for clustCtr in range(0,k):
        clusterPhase = np.argmax(clusterResults[clustCtr][:])
        indices = np.where(clusterLabels==clustCtr)
        predictedPhase[0][indices] = clusterPhase
    return clusterResults,predictedPhase                       
# End of AnalyzeClustersLabels


##Return's array of window votes with mode of votes from k-vector clustering
def getModeClusterAssignment2(windowVotes):
    ''' This function takes the each of the votes produced by the different
        values for k in the clustering and returns the mode of the predictions
    
        '''    
    windowVotesCon = np.zeros(shape = (windowVotes.shape[0], windowVotes.shape[1]) )
    windowVotes = deepcopy(windowVotes)
    windowVotes = windowVotes[:,:,:3]
    for i in range(windowVotes.shape[0]):
        clusterMode = mode(windowVotes[i], axis = 1)
        windowVotesCon[i] = clusterMode[0].flatten()
    return windowVotesCon




def updateWindowCtr(windowCtr, custID, currentIDs):
    """
    Used to only update the window ctr matrix.

    Parameters
    ----------
    windowCtr : ndarray of int, shape (customers,customers) containing 
                    a count of how many windows each customer was clustered in
    custID : list of string containing the IDs of each customer
    currentIDs :  list of strings containing the IDs of each customer
                    clustered in the current window (does not include customers
                    with missing data in this window)
    Returns
    -------
    windowCtr : ndarray of int, shape (customers,customers)  containing 
                    a count of how many windows each customer was clustered in
    """
    allIndices = []
    for custCtr in range(0,len(currentIDs)):
        custIDStr = np.array(custID,dtype=str)
        custIndex = np.where(currentIDs[custCtr]==custIDStr)[0][0]
        allIndices.append(custIndex)
        
    if len(custID) == len(currentIDs):
        windowCtr = windowCtr + 1
    else:
        for custCtr in range(0,len(allIndices)):
            windowCtr[allIndices[custCtr],allIndices] = windowCtr[allIndices[custCtr],allIndices] + 1

    return windowCtr
# End of updateWindowCtr


##############################################################################
#
#       ConvertCSVtoNPY
#

def ConvertCSVtoNPY( csv_file ):
    dataSet = pd.read_csv( csv_file, header=None )
    return np.array( pd.DataFrame(dataSet).values )

# End ConvertCSVtoNPY function


