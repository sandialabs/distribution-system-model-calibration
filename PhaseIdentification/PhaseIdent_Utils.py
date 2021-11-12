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


This file contains misc helper functions for the phase identification task
"""




import numpy as np
import warnings
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import datetime
from scipy import stats


###############################################################################
#
# ConvertToPerUnit_Voltage
#
def ConvertToPerUnit_Voltage(timeseries):
    ''' This function takes a voltage timeseries and converts it into a per
            unit representation.  This function looks at each customer's 
            timeseries individual mean, rounds the mean of the measurements and 
            compares that to a list of known base voltages.  Voltage levels may
            need to be added to that list as more voltages levels are used.  
            This allows for the case where some customers run at 240V and some
            run at 120V in the same dataset. The function will print a warning 
            if some customers have all NaN values, but it will complete 
            successfully in that case with that customer having all NaN values
            in the per-unit timeseries as well.  Supported base voltages are
            120, 240, 7200.  

        Parameters
        ----------
            timeseries: numpy array of float (measurements,customers) - the 
                raw AMI voltage measurements

        Returns:
            voltagePU: numpy array of float (measurements,customers) -  the 
                voltage timeseries converted into per-unit representation
                
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
            
            if index == 0:
                print('Customer index ' + str(custCtr) + ' is a 120V customer')
            # Check for the case where the correct voltage level is not listed
            if np.abs(vDiff[index]) > (voltageMismatchThresh*voltageLevels[index]):
                print('Error!  Customer# ' + str(custCtr) + 'has a mean voltage value of ' + str(meanValue) + '.  This voltage level is not supported in the function.  Please add this voltage level to the source code of the function')
                return (-1)
            voltagePU[:,custCtr] = np.divide(currentCust, voltageLevels[index])
    return voltagePU
# End of ConvertToPerUnit_Voltage


##############################################################################
#
#                           CalcDeltaVoltage
#
def CalcDeltaVoltage(voltage):
    ''' This function takes a voltage timeseries and takes the difference
        between adjacent measurements, converting each measurement into a 
        change in voltage between timesteps.

        Parameters
        ----------
            timeseries: numpy array (measurements,customers), the voltage
                measurements

        Returns:
            deltaVoltage: numpy array (measurements,customers), the voltage 
                timeseries converted into the difference representation, ie
                the change in voltage at each timestep
        '''

    deltaVoltage = np.diff(voltage, n=1, axis=0)
    return deltaVoltage
# End of CalcDelta Voltage



##############################################################################
#
# GetVoltWindow
# 
def GetVoltWindow(voltage,windowSize,windowCtr):
    """ This function takes the voltage time series and retrieves a particular
        window based on the windowCtr parameter
            
            Parameters
            ---------
                voltage: numpy array of float (measurements,customers) time 
                    series voltage measurements
                windowSize: int scalar representing the desired window size
                windowCtr: int scalar representing which window to use

                
                
            Returns
            -------
                voltBatch: numpy array of float (windowSize, customers)
                    one window of the voltage time series
            """
                    
    start = windowCtr * windowSize
    end = windowSize * (windowCtr + 1)
    voltBatch = voltage[start:end,:]
    return voltBatch
# End of GetVoltWindow



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
    #voltWindow = pd.DataFrame(voltWindow)
    return voltWindow,currentCustIDs,currentPhaseLabels
# End of CleanVoltWindow


##############################################################################
#
# CleanVoltWindowNoLabels
#
def CleanVoltWindowNoLabels(voltWindow,currentCustIDs):
    """ This function takes a window of voltage time series and removes customers
        which contain missing data during that window.  This function is for 
        the phase identification task and does not include transformer labels.
        Use CleanVoltWindowTrans for the transformer pairing task.  This version
        of the function does not include the utility phase labels
            
            Parameters
            ---------
                voltWindow: numpy array of float (measurements,customers) time 
                    series voltage measurements
                currentCustIDs: numpy array of strings (customers) with the 
                    IDs of the customers currently in use
            
            Returns
            -------
                voltWindow: numpy array of float (measurements, customers) the
                    same volt window with the customers which had missing data
                    during the window removed
                currentCustIDs: numpy array of strings (customers) same list
                    of customers without the customers which had missing data
                    during the window removed
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
    #voltWindow = pd.DataFrame(voltWindow)
    return voltWindow,currentCustIDs
# End of CleanVoltWindowNoLabels
                         

##############################################################################
#
#       CreateAggWeightMatrix
#
def CreateAggWeightMatrix(custID):
    """ This function takes list of customer IDs and returns an empty (all zero)
        weight matrix for the phase identification case where existing phase
        labels are not used.  
            
            Parameters
            ---------
                custID: list of string containing the IDs of each customer
                
                
            Returns
            -------
                aggWM: ndarray of float the aggregated weight matrix initialized
                    to all zeros.  This will update with weights from each window,
                    tracking paired/unpaired customer information
            """
    aggWM = np.zeros((len(custID),len(custID)),dtype=float)
    return aggWM
# End of CreateAggWeightMatrix
    

###############################################################################
#
# CalcCorrCoef
#
def CalcCorrCoef(voltageWin):
    ''' This function takes a voltage window, calculates the correlation
        coefficients, checks for failure of calculating good correlation
        coefficients (if the resulting matrix is not positive definite, the
        function returns and error), and returns the CC matrix.

        Parameters
        ----------
            voltageWin: numpy array of floats (customers, measurements), the
                window of voltage measurements

        Returns:
            corrCoef: numpy array of floats (customers, customers), the
                resulting correlation coefficient matrix, will be mirrored
                across the diagonal, with 1's on the diagonal
            failFlag: boolean - true if correlation coefficient matrix is
                not positive definite
        '''

    failFlag = 0
    voltageWin = np.array(voltageWin)
    voltageWin = voltageWin.transpose()
    warnings.simplefilter("error", RuntimeWarning)
    try:
        corrCoef = np.corrcoef(voltageWin)
    except RuntimeWarning:
        print('RuntimeWarning caught in CalCorrCoefDistances')
        failFlag = 1
        corrCoef = 0

    return corrCoef,failFlag
# End of CalcCorrCoef
                    

##############################################################################
#
#       UpdateAggWM
#
def UpdateAggWM(clusterLabels,custID,currentIDs,aggWM,windowCtr):
    """ This function takes cluster labels resulting from the spectral clustering
        of a window, the existing weight matrix and the customer ids to update
        the weights based on the current window results.  Paired customers' 
        (based on the spectral clustering labels) weights are incremented. 
            
            Parameters
            ---------
                clusterLabels: ndarray of int representing the cluster labeling
                    of each customer from the spectral clustering algorithm
                custID: list of string containing the IDs of each customer
                currentCustID: list of strings containing the IDs of each customer
                    clustered in the current window (does not include customers
                    with missing data in this window)
                aggWM: ndarray of float, shape (customers,customers) the 
                    aggregated weight matrix previously initialized
                windowCtr: ndarray of int, shape (customers,customers) list containing 
                    a count of how many windows each customer was clustered in
            Returns
            -------
                aggWM: ndarray of float the aggregated weight matrix previously
                    initialized and updated with the new informaiton from this window.
                windowCtr: ndarray of int, shape (1,customers) list containing 
                    a count of how many windows each customer was clustered in
            """
    
    allIndices = []
    for custCtr in range(0,len(currentIDs)):
        custIDStr = np.array(custID,dtype=str)
        custIndex = np.where(currentIDs[custCtr]==custIDStr)[0][0]
        allIndices.append(custIndex)
        updateIndices = np.where(clusterLabels==clusterLabels[custCtr])[0]
        updateIndicesTrue = np.in1d(custIDStr,currentIDs[updateIndices])
        updateIndicesTrue = np.where(updateIndicesTrue==True)[0]
        aggWM[custIndex,updateIndicesTrue] = aggWM[custIndex,updateIndicesTrue] + 1
    if len(custID) == len(currentIDs):
        windowCtr = windowCtr + 1
    else:
        for custCtr in range(0,len(allIndices)):
            windowCtr[allIndices[custCtr],allIndices] = windowCtr[allIndices[custCtr],allIndices] + 1
    # End of custCtr for loop
    return aggWM, windowCtr
# End of UpdateAggWM function
        
               

##############################################################################
#
#       NormalizeAggWM
#
def NormalizeAggWM(aggWM,windowCtr):
    """ This function takes the finished, aggregated weight matrix and divides
        each entry by the number of windows that customer was clustered in.  
        This removes the factor of some customers being clustered in fewer
        windows than other customers.
            
            Parameters
            ---------
                aggWM: ndarray of float, shape (customers,customers) the 
                    aggregated weight matrix previously initialized
                windowCtr: ndarray of int, shape (1,customers) list containing 
                    a count of how many windows each customer was clustered in
            Returns
            -------
                aggWM: ndarray of float the aggregated weight matrix previously
                    initialized and updated with the new informaiton from this window.
            """
    onesArray = np.ones((1,np.shape(windowCtr)[1]))
    for custCtr in range(0,aggWM.shape[0]):
        numberArray= onesArray * windowCtr[0,custCtr]
        aggWM[custCtr,:] = np.divide(aggWM[custCtr,:],numberArray)
    return aggWM
# End of NormalizeAggWM function

                    
################################################################################
#
#       CalcPredictedPhaseNoLabels
#
def CalcPredictedPhaseNoLabels(finalClusterLabels, clusteredPhaseLabelErrors,clusteredIDs):
    ''' This function takes the final cluster labels from an ensemble spectral 
        clustering run and the original utility labels and assigns a predicted
        phase to each customer based on the utility phase labels.  Note that
        the approach works well if you are reasonably confident in the original
        utility labeling, i.e. if 50% or more of the original phase labels are
        incorrect then these predicted phase will likely be innaccurate even 
        though the final cluster labels produced by the ensemble should still
        reflect the correct phase groups.
        
        Parameters
        ---------
            finalClusterLabels: ndarray, shape (customers) containing the final
                cluster labels representing the phase predictions.  But these 
                label numbers will not match the real phases
            clusteredPhaseLabelErrors: ndarray, shape (1,customers) containing 
                phase labels for the customers which were clustered.  The
                dimensions should match the length of finalClusterLabels.  This
                is the original phase labels and may contain errors in the 
                labeling
            clusteredIDs: ndarray of str, shape(customers) has the customers 
                which recieved a predicted phase

        Returns
        -------
            predictedPhases: ndarray of int, shape (1,customers) containing the 
                predicted phase labels based on the majority vote of the final
                clusters
        '''
    predictedPhases = np.zeros((1,clusteredIDs.shape[0]),dtype=int)
    # Assign a predicted (actual) phase to each cluster
    numberOfClusters = (np.unique(finalClusterLabels)).shape[0]
    uniqueClusters = np.unique(finalClusterLabels)
    for clustCtr in range(0,numberOfClusters):
        currentCluster = uniqueClusters[clustCtr]
        indices1 = np.where(finalClusterLabels==currentCluster)[0]     
        clusterPhases = clusteredPhaseLabelErrors[0,indices1]
        pPhase = stats.mode(clusterPhases)[0][0]
        predictedPhases[0,indices1] = pPhase        

    return predictedPhases
# End of  CalcPredictedPhaseNoLabels
 
    

################################################################################
#
#       CalcAccuracyPredwGroundTruth
#
def CalcAccuracyPredwGroundTruth(predictedPhases, clusteredPhaseLabels,clusteredIDs):
    ''' This function takes the predicted phase labels, the ground truth labels
        and the list of clustered customers to calculate accuracy.
        
        Parameters
        ---------
            predictedPhases: ndarray of int, shape (1,customers) containing the 
                predicted phase labels based on the majority vote of the final
                clusters
            clusteredPhaseLabels: numpy array of int (1,customers) - the 
                ground truth phase labels for each customer who received a 
                predicted phase.  This dimensions of this should match 
                predictedPhases
            clusteredIDs: ndarray of str, shape(customers) has the customer IDs 
                for customers which recieved a predicted phase

        Returns
        -------
            accuracy: float, decimal accuracy 
            incorrectCustCount: int, number of customers incorrectly classified
        '''
        
    incorrectCustCount = 0
    for custCtr in range(0,len(clusteredIDs)):
        #index = np.where(clusteredIDs[custCtr]==custIDStr)[0][0]
        if predictedPhases[0,custCtr] != clusteredPhaseLabels[0,custCtr]:
            incorrectCustCount = incorrectCustCount + 1
    numNotClust = clusteredPhaseLabels.shape[1] - len(clusteredIDs)
    accuracy = (clusteredPhaseLabels.shape[1]-(incorrectCustCount+numNotClust)) / clusteredPhaseLabels.shape[1]
    return accuracy, incorrectCustCount
# End of CalcAccuracyPredwGroundTruth

              

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
                    

#############################################################################
#
#                           DropCCUsingLowCCSep
#
def DropCCUsingLowCCSep(ccMatrixInput,lowCCSepThresh,sensIDInput):
    """ This function takes the correlation coefficient results from a single window, 
        assuming that the window is from a mix of customers to sensors and removes any
        CC that have a lower CC Separation Score than the specified threshold
            
            Parameters
            ---------
                ccMatrixInput: numpy array of float (customer,sensors)
                    the CC between customers and sensors for a single window
                lowCCSepThresh: float - the CC Separation threshold, any
                    CC values with CC Separation lower than this are discarded
                sensIDInput: list of str - the list of sensor IDs
                
            Returns
            -------
                ccMatrixAdjusted: numpy array of float (customers,sensors) -
                    The CC Matrix with CC values with CC Separation less than
                    the threshold discarded

            """                
        
    ccMatrixAdjusted = deepcopy(ccMatrixInput)
    sensUnique = np.unique(sensIDInput)
    for custCtr in range(0,ccMatrixAdjusted.shape[0]):
        #### Calculate the ccSeparation section
        currCC = ccMatrixAdjusted[custCtr,:]
        for sensCtr in range(0,len(sensUnique)):
            currSensor = sensUnique[sensCtr]
            indices = np.where(np.array(sensIDInput)==currSensor)[0]
            ccSet = np.sort(currCC[indices])
            ccDiff = ccSet[-1] - ccSet[-2]            
            if ccDiff < lowCCSepThresh:
                ccMatrixAdjusted[custCtr,indices] = 0
    return ccMatrixAdjusted
# End of DropCCUsingLowCCSep




##############################################################################
#
# FilterPredictedCustomersByConf
# 
def FilterPredictedCustomersByConf(custIDPredInc,custIDInput,newPhaseLabels,orgDiffPhaseLabels,winVotesConfScore=-1,
                                   ccSeparation=-1,sensVotesConfScore=-1,combConfScore=-1,winVotesThresh=-1, 
                                   ccSepThresh=-1,sensVotesThresh=-1,combConfThresh=-1):
    """ This function takes a list of customers predicted to have incorrect 
            phases and filters them by confidence scores using provided 
            thresholds.  Any provided confidence scores and threshold provided
            are used, otherwise they are ignored
            
            Parameters
            ---------
                custIDPredInc: list of str - the list of customer IDs that 
                    we are predicted to have incorrect phase labels
                custIDInput: list of str - the full list of customer IDS
                newPhaseLabels: numpy array of int (1,customers) - the 
                    predicted phase labels for the customers in custIDPredInc
                orgDiffPhaseLabels: numpy array of int (1,customers) - the 
                    original phase labels for the customers in custIDPredInc
                winVotesConfScore: list of float - the window voting confidence
                    score for each customer. 
                ccSeparation: list of float - the correlation coefficient 
                    separation for each customer
                sensVotesConfScore: list of float - the sensor agreement
                    score for each customer
                combConfScore: list of float - the combined confidence score
                    for each customer
                winVotesThresh: float - the threshold to use for the window
                    voting score
                ccSepThresh: float - the threshold to use for the CC separation
                    score
                sensVotesThresh: float - the threshold to use for the 
                    sensor agreement score
                combConfThresh: float - the threshold to use for the combined
                    confidence score
            Returns
            -------
                filteredCustIDPredInc: list of str - the custIDPredInc list
                    filtered by confidence scores.  The remaining customers
                    will have confidence scores above the specified thresholds
                filteredNewPhaseLabels: numpy array of int (1,customers) - the
                    predicted phase labels for the customers in filteredCustIDPredInc
                filteredOrgPhaseLabels: numpy array of int (1,customers) - the
                    original phase labels for the customers in filteredCustIDPredInc
                    
            """
                    
    deleteList = set({})
    for custCtr in range(0, len(custIDPredInc)):
        currIndex = custIDInput.index(custIDPredInc[custCtr])
        
        if type(winVotesThresh) != int:
            if winVotesConfScore[currIndex] < winVotesThresh:
                deleteList.add(custCtr)
        if type(ccSepThresh) != int:
            if ccSeparation[currIndex] < ccSepThresh:
                deleteList.add(custCtr)
        if type(sensVotesThresh) != int:
            if sensVotesConfScore[currIndex] < sensVotesThresh:
                deleteList.add(custCtr)
        if type(combConfThresh) != int:
            if combConfScore[currIndex] < combConfThresh:
                deleteList.add(custCtr)
    deleteList = list(deleteList)
    filteredCustIDPredInc = list(np.delete(np.array(custIDPredInc),deleteList))
    filteredNewPhaseLabels = np.delete(newPhaseLabels,deleteList)
    filteredOrgPhaseLabels = np.delete(orgDiffPhaseLabels,deleteList)
    return filteredCustIDPredInc, filteredNewPhaseLabels,filteredOrgPhaseLabels
# End of FilterPredictedCustomersByConf

#############################################################################
#
#                           CountClusterSizes
#
def CountClusterSizes(clusterLabels):
    """ This function takes the labels produced by spectral clustering (or
        other clustering algorithm) and counts the members in each cluster.  
        This is primarily to see the distribution of cluster sizes over all
        windows, particularly to see if there singleton clusters or a significant
        number of clusters with a small number of members.
            
            Parameters
            ---------
                clusterLabels: numpy array of int (clustered customers) - the cluster 
                    label of each customer
                
            Returns
            -------
                clusterCounts: numpy array of int (0,k) - the number of customers
                    in each cluster
            """                
        
    currentK = len(np.unique(clusterLabels))
    clusterCounts = np.zeros((1,currentK),dtype=int)
    for clustCtr in range(0,currentK):
        indices = np.where(clusterLabels==clustCtr)[0]
        clusterCounts[0,clustCtr] = len(indices)
    return clusterCounts
# End of CountClusterSizes


# Start of the PlotHistogramOfWinVotesConfScore function.
def PlotHistogramOfWinVotesConfScore(winVotesConfScore,savePath=-1):
    """ This function takes the list of the window votes score which is a confidence
            score based on the percentage of window votes which were the same
            for each customer, and plots a histogram.  For example, if each
            window were taken individually (not in an ensemble) and a predicted
            phase was assigned for each customer in each window, the window 
            votes confidence score for a particular customer would be the percentage
            of windows which agree on the phase.  
            
    Parameters
    ---------
        winVotesConfScore: list of float - the list containing the decimal
            confidence score defined as the mode of the phase votes over all
            windows divided by the total number of windows
        savePath: str or pathlib object - the path to save the histogram 
            figure.  If none is specified the figure is saved in the current
            directory
                
    Returns
    -------
        None
                
            """
    
    plt.figure(figsize=(12,9))
    sns.histplot(winVotesConfScore)
    plt.xlabel('Window Votes Confidence Score', fontweight = 'bold',fontsize=32)
    plt.ylabel('Count', fontweight = 'bold',fontsize=32)
    plt.yticks(fontweight='bold',fontsize=20)
    plt.xticks(fontweight='bold',fontsize=20)
    plt.title('Histogram of Window Votes Confidence Score',fontweight='bold',fontsize=12)
    plt.show()
    plt.tight_layout()    
    today = datetime.datetime.now()
    timeStr = today.strftime("%Y-%m-%d_%H-%M-%S")
    filename = 'WinVotesConfScore_HIST'
    filename = filename + timeStr + '.png'
    # Save Figure
    if type(savePath) is int:   
        plt.savefig(filename)
    else:
        plt.savefig(Path(savePath,filename))  
#End of PlotHistogramOfWinVotesConfScore



# Start of the PlotHistogramOfCombinedConfScore function.
def PlotHistogramOfCombinedConfScore(confScoreCombined,savePath=-1):
    """ This function takes the list of the score which is the combination 
            (multiplied together) of the window voting score and the 
            sensor agreement confidence socre and plots a histogram
            
    Parameters
    ---------
        confScoreCombined: list of float - the list containing combination of
            the winVotesConfScore and the sensVotesConfScore by multiplying
            them together
        savePath: str or pathlib object - the path to save the histogram 
            figure.  If none is specified the figure is saved in the current
            directory
                
    Returns
    -------
        None
                
            """
    
    plt.figure(figsize=(12,9))
    sns.histplot(confScoreCombined)
    plt.xlabel('Combined Confidence Score', fontweight = 'bold',fontsize=32)
    plt.ylabel('Count', fontweight = 'bold',fontsize=32)
    plt.yticks(fontweight='bold',fontsize=20)
    plt.xticks(fontweight='bold',fontsize=20)
    plt.title('Histogram of Combined Confidence Score',fontweight='bold',fontsize=12)
    plt.show()
    plt.tight_layout()    
    today = datetime.datetime.now()
    timeStr = today.strftime("%Y-%m-%d_%H-%M-%S")
    filename = 'CombinedConfScore_HIST'
    filename = filename + timeStr + '.png'
    # Save Figure
    if type(savePath) is int:   
        plt.savefig(filename)
    else:
        plt.savefig(Path(savePath,filename))  
#End of PlotHistogramOfCombinedConfScore
   
    

# Start of the PlotHistogramOfSensVotesConfScore function.
def PlotHistogramOfSensVotesConfScore(sensVotesConfScore,savePath=-1):
    """ This function takes the list of the scores representing the percentage
            of sensors which agreed in the phase prediction for each customer
            
    Parameters
    ---------
        sensVotesConfScore: list of float - the list containing the decimal 
            value for the percentage of sensors which agreed in the phase
            prediction for each customers.  This will be 1 if all sensors 
            agree on the prediction
        savePath: str or pathlib object - the path to save the histogram 
            figure.  If none is specified the figure is saved in the current
            directory
                
    Returns
    -------
        None
                
            """
    
    percentages = np.array(sensVotesConfScore) * 100
    plt.figure(figsize=(12,9))
    sns.histplot(percentages)
    plt.xlabel('Percentage of Sensor Agreement', fontweight = 'bold',fontsize=32)
    plt.ylabel('Count', fontweight = 'bold',fontsize=32)
    plt.yticks(fontweight='bold',fontsize=20)
    plt.xticks(fontweight='bold',fontsize=20)
    plt.title('Histogram of Sensor Agreement',fontweight='bold',fontsize=12)
    plt.show()
    plt.tight_layout()    
    today = datetime.datetime.now()
    timeStr = today.strftime("%Y-%m-%d_%H-%M-%S")
    filename = 'SensorAgreementConfScore_HIST'
    filename = filename + timeStr + '.png'
    # Save Figure
    if type(savePath) is int:   
        plt.savefig(filename)
    else:
        plt.savefig(Path(savePath,filename))  
#End of PlotHistogramOfSensVotesConfScore
    





# Start of the PlotHistogramOfCCSeparation function.
def PlotHistogramOfCCSeparation(ccSeparation,xLim=0.2,savePath=-1):
    """ This function takes the list of the correlation coefficient separation
            confidence scores and plots a histogram.  
            
    Parameters
    ---------
        ccSeparation: list of float - the list containing the separation 
            between the labeled phase CC and the next highest CC for each 
            customer.  This is used as a type of confidence score
        xLim: float - the value for the x-axis limit for the figure.  0.2 is
            the default because that works well with the our utility data. If -1 is 
            used, the function will not specify an x-axis limit.
        savePath: str or pathlib object - the path to save the histogram 
            figure.  If none is specified the figure is saved in the current
            directory
                
    Returns
    -------
        None
                
            """
            
    plt.figure(figsize=(12,9))
    sns.histplot(ccSeparation)
    plt.xlabel('Correlation Coefficient Separation', fontweight = 'bold',fontsize=32)
    plt.ylabel('Count', fontweight = 'bold',fontsize=32)
    plt.yticks(fontweight='bold',fontsize=20)
    plt.xticks(fontweight='bold',fontsize=20)
    if xLim != -1:
        plt.xlim(0,xlim=xLim)
    plt.title('Histogram of Correlation Coefficient Separation',fontweight='bold',fontsize=12)
    plt.show()
    plt.tight_layout()    
    today = datetime.datetime.now()
    timeStr = today.strftime("%Y-%m-%d_%H-%M-%S")
    filename = 'CCSeparation_HIST'
    filename = filename + timeStr + '.png'
    # Save Figure
    if type(savePath) is int:   
        plt.savefig(filename)
    else:
        plt.savefig(Path(savePath,filename))
    plt.close()
#End of PlotHistogramOfCCSeparation
    












   
