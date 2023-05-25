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


 OnlineChangepointFunctions.py

# This file contains the primary functions for the online phase change detection algorithm

   Function List
     - run_TDCMonteCarlo
     - runOnline_UpdatePredictions
     - SpC_MultKWin_SingleWindow
     - SPClustering
     - calculateConfidenceScore_IndividualWindow
     - CalculatePlot_TimeInPhaseScores
     - getPredictedPhase_TimeInPhaseMethod
     - getCumulativeTPP_forTDCCreation
     - Calculate_TimeInPhaseScores_IndividualCustomer
     - useMCResultsToFitTDCurve
     - PlotTDC_SCATTERPLOT_EventWin2
     - runOnline_Initialize
     - OnlineChangepointFunc
     - determineEvent_TimeDuration
     - getChangePointPredictions_CumulativeTPP
     - getFP_FN_TP_Analysis
     - getTime_To_Detection_TP
     - PlotFPOverTime_LINE
     - plotTimeToFlaggedDecided_HIST
     - plotTP_Flagged_Decided_LINE

Publications related to this method:
    
B. D. Peña, L. Blakely, and M. J. Reno, “Online Data-Driven Detection of Phase Changes in Evolving Distribution Systems,” presented at the ISGT, 2023.
B. D. Peña, L. Blakely, M. J. Reno, “Data-Driven Detection of Phase Changes in Evolving Distribution Systems,” presented at TPEC, 2022.

This code primary developed by Bethany D. Peña and Logan Blakely
    
"""



###############################################################################
#   Import Python Libraries
#
import numpy as np
from sklearn.cluster import SpectralClustering
import pandas as pd
from copy import deepcopy
import warnings
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy import stats

# Import Custom Libraries
import ChangepointUtils as CPUtils
    

def run_TDCMonteCarlo(phaseLabelsInput,phaseLabelErrors,newVoltage,custIDs,misLabeledCusts,windowSize=384,kVector=[3,6,12,15,30],savePath=-1):
    '''
    This functions runs a single instance of the monte carlo used to create
        the time duration curve.  
    
    ** This is after initial data has been run and builds on previous predictions**
    
    Parameters
    ----------
    phaseLabelsInput: ndarray of int (1,customers) - the phase labels for each
        customer
    phaseLabelErrors: ndarray of int (1,customers) - the phase labels with 
        injected mislabeling
    newVoltage: ndarray of float (measurements,customers) - the timeseries
        voltage for each customer
    custIDs: list of str - customer ids
    misLabeledCust: list of int - indices of customer with intentionally
        mislabeled phases
    windowSize: int - the number of samples to use in each window of the 
        spectral clustering algorithm.  The default is 384 samples based on
        prior work, and we recommend using that value
    kVector: list of int - the values for number of clusters for the spectral
        clustering algorithm.  This parameter is set based on prior work, and
        we recommend using these values
    savePath: pathlib object or str - the path to save results

    Returns
    -------
    predictedPhasesAll: ndarray of int (customers,windows) - the predicted 
        phase for each customer in each window
    aggMatAll:  list of ndarray of float - each entry in the list is the 
        co-association matrix for that run of the monte carlo
    possibleChangePointsAll:  dict of list of bool - dictionary keyed by 
        customer id with a boolean entry for each window indicating if there 
        was a possible event for that customer in that window
    rankedPredictionsOverTime: list of pandas Dataframes - a dataframe for 
        each window with details on every possible event that is being tracked
    misLabeledCusts:  ndarray of int - the indices of customers which were
        intentionally mislabeled
    cTPPOverTime_Noise:  list of tuples of float - this list tracks the 
        confidence score for noise windows, the first entry in the tuple is an
        integer which indicates how many windows since the (noise) even was
        detected and the second entry in the tuple is the accumulated 
        confidence score
    '''
        
    ### Initialize elements to keep track of        
    rankedPredictionsOverTime = []
    rankedPredictions = 0
    cTPPOverTime_Noise = []
    possibleChangePoints = {}
    for cust in custIDs:
        possibleChangePoints[cust] = []
    
    ### Confidence Scores per window prediction
    confidenceScores = {}
    for cust in custIDs:
        confidenceScores[cust] = []        
    
    possibleChangePointsAll = possibleChangePoints
    confidenceScoresAll = confidenceScores
    aggMatAll = []
    predictedPhasesAll = []
    windowCtrAll =[]
    
    ### Run remaining windows
    totalWindows = int(newVoltage.shape[0] / windowSize)
    
    for windowTracker in range(0, totalWindows):   
        startIndex = windowTracker * windowSize
        endIndex = startIndex + windowSize
        currVoltage=newVoltage[startIndex:endIndex,:]
        
        possibleChangePoints, confidenceScores, windowVotes, aggMat, windowCtr, clusterLabels, predictedPhases = runOnline_UpdatePredictions(currVoltage, windowSize, kVector, custIDs, phaseLabelErrors, aggMatAll, windowCtrAll, possibleChangePointsAll, confidenceScoresAll, predictedPhasesAll)
        
        ### Update tracking 
        possibleChangePointsAll = possibleChangePoints
        confidenceScoresAll = confidenceScores
        aggMatAll.extend(aggMat)
        predictedPhasesAll = predictedPhases
        windowCtrAll.extend(windowCtr)
        
        ### Get Predictions
        rankedPredictions = getCumulativeTPP_forTDCCreation(possibleChangePointsAll,confidenceScores, np.array(aggMatAll), predictedPhasesAll, custIDs,phaseLabelErrors,rankedPredictions)
        rankedPredictionsOverTime.append(rankedPredictions)
        
        ### Get Cumulative TPP for Noise Customers
        for index, prediction in rankedPredictions.iterrows():
            length = prediction['Length After']
            score = prediction['Cumulative TPP After']
            if prediction['Event Phase'] != phaseLabelsInput[0,custIDs.index(prediction['CustID'])]:                
                cTPPOverTime_Noise.append( (length, score) )
    # end of windowTracker for loop    
        
    return predictedPhasesAll, aggMatAll, possibleChangePointsAll, rankedPredictionsOverTime, misLabeledCusts, cTPPOverTime_Noise
# End of run_IncorrectLabels function

def runOnline_UpdatePredictions(voltageWindow, windowSize, kVector, custIDs, phaseLabelsCurrent, initAggMat, initWindowCtr, initPossibleChangePoints, initConfidenceScores, initPredictedPhases):
    '''
    This functions implements the online version of the change point detection for 
    phase changes algorithm.
    This function is used to update predictions after being initialized. 
        
    ** This is after initial data has been run and builds on previous predictions**
    
    Parameters
    ----------
    voltageWindow : array, float (measurements,customers)
        Initial voltage data to calculate predicted phases with.
    windowSize : int
        The size (in number of measurements) of the sliding window.
    kVector : list or array of ints
        Array or list of the number of clusters in the kmeans clustering step.
    custIDs : list, str
        List of customer ids.
    phaseLabelsCurrent : numpy array of int (1,customers)
        The original phase labels for each customer.
    initAggMat: list of ndarray of float - each list entry is the co-assocation
        matrix produced by the spectral clustering algorithm for that window
    initWindowCtr: list of ndarray of int - each list entry is an array indicating
        if customers were available to be paired together in that window, ie
        only customers without missing data in that window
    initPossibleChangePoints: dict of list of bool - dictionary keyed to 
        customer id with a list of boolean values indicating for each window
        if that customer had a possible changepoint
    initConfidenceScores:  dict of list ndarray of float - dictionary keyed
        by customer ID with an array of confidence scores for each phase for
        each window
    initPredictedPhases: ndarray of int (customers,windows) - the predicted
        phase for each customer in each window
    Returns
    -------
    possibleChangePoints : dictionary
        Contains list of True/False values per window for each customer.
        Keys are customer ids.
    confidenceScores : dictionary
        Contains confidence scores
    windowVotes: array of floats (# of customers, # of windows, length of kVector)
        Contains window votes per kVector per window for every customer
    aggMat_Ens: list of ndarray of float - each list entry is the co-assocation
        matrix produced by the spectral clustering algorithm for that window.
        Updated version of initAggMat
    windowCtr_Ens:  list of ndarray of int - each list entry is an array indicating
        if customers were available to be paired together in that window, ie
        only customers without missing data in that window.  Updated version 
        of initWindowCtr
    clusterLabels: ndarray of int (customers,k values) - the cluster labels
        from the spectral clustering algorithm for each customer and each value
        for the number of clusters
    predictedPhases: array of floats (# of customers, # of windows)
        Predicted phase per window
    '''
    ensTotal = int(np.floor(voltageWindow.shape[0] / windowSize))  # This determines the total number of windows based on available data and window size
    ### Initialize dictionaries
    possibleChangePoints = deepcopy(initPossibleChangePoints)    
    confidenceScores = deepcopy(initConfidenceScores)
    
    windowVotes = np.zeros((len(custIDs), len(kVector)))
    clusterLabels = np.zeros(windowVotes.shape)
    if type(initPredictedPhases) != list:
        predictedPhases = deepcopy(list(initPredictedPhases.T )) ## convert back to original format 
    else:
        predictedPhases = deepcopy(initPredictedPhases)
    aggMat_Ens = []
    windowCtr_Ens = []
    numPrevWindows = len(predictedPhases)
    
    ### Calculate predictions for each available window
    currentIDs, \
    kMeansVotes, \
    kMeansClusters, \
    aggKM, \
    predictedPhaseClustering = SpC_MultKWin_SingleWindow(voltageWindow,kVector,custIDs,phaseLabelsCurrent)
    
    predictedPhaseClustering = predictedPhaseClustering.astype(int)
    windowVotes[:, :] = kMeansVotes
    clusterLabels[:, :] = kMeansClusters
    aggMat_Ens.append(deepcopy(aggKM))
    
    ### Determine Predicted Phase and Individual Window Confidence Metrics
    scores = calculateConfidenceScore_IndividualWindow(aggKM, predictedPhaseClustering.T)            

    timePerPhase = scores
    predictedPhaseTPP = getPredictedPhase_TimeInPhaseMethod(timePerPhase)
    predictedPhases.append(predictedPhaseTPP)

    ### update individual window confidence scores
    for cust in custIDs:
        confidenceScores[cust].append(timePerPhase[custIDs.index(cust)])
    
    ### Find customers that had predictions differing from previous window      
    changePoints = []
    noChangePoint = []
    
    if numPrevWindows == 0:
        ## If current window is a missing data window
        for index in range(len(custIDs)):
            if predictedPhases[numPrevWindows][index] == -1:
                noChangePoint.append(index)
                continue         
            if phaseLabelsCurrent[0,index] == -1:
                noChangePoint.append(index)
            else:       
                if predictedPhases[numPrevWindows][index] == phaseLabelsCurrent[0,index]:
                    noChangePoint.append(index)
                else:
                    changePoints.append(index)
    else:
        ## If current window is a missing data window
        for index in range(len(custIDs)):
            if predictedPhases[numPrevWindows][index] == -1:
                noChangePoint.append(index)
                continue
                
            ## If previous window is a missing data window find most recent real data window
            if predictedPhases[(-1) + numPrevWindows][index] == -1:
                nonMissing = np.where(np.array(predictedPhases)[:,index] != -1)[0]
                if len(nonMissing) > 1:
                    nonMissing = nonMissing[-2]          
                    if predictedPhases[numPrevWindows][index] == predictedPhases[nonMissing][index]:
                        noChangePoint.append(index)
                    else:
                        changePoints.append(index)
                else:
                    noChangePoint.append(index)
            else:      
                if predictedPhases[numPrevWindows][index] == predictedPhases[(-1) + numPrevWindows][index]:
                    noChangePoint.append(index)
                else:
                    changePoints.append(index)
        
    ### Customers without change point event in current window
    customersNoChangePoints = np.array(custIDs)[noChangePoint]
    for cust in customersNoChangePoints:
        possibleChangePoints[cust].append(False)        
        
    ### Customers with change point event in current window
    customersWithChangePoints = np.array(custIDs)[changePoints]
    for cust in customersWithChangePoints:
        possibleChangePoints[cust].append(True)
    ### End finding change points and calculating confidence scores

    ## Format predicted phases and convert from 0,1,2 to 1,2,3
    predictedPhases = np.array(predictedPhases).T 
    
    return possibleChangePoints, confidenceScores, windowVotes, aggMat_Ens, windowCtr_Ens, clusterLabels, predictedPhases
## End runOnline_UpdatePredictions


def SpC_MultKWin_SingleWindow(voltage,kVector,custID,phaseLabelsInput):
    """ This function uses spectral clustering for a single window and produces 
        
            Parameters
            ---------
                voltage:  array, float (measurements,customers) 
                    voltage profiles in window for each customer
                kVector: numpy array of int - a vector of the possible values of
                    k for the windows    
                custID: list, str
                    list of customer ids
                phaseLabelsInput: numpy array of int (1,customers) - the original 
                    phase labels for each customer          
                
            Returns
            -------
                currentIDs: list of str - the list of customer ids which were
                    clustered in this window, i.e. did not have missing data
                    in this window
                kMeansVotes: ndarray of int (customers,k values) - the phase 
                    
            """       
    
    # predictedPhases = np.zeros((1,len(custID)),dtype=int)
    aggKM = CPUtils.CreateAggWeightMatrix(custID) # This is an aggregate of the k-means results
   
    # allClusterCounts = []
    custWindowCounts = np.zeros((len(custID)),dtype=int)
    kMeansVotes = np.ones((len(custID), len(kVector)))
    kMeansVotes[:] = -1
    kMeansClusters = np.ones((len(custID), len(kVector)))
    kMeansClusters[:] = -1  
    #Select the next time series window and remove customers with missing data in that window
    windowDistances = voltage
    currentDistances,currentIDs,currentPhaseLabels = CPUtils.CleanVoltWindow(deepcopy(windowDistances), deepcopy(custID),deepcopy(phaseLabelsInput))
    custWindowCounts = CPUtils.UpdateCustWindowCounts(custWindowCounts,currentIDs,custID)                   
    
    # Check for the case where all the distance matrix is all nans
    if not ((currentDistances.shape[0] == 1) and (currentDistances.shape[1] == 1)):
        currentDistances = currentDistances.transpose()
        
        # Loop through each value of k (number of clusters) to use multiple numbers of clusters in each available window
        for kCtr in range(0,len(kVector)):
            k = kVector[kCtr]
            #Check if the cleaning reduced the number of available customers to less than the number of clusters
            if (currentDistances.shape[0] <= k):
                 continue
            #Do the clustering
            clusterLabels, affMatrix = SPClustering(currentDistances,k)
                 
            #Update kmeans weight matrix for just this window
            aggKM = CPUtils.UpdateAggKM(clusterLabels, custID, currentIDs, aggKM)            
            # Update the window voting matrix
            currentResults,predictedPhase = CPUtils.AnalyzeClustersLabels(clusterLabels,currentPhaseLabels,k)
            for custCtr in range(0,len(currentIDs)):
                index = custID.index(currentIDs[custCtr])
                kMeansVotes[index,kCtr] = predictedPhase[0,custCtr]
            
            # Update the cluster counting matrix
            for custCtr in range(0,len(currentIDs)):
                index = custID.index(currentIDs[custCtr])
                kMeansClusters[index,kCtr] = clusterLabels[custCtr]
        #End of kCtr for loop           
    predictedPhaseCon = CPUtils.getModeClusterAssignment2(np.expand_dims(kMeansVotes, axis = 1))
    return currentIDs, kMeansVotes, kMeansClusters, aggKM, predictedPhaseCon
# End of SpC_MultKWin_SingleWindow


def SPClustering(features,k):
    """ This function takes a window of timeseries data for the total number of
         customers and the number of desired clusters and performs the spectral 
         clustering algorithm on that data, returning the cluster labels for each 
         customer.  The kernel function has been hardcoded here to be the Radial
         Basis Function ('rbf') based on the results of this research.
            
            Parameters
            ---------
                features: array, float (customers,measurments) 
                    'window' of time series measurements
                k:  int
                    Number of clusters
                
            Returns
            -------
                clusterLabels:  list, int
                    The resulting label of each customer (1-k)
            """       
    
    sp = SpectralClustering(n_clusters=k,affinity='rbf')
    clusterLabels = sp.fit_predict(features)    
    return clusterLabels , sp.affinity_matrix_       
# End of SPClustering


def calculateConfidenceScore_IndividualWindow(aggKM, currentPhaseLabels, score = 'Predicted'):
    '''
    This function is used to calculate the individual window confidence score metric,
    the time per phase, for each window. This score is saved in the confidence scores
    dictionary. In the current version of this algorithm, this score is not used
    for any of the final predictions. 

    Parameters
    ----------
    aggKM : ndarray of float (customers,customers)
        Co-Association matrix for window.
    currentPhaseLabels : ndarray of int (1,customers)
        Array of phase labels for window.
    score : string, optional
        The type of score to be calculated. 'Predicted' gives the time per predicted phase
        and 'Separation' gives the difference between the highest and second highest
        scores. The default is 'Predicted'.

    Returns
    -------
    tuple or array of floats
        'Predicted' returns an array of the time per phase scorse, 'Separation' returns 
        both a list of the separation scores and an array of the time per phase scores.

    '''
    
    timePerPhase,timePerPredictedPhase = CalculatePlot_TimeInPhaseScores(aggKM,currentPhaseLabels)
    
    if score == 'Predicted':
        return timePerPhase   
    elif score == 'Separation':
        scores = []
        ### Separation Score
        for index, cust in enumerate(timePerPhase):
            # maximum = np.where(cust == np.max(cust))[0][0] + 1
            maximum = np.max(cust)
            secondMax = 0
            
            if maximum < 1.0:
                secondMax = np.max(cust[np.where(cust < maximum)])
                
            separation = maximum - secondMax
            scores.append(separation)
    else:
        print("WRONG KEYWORD FOR CALCULATING CONFIDENCE SCORE!!!!!")
        return -1,-1   
    return scores, timePerPhase
## End calculateConfidenceScore_IndividualWindow


def CalculatePlot_TimeInPhaseScores(caMatrix,predictedPhases):
    """ This function takes the results from running the Ensemble Spectral Cluster
        Phase Identification algorithm, and calculates the time-in-phase metric
        using the co-association matrix.  The time-in-phase metric is the 
        percentage of time, across all instances in the ensemble, that a customer
        was clustered with a particular phase.
            
    Parameters
    ---------
        caMatrix: ndarray of float (customers,customers) - the co-association
            matrix produced by the spectral clustering ensemble.  This is an
            affinity matrix.  Note that the indexing of all variables must
            match in customer order.
        predictedPhases: ndarray of int (1,customers) - the integer predicted
            phase label for each customer
                
    Returns
    -------
        timePerPhase: ndarray of float (customers,number of unique predicted phases)
            - the time-in-phase for each customer
        timePerPredictedPhase: list of float (customers) the time in the 
            predicted phase for each customer
                
            """

    # Replace diagonal of the co-association matrix with NaN
    for ctr in range(0,caMatrix.shape[0]):
        caMatrix[ctr,ctr]= np.nan
    
    phasesUnique = np.unique(predictedPhases)
    indices = np.where(phasesUnique==-1)[0] # -1 is a placeholder for instances of missing data and should not be considered a phase prediction
    if len(indices)>0:
        phasesUnique=np.delete(phasesUnique,indices)
    timePerPhase = np.zeros((predictedPhases.shape[1],len(phasesUnique)),dtype=float)
    caNormalized = np.zeros(caMatrix.shape,dtype=float)
    # Loop through each customer/row of the caMatrix
    for custCtr in range(0,caMatrix.shape[0]):
        if np.nansum(caMatrix[custCtr,:]) == 0: # This may occur with missing data
            for ctr in range(0,len(phasesUnique)):
                timePerPhase[custCtr,ctr] = -1
            continue
        caNormalized[custCtr,:] = caMatrix[custCtr,:] / np.nansum(caMatrix[custCtr,:])
        for phase in phasesUnique:
            if phase == -1:
                continue
            phaseIndices = np.where(predictedPhases==phase)[1]
            if 0 in set(phasesUnique):
                timePerPhase[custCtr,(phase)] = np.nansum(caNormalized[custCtr,phaseIndices])                
            else:
                timePerPhase[custCtr,(phase-1)] = np.nansum(caNormalized[custCtr,phaseIndices])   
    timePerPredictedPhase = []
    for custCtr in range(0,predictedPhases.shape[1]):
        currPhase = predictedPhases[0,custCtr]
        if 0 in set(phasesUnique):
            timePerPredictedPhase.append(timePerPhase[custCtr,(currPhase)])
        else:
            timePerPredictedPhase.append(timePerPhase[custCtr,(currPhase-1)])
    return timePerPhase, timePerPredictedPhase
# End of CalculatePlot_TimeInPhaseScores Function   

def getPredictedPhase_TimeInPhaseMethod(timePerPhase):
    '''
    This function uses the the time in phase metric to determine the predicted
    phase per window. The phase that had the maximum time in phase is the phase
    prediction for that window.
    
    Parameters
    ----------
    timePerPhase :  ndarray of float (customers, number of windows, 3)
        Array of the time in phase scores per customer and window.

    Returns
    -------
    predictedPhases : ndarray of int (customers, number of windows)
        Array of predicted phases per window for each customer
        obtained by the time in phase metric.

    '''
    
    maximums = np.max(timePerPhase, axis = 1)
    predictedPhases = [] 
    for i in range(maximums.shape[0]):
        ##Check for case of missing data (all zeros)
        maxLoc = np.where(timePerPhase[i] ==  maximums[i])[0]
        if len(maxLoc) > 1:
            phase = -1
        else:
            phase = maxLoc[0] + 1
        predictedPhases.append(phase)   
    return predictedPhases
## End getPredictedPhase_TimeInPhaseMethod
    

def getCumulativeTPP_forTDCCreation(possibleChangePoints, confidenceScores, aggMatCurrent, predictedPhases, custIDs, phaseLabelsOriginal,currentPredictions,unflaggingWindowCutoff=23,unflaggingCumTPP=0.2,maxPriorWindows=23):
    '''
    This function is used to process all possible change point results and determine
    whether possible flagged events remain  possible events or are determined to be 
    true events.  Creates a dataframe with a row per possibleChangePoint with
    information about when the event was flagged, number of windows before and
    after an event and the confidence scores for the event
    
    Note that the window in which an event occurs is defined to be window #1
    (not 0), thus if an event occurs at window 20 and is flagged at window 21, 
    it was flagged on window #2 for the event.  
    
    Parameters
    ----------
    possibleChangePoints : dict
        Dictionary keyed by customer ID. Dictionary of lists of boolean values
        for each window. True indicates a possible event was flagged at this location.
    aggMatCurrent : list of ndarray of floats (# of windows, # of customers, # of customers)
        List of co-association matrices over time. After an additional window of data
        is passed through the algorithm, the co-association matrix for that new window
        is appended to the list.
    predictedPhases : ndarray of int (# of customers, number of windows) - the integer predicted
            phase label for each customer and each window
    custIDs : list of strings
        list of customer IDs
    phaseLabelsOriginal: ndarray of int (1,customers) - the integer phase labels
        which may contain phase errors
    currentPredictions: DataFrame - the current predictions for the list of 
        possible changepoints.  This is needed to keep possible changepoints
        which were unflagged consistently unflagged.  The first time this 
        function runs currentPredictions should be supplied as 0
    unflaggingWindowCutoff: int - the number of windows after which an event is 
        unflagged, i.e. not considered a possible event anymore.  Note that
        the event will still be listed in possibleChangePoints but will have
        an event status of unflagged.  The default value for this is 23 whic
        is approximately 3 months with 15-minute data.
    unflaggingCumTPP: float - the minimum value for the cumulative time per
        phase metric below which a possible event is unflagged.  The default
        value for this is 0.4, set off of the monte carlo correct/incorrect
        phase scatterplot
    maxPriorWindows: int - the maximum number of windows to include in the 
        confidence score calculation prior to an event.  The default for this
        is 23 windows which is approximately 3 months with 15-minute data

    Returns
    -------
    finalPredictions : DataFrame
        Dataframe containing information about each possible event and whether it
        was determined to be an event.  Fields for CustID, Event Location,
        Cumulative TPP Before, Length After, Cumulative TPP After, Event Phase,
        Current Window
        CustID: str - customer ID
        Event Location: int -The window in which the possible event was first flagged
        Cumulative TPP Before: float - the confidence score,
            time-per-phase/cumulative-phase-distribution of window prior to the
            flagged window
        Event Phase: int - the predicted phase which is different from the prior
            window, 0,1,2,3 
        Current Window: int - the number of the current window    
    '''
    ## Determine final changepoint predictions
    predictions = []
    
    ## Loop through each customers possible events
    for key, windows in possibleChangePoints.items():
        possibleChangePointsIndices = np.where(np.array(windows) == True)[0]
        currWindow = len(windows) - 1
        custIndex=custIDs.index(key)
        # If the current prediction for this customers is -1, i.e. no prediction for this window -> this customer should not be included because no addition information has been added to the co-association matrix, aggMatCurrent
        if predictedPhases[custIndex,-1] == -1:
            continue

        for index, event in enumerate(possibleChangePointsIndices):         
            ## Get phase of event and phase of window prior to event
            eventPhase = predictedPhases[custIDs.index(key), event]
            lastPhase = predictedPhases[custIDs.index(key), event-1] # This will be nonsensical if event==0, but that is dealt with below

            ##If last phase is missing find last non-missing phase and index
            if lastPhase == -1:
                lastIndex = np.where(predictedPhases[custIDs.index(key),:] != -1)
                lastIndex = lastIndex[0][-1]
                lastPhase = predictedPhases[custIDs.index(key), lastIndex]
            else:
                lastIndex = event-1
            
            ## Get the number of windows prior to the event and determine cumulative
            ## time per phase 
            totalLengthLast = event
            custIndex = custIDs.index(key)
            lengthLast = 0 
            for windowCtr in range(0,totalLengthLast):
                if predictedPhases[custIndex,windowCtr] != -1:
                    lengthLast = lengthLast + 1
            # Use only up to the maxPriorWindows to calculate the prior to event time per phase.  This is just to limit the data usage to something reasonable (i.e. we don't want to use years of prior data)
            if lengthLast < maxPriorWindows:
                startingIndex=0
            else:
                startingIndex = event-maxPriorWindows
            if event != 0:
                cumulativeMatrix = np.sum(aggMatCurrent[startingIndex:event], axis = 0)
                timePerPhase = Calculate_TimeInPhaseScores_IndividualCustomer(cumulativeMatrix,predictedPhases[:,lastIndex], custIDs.index(key))
           
            # Depending on how predicted phases is formatted (0,1,2) or (1,2,3)
            if event == 0:
                cumulativeTPPLast = 0
            else:
                if 0 in predictedPhases:
                    cumulativeTPPLast = timePerPhase[lastPhase]
                else:
                    cumulativeTPPLast = timePerPhase[lastPhase-1]               
            
            ## Determine length of windows after event - Key Point - In this implementation missing windows are not counted in windows after an event
            custIndex = custIDs.index(key)
            totalWindows = len(windows)
            nonMissingCount=0
            for windowCtr in range(event,totalWindows):
                if predictedPhases[custIndex,windowCtr] != -1:
                    nonMissingCount=nonMissingCount+1
            length = nonMissingCount
    
            ## Calculate cumulativeTPP for current event
            cumulativeMatrix = np.sum(aggMatCurrent[event:], axis = 0)
            timePerPhase = Calculate_TimeInPhaseScores_IndividualCustomer(cumulativeMatrix,predictedPhases[:,event], custIDs.index(key))
    
            #Depending on how predicted phases is formatted (0,1,2) or (1,2,3)
            if 0 in predictedPhases:
                cumulativeTPPNext = timePerPhase[eventPhase]
            else:
                cumulativeTPPNext = timePerPhase[eventPhase-1] 
            predictions.append((key, event, lengthLast, cumulativeTPPLast, length, cumulativeTPPNext, eventPhase, currWindow))                  
            
    #Create Predictions DataFrame 
    finalPredictions = {'CustID': [], 'Event Location': [], 'Length Before': [], 'Cumulative TPP Before': [], 'Length After': [], 'Cumulative TPP After': [], 'Event Phase': [], 'Current Window': []}
    for key, event, lengthLast, cumulativeTPPLast, length, cumulativeTPPNext, eventPhase, currWindow in predictions:
        finalPredictions['CustID'].append(key)
        finalPredictions['Event Location'].append(event)
        finalPredictions['Length Before'].append(lengthLast)
        finalPredictions['Cumulative TPP Before'].append(cumulativeTPPLast)
        finalPredictions['Length After'].append(length)
        finalPredictions['Cumulative TPP After'].append(cumulativeTPPNext)     
        finalPredictions['Event Phase'].append(eventPhase)
        finalPredictions['Current Window'].append(currWindow)                       
    finalPredictions = pd.DataFrame(finalPredictions)   
    return finalPredictions
## End getCumulativeTPP_forTDCCreation


def Calculate_TimeInPhaseScores_IndividualCustomer(caMatrix, currentPredictedPhases, customerIndex):
    """ This function takes the results from running the Ensemble Spectral Cluster
        Phase Identification algorithm, and calculates the time-in-phase metric
        using the co-association matrix.  The time-in-phase metric is the 
        percentage of time, across all instances in the ensemble, that a customer
        was clustered with a particular phase.
        
        This version calculates the metric for a single customer rather 
        than the entire set of customers
            

    Parameters
    ---------
        caMatrix: ndarray of float (customers,customers) - the co-association
            matrix produced by the spectral clustering ensemble.  This is an
            affinity matrix.  Note that the indexing of all variables must
            match in customer order.
        currentPredictedPhases: ndarray of int (1,customers) - the integer predicted
            phase label for each customer
        customerIndex: int - the index of the customer to calculate the time in 
        phase metric of.

    Returns
    -------
        timePerPhase: ndarray of float (number of unique predicted phases)
            - the time-in-phase for the given customer
            
    """
            
    caMatrixCopy = deepcopy(caMatrix)
    # Replace diagonal of the co-association matrix with NaN    
    for ctr in range(0,caMatrixCopy.shape[0]):
        caMatrixCopy[ctr,ctr]= 0  
    phasesUnique = np.unique(currentPredictedPhases)   
    timePerPhase = np.zeros(3,dtype=float)       
    caNormalized = np.zeros(caMatrixCopy.shape,dtype=float)
    warnings.simplefilter("error",RuntimeWarning)

    #Grab customer row
    try:
        try:
            caNormalized[customerIndex,:] = caMatrixCopy[customerIndex,:] / np.nansum(caMatrixCopy[customerIndex,:])
        except RuntimeWarning:
            print('RuntimeWarning custindex = ' + str(customerIndex) + ', nansum = ' + str(np.nansum(caMatrixCopy[customerIndex,:])) + ', num nans = ' + str(np.sum(np.isnan(caMatrixCopy[customerIndex,:]))))           
    except:
        for phase in set(phasesUnique):    
            if phase == -1:
                continue
            if 0 in set(phasesUnique):
                timePerPhase[(phase)] = 0
            else:
                timePerPhase[(phase-1)] = 0
                
        return timePerPhase    
    for phase in set(phasesUnique):
        if phase == -1:
            continue
        phaseIndices = np.where(currentPredictedPhases==phase)[0]

        if 0 in set(phasesUnique):
            # print(phasesUnique)
            timePerPhase[(phase)] = np.nansum(caNormalized[customerIndex,phaseIndices])

        else:
            timePerPhase[(phase-1)] = np.nansum(caNormalized[customerIndex,phaseIndices])               
    return timePerPhase
## End Calculate_TimeInPhaseScores_IndividualCustomer


def useMCResultsToFitTDCurve(cTPPOverTime_NoiseAll,savePath=-1):
    '''
    This function uses the results of the Monte Carlo simulation to construct
        the time duration curve for use in the online phase changepoint 
        algorithm.  The results are used to fit an exponential curve which 
        is used as the basis for the Time Duration Curve.  The curve parameters
        are also saved.

    ----------
    cTPPOverTime_NoiseAll: list of list of tuples (number of simulations) - 
        each tuple contains the number of windows since the prediction and the 
        score (CPCD) for each customer with a noisy (incorrect) prediction 
        in the MC simulation
    savePath: str or pathlib object - the path to save the time duration curve
        parameters.  If no value is specified the file will be saved in the 
        current directory

    Returns
    -------
    td_curve_params: ndarray of float - The exponential curve parameters for the
        time duration curve.  This is the key result from the Monte Carlo 
        simulation.
    noise_events_max: list of tuples - the maximum CPCD score for noise events
        for each window after a possible event has occured.  i.e. 2 windows
        after a noise (incorrect) event what is the max confidence score?
    
    '''        
    noise_events =  [item for sublist in cTPPOverTime_NoiseAll for item in sublist]
    
    ## Save minimum CTPP values for noise windows
    noise_events.sort(key = lambda x: x[0])
    noise_events_max = []
    index = 0
    while index < len(noise_events):
        length = noise_events[index][0]
        currentScores = []
        while index < len(noise_events) and noise_events[index][0] == length:
            currentScores.append(noise_events[index][1])
            index += 1
        try:
            noise_events_max.append( (length, max(currentScores)) )
        except:
            continue  
    ## Get lengths and corresponding scores
    lengths = [x[0] for x in noise_events_max]
    scores =  [x[1] for i, x in enumerate(noise_events_max)]
    funct  = lambda x, a, b, c: a * np.exp(-b * x) + c
    try:
        optimizedParameters, pcov = opt.curve_fit(funct, lengths, scores, bounds = (0, [1,np.inf,1]))
    except RuntimeError:
        print("No Fit Found")
    
    ## Save Time Duration Curve Parameters
    filename = savePath + '/td_curve_params'
    np.save(filename,optimizedParameters)  
    return optimizedParameters, noise_events_max
# end of useMCResultsToFitTDCurve
    

def PlotTDC_SCATTERPLOT_EventWin2(noise_events_max,numWindows,numSims,td_curve,td_curveEXP,cTPPOverTime_NoiseAll,tdFlatLineCutoff,savePath):
    '''
    This function plots the created time duration curve, the baseline 
    exponential curve, and all of the noise points in a scatterplot

    ----------
    noise_events_max: list of tuples - the maximum CPCD score for noise events
        for each window after a possible event has occured.  i.e. 2 windows
        after a noise (incorrect) event what is the max confidence score?
    numWindows: int - the number of windows used in the creation of the time
        duration curve
    numSims: int - the number of runs in the Monte Carlo simulation
    td_curve: function - the created time duration curve
    td_curveEXP: function - the baseline exponential function
    cTPPOverTime_NoiseAll: list of list of tuples (number of simulations) - 
        each tuple contains the number of windows since the prediction and the 
        score (CPCD) for each customer with a noisy (incorrect) prediction 
        in the MC simulation    
    tdFlatLineCutoff: float - the cutoff threshold for the tdc
    savePath: str - the path to save the time duration curve
        parameters. 

    Returns
    -------
    None
    
    '''            
    
    ## Plot Scatter Plot with Time Duration Curve - This version considers the event window to be Window 0 -> This version was in the paper.
    windows = list(range(numWindows))
    xAxis=[]
    tdCurvePlot = []
    tdCurveTruncPlot = []
    for ctr in range(0,len(windows)):
        xAxis.append(ctr)
        
    for ctr in range(1,len(windows)+1):
        tdCurvePlot.append(td_curveEXP(ctr))
        tdCurveTruncPlot.append(td_curve(ctr))
    
    plt.figure(figsize=(12,9))
    lengths = []
    scores = []
    minValue = 10
    maxValue = 0
    numberOfNoiseEvents = 0
    for i in range(numSims):
      currList = cTPPOverTime_NoiseAll[i]
      for ctr in range(0, len(currList)):
          currEntry = currList[ctr]
          currLength= currEntry[0] - 1
          if currLength < minValue:
              minValue = currLength
          if currLength > maxValue:
              maxValue = currLength
          lengths.append((currEntry[0]-1))
          scores.append(currEntry[1])
          if currLength == 0:
              numberOfNoiseEvents = numberOfNoiseEvents + 1
      plt.scatter(lengths, scores, s = 0.5, color = 'dimgray',linewidth=1)  
      
    plt.plot([x[1] for x in noise_events_max], marker = '.',markersize=7,linewidth=3,color = 'purple',label='Max Noise Value')
    plt.plot(xAxis,tdCurvePlot, color = 'royalblue', linestyle = '--',linewidth=4, label = 'Exponential Fit')
    plt.plot(xAxis,tdCurveTruncPlot, color = 'red', linestyle = 'dashdot',linewidth=4, label = 'TD Curve')
    plt.ylim((0,1))
    plt.ylabel('Cumulative Phase Cluster Distribution', fontsize = 22,fontweight='bold')
    plt.xlabel('Windows Since Flagging', fontsize = 22,fontweight='bold')    
    plt.xticks(fontweight='bold',fontsize = 16)
    plt.yticks(fontweight='bold',fontsize = 16)
    plt.legend(fontsize = 16)
    plt.tight_layout()
    plt.show()
    figName =  '/TimeDurationCurve_LINESCATTER'
    filename = savePath + figName
    plt.savefig(filename)
# End of PlotTDC_SCATTERPLOT_EventWin2    





def OnlineChangepointFunc(custIDs,voltageTimeseries,phaseLabelErrors,td_curve,\
                          kVector=[3,6,12,15,30],windowSize=384,\
                              numInitialWindows=3):
    '''
    This is the primary function for the online phase changepoint detection
        algorithm.  The algorithm is initialized with a few windows to correct
        any initial incorrect phase labels, and then each subsequent window is 
        processed individually.  In practice each subsequent window would be
        run as it became availale.  Returns a list of dataframes, one per 
        window with the possible events tracked during that window.

    ----------
    custIDs: list of str - the list of customer IDs
    voltageTimeseries: ndarray of float (measurements,customers) - The voltage
        timeseries for each customr.  The customer dimension should match the 
        indexing in custIDs.  The voltage should be in per-unit and in 
        difference representation.  Any data quality issues (for synthetic data)
        should have been injected already
    phaseLabelErrors: ndarray of int (1,customers) - the initial integer phase
        labels for each customer.  The customer dimension should match the 
        indexing of custID.  These labels are assumed to have some level of 
        errors (<50% incorrect).
    td_curve: function - the time duration curve function 
    kVector: list of int - the list of values for the number of clusters to
        use in the spectral clustering component of the algorithm.  The default
        parameters match prior work results on phase identification and should
        probably be used as-is
    windowSize: the number of voltage samples per window.  The default is based
        on prior work results on phase identification and should probably be
        used as-is
    numInitialWindows: int - the number of windows to use for the initialization
        period.  Three is the default and should be considered the minimum.  
        More windows will increase the accuracy in detecting incorrectly 
        labeled customers in the initialization phase, but increase the amount 
        of required historical data.  Likewise, the initialization period 
        assumes that no customers have changed phase during that initial period.
    

    Returns
    -------
    predictionsOverTime : list of pandas dataframe
        The change point predictions for each window.  Note that the initialization
        windows will be compressed into the first entry in the list and thus
        the length of the list will not directly match the total number of 
        windows in the ensemble
        Each dataframe contains information about each possible event and whether it
        was determined to be an event.  Fields for CustID, Event Location,
        Cumulative TPP Before, Length After, Cumulative TPP After, Event Phase,
        Meets TD Req, Status
        CustID: str - customer ID
        Event Location: int -The window in which the possible event was first flagged
        Cumulative TPP Before: float - the confidence score,
            time-per-phase/cumulative-phase-distribution of window prior to the
            flagged window
        Event Phase: int - the predicted phase which is different from the prior
            window, 0,1,2,3 
        Meets TD Req: boolean - if the Cumulative TPP before and after the event
            both meet the requirements from the time duration curve which was
            used this will be true
        Status: str - 'possible', 'unflagged', 'event' the status of this 
            possible event.  'Event' means the possible was determined to be
            an event based on the time duration curve, 'unflagged' means the
            possible event was discarded     
    aggMatAll: list of ndarray of float - each list entry is the co-association
        matrix produced by the spectral clustering algorithm for an individual
        window
    possibleChangePointsAll: dict of list of bool - dictionary keyed by 
        customer id which has a bool for each window indicating if that 
        customer had a possible changepoint at that window
    confidenceScoresAll: dict of list of ndarray of float - dictionary keyed
        by customer id with an array for each window containing the confidence
        scores for each phase
    predictedPhasesAll: ndarray of int (customers, windows) - the predicted 
        phase for each customer in each window
    '''      

    print('   Beginning Initialization Phase')
    endIndexForInit = windowSize * numInitialWindows
    initVoltage = voltageTimeseries[0:endIndexForInit,:]
    ### Initialize with two windows - This is necessary because there may be incorrect phase labels (unrelated to events) in the dataset
        # The initialization windows will be used to correct any initial phase mislabeling
    predictions = 0
    
    possibleChangePoints, confidenceScores, windowVotes, aggMat, windowCtr, clusterLabels, predictedPhases = runOnline_Initialize(initVoltage, windowSize, kVector, custIDs, phaseLabelErrors)
    predictions = getChangePointPredictions_CumulativeTPP(possibleChangePoints, confidenceScores, aggMat, predictedPhases, custIDs, phaseLabelErrors,predictions, td_curve = td_curve)
    
    ### Correct any phase label errors found within the initializtion windows - as long as the meet the time duration curve requirements
    phaseLabelsFirst5 = np.squeeze(stats.mode(predictedPhases, axis = 1)[0].T)
    labelsDontMatch =np.where( (phaseLabelErrors[0,:] == phaseLabelsFirst5[:]) == False)[0]
    phaseLabelsInputUpdated = deepcopy(phaseLabelErrors)
    for cust in labelsDontMatch:
        if phaseLabelsFirst5[cust] == -1: # Case where all/most initialization windows were missing
            continue
        numMissing = len(np.where(predictedPhases[cust,:]==-1)[0])
        length = numInitialWindows-numMissing
        cumulativeMat = np.sum(aggMat, axis = 0)
        cTPP = Calculate_TimeInPhaseScores_IndividualCustomer(cumulativeMat,phaseLabelsFirst5, cust)
        cTPPMax = max(cTPP)
        newPhase = np.where(cTPP == max(cTPP))[0][0] + 1
        meetsTimeDurationReq = determineEvent_TimeDuration(cTPPMax, length, td_curve)
        if meetsTimeDurationReq:
            phaseLabelsInputUpdated[0,cust] = newPhase        
        
    ### Initialize changepoint elements to track from the initialization windows
    possibleChangePointsAll = possibleChangePoints
    confidenceScoresAll = confidenceScores
    aggMatAll = aggMat
    predictedPhasesAll = predictedPhases
    predictionsOverTime = [predictions]
    windowCtrAll = windowCtr
            
    ### Run remaining windows - although this is structured to use historical data, in practice each iteration of this loop would run as a new window of data became available
    totalWindows = int(voltageTimeseries.shape[0] / windowSize)
    print('   Beginning Online Changepoint Detection')
    print('   Total Windows:  ' + str(totalWindows))
    for i in range(numInitialWindows, totalWindows):  
        print('     Window: ' + str(i) + '/' + str(totalWindows))
        startIndex = i*windowSize
        endIndex = startIndex + windowSize
        currVoltage = voltageTimeseries[startIndex:endIndex,:]
        possibleChangePoints, confidenceScores, windowVotes, aggMat, windowCtr, clusterLabels, predictedPhases = runOnline_UpdatePredictions(currVoltage, windowSize, kVector, custIDs, phaseLabelsInputUpdated, aggMatAll, windowCtrAll, possibleChangePointsAll, confidenceScoresAll, predictedPhasesAll)
        
        ### Update tracking 
        possibleChangePointsAll = possibleChangePoints
        confidenceScoresAll = confidenceScores
        aggMatAll.extend(aggMat)
        windowCtrAll.extend(windowCtr)      
        predictedPhasesAll = predictedPhases      
        ### Get Predictions
        predictions = getChangePointPredictions_CumulativeTPP(possibleChangePoints, confidenceScores, np.array(aggMatAll), predictedPhases, custIDs, phaseLabelsInputUpdated,predictions, td_curve = td_curve)
        predictionsOverTime.append(predictions)
 
    return predictionsOverTime, aggMatAll, possibleChangePointsAll, confidenceScoresAll, predictedPhasesAll
# End OnlineChangepointFunc



## Begin runOnline_Initialize 
def runOnline_Initialize(voltageInit, windowSize, kVector, custID, phaseLabelsInput):
    '''
    This function implements the online version of the change point detection for phase changes algorithm.
    This is used to initialize the algorithm and requires atleast five windows of data. 
    The initialization step is used to evaluate the input phase labels and update them
    if the predicted phases for the five initialization windows differ.
        
    Parameters
    ----------
    voltageInit : array, float (measurements,customers)
        Initial voltage data to calculate predicted phases with.
    windowSize : int
        The size (in number of measurements) of the sliding window.
    kVector : list or array of ints
        Array or list of the number of clusters in the kmeans clustering step.
    custID : list, str
        List of customer ids.
    phaseLabelsInput : numpy array of int (1,customers)
        The original phase labels for each customer.
    
    Returns
    -------
    possibleChangePoints : dictionary
        Contains list of True/False values per window for each customer.
        Keys are customer ids.
    confidenceScores : dictionary
        Contains confidence scores
    windowVotes: array of floats (# of customers, # of windows, length of kVector)
        Contains window votes per kVector per window for every customer
    aggMat_Ens: list of ndarray of float - each list entry is the co-association
        matrix produced by the clustering algorithm for that window
    windowCtr_Ens: list of ndarray of int - each list entry is a matrix of if
        each customer was present together in that window, i.e. did not have 
        missing data
    clusterLabels: ndarray of float (customers,number of init win,length of kvector)
        The cluster labels for each customer in each window in each value for
        the number of clusters in the initialization phase     
    predictedPhases: array of floats (# of customers, # of windows)
        Predicted phase per window

    '''
    # This determines the total number of windows based on available data and window size
    ensTotal = int(np.floor(voltageInit.shape[0] / windowSize))  
    print('   Total Initialization Windows: ' + str(ensTotal))

    ### Initialize dictionaries
    possibleChangePoints = {}
    for cust in custID:
        possibleChangePoints[cust] = []
    
    ### Confidence Scores per window prediction
    confidenceScores = {}
    for cust in custID:
        confidenceScores[cust] = []    
    windowVotes = np.zeros((len(custID), ensTotal, len(kVector)))
    clusterLabels = np.zeros(windowVotes.shape)
    predictedPhases = []
    aggMat_Ens = []
    windowCtr_Ens = []
    windowCtr = CPUtils.CreateAggWeightMatrix(custID)
    
    ### Calculate predictions for each available window
    for ctr in range(0,ensTotal):      
        #Grab Window of Voltage
        currentVoltage = voltageInit[(ctr)*windowSize:(ctr+1)*windowSize,:]
        print('     Init Window: ' + str(ctr) + '/' + str(ensTotal))
        
        ### Calculate Predicted Phase
        currentIDs, \
        kMeansVotes, \
        kMeansClusters, \
        aggKM, \
        predictedPhase = SpC_MultKWin_SingleWindow(currentVoltage,kVector,custID,phaseLabelsInput)
        
        ### Update arrays tracking info        
        predictedPhase = predictedPhase.astype(int)
        windowVotes[:, ctr, :] = kMeansVotes
        clusterLabels[:, ctr, :] = kMeansClusters        
        aggMat_Ens.append(deepcopy(aggKM))        
        windowCtr = CPUtils.updateWindowCtr(windowCtr, custID, currentIDs)
        windowCtr_Ens.append(deepcopy(windowCtr))
        
        ### Determine Predicted Phase and Individual Window Confidence Metrics
        if ctr == 0:
            scores = calculateConfidenceScore_IndividualWindow(aggKM, phaseLabelsInput)
        else:
            scores = calculateConfidenceScore_IndividualWindow(aggKM, predictedPhase.T)            
        timePerPhase = scores
        predictedPhase = getPredictedPhase_TimeInPhaseMethod(timePerPhase)
        predictedPhases.append(predictedPhase)

        ### update individual window confidence scores
        for cust in custID:
            confidenceScores[cust].append(timePerPhase[custID.index(cust)])
            
        ### Find customers that had predictions differing from previous window            
        if ctr != 0:
            changePoints = []
            noChangePoint = []
            for index in range(len(custID)):
                
                ## If current window is a missing data window
                if predictedPhases[ctr][index] == -1:
                    noChangePoint.append(index)
                    continue
                
                ## If previous window is a missing data window find most recent real data window
                if predictedPhases[ctr-1][index] == -1:
                    nonMissing = np.where(np.array(predictedPhases)[:,index] != -1)[0]
                    if len(nonMissing) > 1:
                        nonMissing = nonMissing[-2]
                    else:
                        noChangePoint.append(index)
                        continue
                    
                    if predictedPhases[ctr][index] == predictedPhases[nonMissing][index]:
                        noChangePoint.append(index)
                    else:
                        changePoints.append(index)
                else:
                    
                    if predictedPhases[ctr][index] == predictedPhases[ctr-1][index]:
                        noChangePoint.append(index)
                    else:
                        changePoints.append(index)       
        
            ### Customers without change point event in current window
            customersNoChangePoints = np.array(custID)[noChangePoint]
            for index, cust in enumerate(customersNoChangePoints):
                possibleChangePoints[cust].append(False)
                        
            ### Customers with change point event in current window
            customersWithChangePoints = np.array(custID)[changePoints]
            for index, cust in enumerate(customersWithChangePoints):
                possibleChangePoints[cust].append(True)
        else:
            for cust in custID:
                possibleChangePoints[cust].append(False)
    ### End finding change points and calculating confidence scores
                  
    ## Format predicted phases 
    predictedPhases = np.array(predictedPhases).T 
    
    return possibleChangePoints, confidenceScores, windowVotes, aggMat_Ens, windowCtr_Ens, clusterLabels, predictedPhases
## End runOnline_Initialize 




## Begin determineEvent_TimeDuration
def determineEvent_TimeDuration(cumulativeTPPScore, length, td_curve = -1):
    '''
    This function is used to determine whether the cumulative TPP score
    for a given event meets the threshold determined by the time duration curve.
    The default td curves are the piecewise functions that were developed
    arbitrarily for the sake of testing the time duration curve.
    
    A time duration curve function can be passed into this function that will 
    return the time duration threshold based on the length of the event.


    Parameters
    ----------
    cumulativeTPPScore : float
        The cumulative time per phase calculated for the event.
    length : int
        The number of windows since or before the event occurence.
    td_curve : function, optional
        Time duration curve function. This is a function of length.
        If not set, uses defualt piece-wise function.
        The default is -1.

    Returns
    -------
    boolean
        True if event meets time duration threshold, otherwise false.

    '''
    if type(td_curve) is int:
        td_curve = lambda w : 0.98 - 0.01*(w-1) if w <= 8 else (0.90 - 0.019*(w-8) if (w > 8) and (w < 16) else 0.6)
    return cumulativeTPPScore >= td_curve(length)
## End determineEvent_TimeDuration



## Begin getChangePointPredictions_CumulativeTPP
def getChangePointPredictions_CumulativeTPP(possibleChangePoints, confidenceScores, aggMatCurrent, predictedPhases, custIDs, phaseLabelsOriginal,currentPredictions,unflaggingWindowCutoff=23,unflaggingCumTPP=0.2,maxPriorWindows=23, td_curve = -1):
    '''
    This function is used to process all possible change point results and determine
    whether possible flagged events remain  possible events or are determined to be 
    true events.  Creates a dataframe with a row per possibleChangePoint with
    information about when the event was flagged, number of windows before and
    after an event and the confidence scores for the event
    
    Note that the window in which an event occurs is defined to be window #1
    (not 0), thus if an event occurs at window 20 and is flagged at window 21, 
    it was flagged on window #2 for the event.  
    
    
    Parameters
    ----------
    possibleChangePoints : dict
        Dictionary keyed by customer ID. Dictionary of lists of boolean values
        for each window. True indicates a possible event was flagged at this location.
    aggMatCurrent : list of ndarray of floats (# of windows, # of customers, # of customers)
        List of co-association matrices over time. After an additional window of data
        is passed through the algorithm, the co-association matrix for that new window
        is appended to the list.
    predictedPhases : ndarray of int (# of customers, number of windows) - the integer predicted
            phase label for each customer and each window
    custIDs : list of strings
        list of customer IDs
    phaseLabelsOriginal: ndarray of int (1,customers) - the integer phase labels
        which may contain phase errors
    currentPredictions: DataFrame - the current predictions for the list of 
        possible changepoints.  This is needed to keep possible changepoints
        which were unflagged consistently unflagged.  The first time this 
        function runs currentPredictions should be supplied as 0
    unflaggingWindowCutoff: int - the number of windows after which an event is 
        unflagged, i.e. not considered a possible event anymore.  Note that
        the event will still be listed in possibleChangePoints but will have
        an event status of unflagged.  The default value for this is 23 whic
        is approximately 3 months with 15-minute data.
    unflaggingCumTPP: float - the minimum value for the cumulative time per
        phase metric below which a possible event is unflagged.  The default
        value for this is 0.4, set off of the monte carlo correct/incorrect
        phase scatterplot
    maxPriorWindows: int - the maximum number of windows to include in the 
        confidence score calculation prior to an event.  The default for this
        is 23 windows which is approximately 3 months with 15-minute data
    td_curve : lambda function, optional
        The time duration curve as a function of length. The default is -1.
        If not set, will use one of the default piece-wise functions.
    

    Returns
    -------
    finalPredictions : DataFrame
        Dataframe containing information about each possible event and whether it
        was determined to be an event.  Fields for CustID, Event Location,
        Cumulative TPP Before, Length After, Cumulative TPP After, Event Phase,
        Meets TD Req, Status
        CustID: str - customer ID
        Event Location: int -The window in which the possible event was first flagged
        Cumulative TPP Before: float - the confidence score,
            time-per-phase/cumulative-phase-distribution of window prior to the
            flagged window
        Event Phase: int - the predicted phase which is different from the prior
            window, 0,1,2,3 
        Meets TD Req: boolean - if the Cumulative TPP before and after the event
            both meet the requirements from the time duration curve which was
            used this will be true
        Status: str - 'possible', 'unflagged', 'event' the status of this 
            possible event.  'Event' means the possible was determined to be
            an event based on the time duration curve, 'unflagged' means the
            possible event was discarded
        

    '''
    ## Determine final changepoint predictions
    predictions = []
    
    ## Loop through each customers possible events
    for key, windows in possibleChangePoints.items():
        #print(key)
        possibleChangePointsIndices = np.where(np.array(windows) == True)[0]
        currWindow = len(windows) - 1
        custIndex=custIDs.index(key)
        # If the current prediction for this customers is -1, i.e. no prediction for this window -> this customer should not be included because no addition information has been added to the co-association matrix, aggMatCurrent
        if predictedPhases[custIndex,-1] == -1:
            continue
        
        for index, event in enumerate(possibleChangePointsIndices):
            #print(index)
            ## Get phase of event and phase of window prior to event
            eventPhase = predictedPhases[custIDs.index(key), event]
            lastPhase = predictedPhases[custIDs.index(key), event-1] # This will be nonsensical if event==0, but that is dealt with below

            ##If last phase is missing find last non-missing phase and index
            if lastPhase == -1:
                lastIndex = np.where(predictedPhases[custIDs.index(key),0:event] != -1)[0]
                if len(lastIndex) > 1:
                    lastIndex = lastIndex[-1]
                    lastPhase = predictedPhases[custIDs.index(key), lastIndex]
                else:
                    lastIndex = -99
                    lastPhase = -99
            else:
                lastIndex = event-1
            
            ##Check if event is transition back to input label
            if lastPhase != -99:
                if eventPhase == phaseLabelsOriginal[0,custIDs.index(key)]:
                    if determineEvent_TimeDuration(confidenceScores[key][index][lastPhase-1], 1, td_curve):
                        noiseSection = True #previous event must have been noise section
                else:
                    noiseSection = False
            else: 
                noiseSection = False
 
            ## Get the number of windows prior to the event and determine cumulative
            ## time per phase 
            totalLengthLast = event
            custIndex = custIDs.index(key)
            lengthLast = 0 
            for windowCtr in range(0,totalLengthLast):
                if predictedPhases[custIndex,windowCtr] != -1:
                    lengthLast = lengthLast + 1
                    
            # Use only up to the maxPriorWindows to calculate the prior to event time per phase.  This is just to limit the data usage to something reasonable (i.e. we don't want to use years of prior data)
            if lengthLast < maxPriorWindows:
                startingIndex=0
            else:
                startingIndex = event-maxPriorWindows
            
            if event != 0:
                cumulativeMatrix = np.sum(aggMatCurrent[startingIndex:event], axis = 0)
                timePerPhase = Calculate_TimeInPhaseScores_IndividualCustomer(cumulativeMatrix,predictedPhases[:,lastIndex], custIDs.index(key))
           
            # Depending on how predicted phases is formatted (0,1,2) or (1,2,3)
            if event == 0 or lastPhase == -99:
                cumulativeTPPLast = 0
            else:
                if 0 in predictedPhases:
                    cumulativeTPPLast = timePerPhase[lastPhase]
                else:
                    cumulativeTPPLast = timePerPhase[lastPhase-1]    
            
            # Check if windows before event meet time duration threshold
            meetsTimeDurationReqPrev = determineEvent_TimeDuration(cumulativeTPPLast, lengthLast, td_curve)
            
            ## Determine length of windows after event - Key Point - In this implementation missing windows are not counted in windows after an event
            custIndex = custIDs.index(key)
            totalWindows = len(windows)
            nonMissingCount=0
            for windowCtr in range(event,totalWindows):
                if predictedPhases[custIndex,windowCtr] != -1:
                    nonMissingCount=nonMissingCount+1
            length = nonMissingCount
            
            ## Calculate cumulativeTPP for current event
            cumulativeMatrix = np.sum(aggMatCurrent[event:], axis = 0)
            timePerPhase = Calculate_TimeInPhaseScores_IndividualCustomer(cumulativeMatrix,predictedPhases[:,event], custIDs.index(key))
    
            #Depending on how predicted phases is formatted (0,1,2) or (1,2,3)
            if 0 in predictedPhases:
                cumulativeTPPNext = timePerPhase[eventPhase]
            else:
                cumulativeTPPNext = timePerPhase[eventPhase-1]             
            ## Check if current event meets time duration threshold  
            meetsTimeDurationReq = determineEvent_TimeDuration(cumulativeTPPNext, length, td_curve)
            
            try:
                # Use only the cumulativeTPPNext (after event) in the case where all previous windows have been missing, otherwise use both before and after
                if lengthLast == 0 and event != 0:
                    meetsTimeDurationReq = meetsTimeDurationReq
                else:
                    meetsTimeDurationReq = meetsTimeDurationReq and meetsTimeDurationReqPrev               
            except:
                print(timePerPhase.shape)
                print(cumulativeTPPNext)
                print(meetsTimeDurationReq)
                       
            # Get current status for that event (if it already exists)
            if type(currentPredictions) != int:
                customerList=currentPredictions.loc[currentPredictions['CustID']==key]
                if len(customerList) != 0:  #This will be 0 if that customer event does not yet exist
                    eventList=customerList.loc[customerList['Event Location'] == event]
                    if len(eventList) != 0: # This will be 0 if the customer has a possible event, but not at this location
                        currentStatus = eventList['Status'].values[0]  # This grabs the prior status of the event
                    else:
                        currentStatus = 'unknown'                    
                else:
                    currentStatus = 'unknown'
            else:
                currentStatus = 'unknown'
                                  
            ## Determine event status
            if currentStatus == 'unflagged':
                status = 'unflagged'
            elif meetsTimeDurationReq:
                if length < 2 or lengthLast < 2: 
                    status = 'possible'
                elif noiseSection:
                    status = 'possible'
                else:
                    status = 'event'
            elif length > unflaggingWindowCutoff:  # The possible event has not met the requirements after many windows
                status = 'unflagged'
            elif cumulativeTPPNext <= unflaggingCumTPP: # The possible event has dipped below reasonable confidence score
                status = 'unflagged'
            else:
                status = 'possible'
                
            predictions.append((key, event, lengthLast, cumulativeTPPLast, length, cumulativeTPPNext, eventPhase, meetsTimeDurationReq, status,currWindow))                  
            
    #Create Predictions DataFrame 
    finalPredictions = {'CustID': [], 'Event Location': [], 'Length Before': [], 'Cumulative TPP Before': [], 'Length After': [], 'Cumulative TPP After': [], 'Event Phase': [], 'Meets TD Req': [], 'Status': [], 'Current Window': []}
    for key, event, lengthLast, cumulativeTPPLast, length, cumulativeTPPNext, eventPhase, meetsTimeDurationReq, status, currWindow in predictions:
        finalPredictions['CustID'].append(key)
        finalPredictions['Event Location'].append(event)
        finalPredictions['Length Before'].append(lengthLast)
        finalPredictions['Cumulative TPP Before'].append(cumulativeTPPLast)
        finalPredictions['Length After'].append(length)
        finalPredictions['Cumulative TPP After'].append(cumulativeTPPNext)     
        finalPredictions['Event Phase'].append(eventPhase)
        finalPredictions['Meets TD Req'].append(meetsTimeDurationReq)
        finalPredictions['Status'].append(status)
        finalPredictions['Current Window'].append(currWindow)                        
    finalPredictions = pd.DataFrame(finalPredictions)     
    return finalPredictions
## End getChangePointPredictions_CumulativeTPP


## Begin getFP_FN_TP_Analysis
def getFP_FN_TP_Analysis(predictionsOverTime, realEventsIDs, windowSize):
    '''
    This function takes the predictions over time and determines the number of
    false positives, false negatives, and true positives over time, and returns
    information about these events. 

    Parameters
    ----------
    predictionsOverTime : list of pandas dataframes
        The change point predictions for each window.  Note that the initialization
        windows will be compressed into the first entry in the list and thus
        the length of the list will not directly match the total number of 
        windows in the ensemble
        Each dataframe in the list contains information about each possible event and whether it
        was determined to be an event.  Fields for CustID, Event Location,
        Cumulative TPP Before, Length After, Cumulative TPP After, Event Phase,
        Meets TD Req, Status      
    realEventsIDs : dict
        Dictionary keyed by customer ID containing the timestep locations for
        the events.
    windowSize : int
        The number of timesteps in a window.

    Returns
    -------
    resultsOverTime : list of tuples
       A list of tuples containing the dataframes for fp, fn, and tp customers,
       repectively.  Note the the length of the list will match the length of
       predictionsOverTime and not directly match the total number of windows.

    '''
    
    resultsOverTime = []    
    for window, prediction in enumerate(predictionsOverTime):       
        ### Determine FP, TP, FN for this window
        falsePositives = []
        falsePositiveIDs = []      
        trueEventsFlagged = []
        trueEventsIDs = []
        falseNegatives = []
        falseNegativeIDs = []
        allFlaggedIDs = set({})
        for index, item in prediction.iterrows():
            allFlaggedIDs.add(item['CustID'])
            ## For customers that do not have events
            if item['CustID'] not in list(realEventsIDs.keys()):
                if item['Status'] == 'event':
                    #print(item['CustID'])
                    falsePositives.append(item)
                    falsePositiveIDs.append(item['CustID'])
                    
            ## For customers that do have events, but may have false positive events flagged
            else:
                eventLocation = int(np.ceil(realEventsIDs[item['CustID']][0]/windowSize)) - 1
                flaggedLocation = item['Event Location']
                
                # Two window margin for considering an event detection to be correct
                if not (flaggedLocation > eventLocation - 2 and flaggedLocation < eventLocation + 2):
                    if item['Status'] == 'event':
                        #print(item['CustID'])
                        #print(flaggedLocation)
                        falsePositives.append(item)
                        falsePositiveIDs.append(item['CustID'])
                else:
                    if item['Status'] == 'event':
                        trueEventsFlagged.append(item)
                        trueEventsIDs.append(item['CustID'])
                    else:
                        falseNegatives.append(item)
                        falseNegativeIDs.append(item['CustID'])                       
        falsePositives = pd.DataFrame(falsePositives)
        falsePositiveIDs = np.unique(np.array(falsePositiveIDs))
        trueEventsFlagged = pd.DataFrame(trueEventsFlagged)
        falseNegatives = pd.DataFrame(falseNegatives)        
        ### Save Results for window
        resultsOverTime.append( (falsePositives, falseNegatives, trueEventsFlagged) )
    return resultsOverTime
## End getFP_FN_TP_Analysis
    

def getTime_To_Detection_TP(predictionsOverTime, realEventsIDs,realEventsPhases, windowSize,maxWindowsEarly=5):
    '''
    This function analyzes the difference between the window of event occurnce
    and the window the event was flagged in and the window the event was decided
    to represent a real event in.  This function only considers true events
    and true event flagging, and does not consider any of the types of false 
    positives.  This means that customers which do not have a 
    real event (false positives) and customers which do have an event but the 
    possible event was the wrong phase (false positive), or customers which 
    do have a real event, and possible event was the correct phase, but are 
    too early to be reasonably considered (false positive) are omitted.  If 
    there are noise windows around an event of the true phase of an event, this
    can result in, what we are considering to be, early detection of an event
    because if the noise windows are just right there is no other way to deal
    with this situation.  In this case, the window difference will be negative
    and the event will be noted as detected early.  
    
    TODO: Note:  This may not be the best way to deal with noisy early 
    detection.  If this causes trouble, revist this! 
    
    Note that the window in which an event occurs is defined to be window #1
    (not 0), thus if an event occurs at window 20 and is flagged at window 21, 
    it was flagged on window #2 for the event.  For the sake of plotting this 
    is ignored in the early detection case, if the event was flagged on window
    18, then it was flagged on Window -2 for the event.  
    
    Parameters
    ----------
    predictionsOverTime : list of pandas dataframe
        The change point predictions for each window.  Note that the initialization
        windows will be compressed into the first entry in the list and thus
        the length of the list will not directly match the total number of 
        windows in the ensemble
        Each dataframe contains information about each possible event and whether it
        was determined to be an event.  Fields for CustID, Event Location,
        Cumulative TPP Before, Length After, Cumulative TPP After, Event Phase,
        Meets TD Req, Status
        CustID: str - customer ID
        Event Location: int -The window in which the possible event was first flagged
        Cumulative TPP Before: float - the confidence score,
            time-per-phase/cumulative-phase-distribution of window prior to the
            flagged window
        Event Phase: int - the predicted phase which is different from the prior
            window, 0,1,2,3 
        Meets TD Req: boolean - if the Cumulative TPP before and after the event
            both meet the requirements from the time duration curve which was
            used this will be true
        Status: str - 'possible', 'unflagged', 'event' the status of this 
            possible event.  'Event' means the possible was determined to be
            an event based on the time duration curve, 'unflagged' means the
            possible event was discarded        
        
    realEventsIDs :  dict
        Dictionary keyed by customer ID containing the timestep locations for
        the events.
    realEventsPhases :  dict
        Dictionary keyed by customer ID containing the new customer phases for
        the events.        
    windowSize : int
        The number of timesteps in a window.
        
    Returns
    -------
    timeToFlagged : dict
        Dictionary keyed by customer ID containing the number of windows between 
        the event location and flagging location.
    timeToDecided : dict
        Dictionary keyed by customer ID containing the number of windows between
        the event location and the location of decision.

    '''

    timeToFlagged = {}
    timeToDecided = {}
    for window, predictions in enumerate(predictionsOverTime):
         for index, item in predictions.iterrows():
            custID = item['CustID']
            if (custID in list(realEventsIDs.keys())) and (item['Event Phase'] == realEventsPhases[custID]):
                eventLocation = int(np.ceil((realEventsIDs[custID][0] / windowSize))) - 1
                windowDifference = item['Current Window'] - eventLocation
                # Note that the window in which an event occurs is defined to be window 1, thus 1 is added to the window difference
                if windowDifference >=0:
                    windowDifference= windowDifference + 1               
                if windowDifference >= (maxWindowsEarly*-1):  # Do not possible events which are too early - This may not be the best solution to this issue                  
                    if custID not in timeToFlagged.keys():
                        if item['Status'] != 'unflagged':
                            timeToFlagged[custID] = (item['Current Window'], windowDifference)                           
                    if custID not in timeToDecided.keys():
                        ##Check if decided
                        if item['Status'] == 'event':
                            timeToDecided[custID] = (item['Current Window'], windowDifference)
                        
    return timeToFlagged, timeToDecided
# End of getTime_To_Detection_TP function               
                

def PlotFPOverTime_LINE(resultsOverTime,savePath=-1):
    '''
    This function uses the results of the Monte Carlo simulation to construct
        the time duration curve for use in the online phase changepoint 
        algorithm.  The results are used to fit an exponential curve which 
        is used as the basis for the Time Duration Curve.  The curve parameters
        are also saved.

    ----------
    resultsOverTime : list of tuples
       A list of tuples containing the dataframes for fp, fn, and tp customers,
       repectively.  Note the the length of the list will match the length of
       predictionsOverTime and not directly match the total number of windows.
    savePath: str or pathlib object - the path to save the time duration curve
        parameters.  If no value is specified the file will be saved in the 
        current directory

    Returns
    -------
    td_curve_params: ndarray of float - The exponential curve parameters for the
        time duration curve.  This is the key result from the Monte Carlo 
        simulation.
    noise_events_max: list of tuples - the maximum CPCD score for noise events
        for each window after a possible event has occured.  i.e. 2 windows
        after a noise (incorrect) event what is the max confidence score?
    
    '''  
    
    numFP = list(map(lambda x: len(x[0]) , resultsOverTime))
    ## Plot False Positives Over Time
    plt.figure(figsize=(12,9))
    plt.xlabel('Windows', fontweight='bold',fontsize = 25)
    plt.ylabel('Number of False Positive Events', fontweight='bold',fontsize = 23)
    plt.xticks(fontweight='bold',fontsize = 20)
    plt.yticks([0,1,2], [0,1,2], fontweight='bold',fontsize = 20)
    plt.plot(numFP, color = 'red', marker = '.')
    plt.tight_layout()
    plt.show()
    figName =  '/FPOverTime_LINE.png'
    filePath = str(savePath) + figName
    plt.savefig(filePath)
# End of PlotFPOverTime



def plotTimeToFlaggedDecided_HIST(timeToFlagged,timeToDecided,savePath=-1):
    '''
    This function plots a histogram of the time to flagged and time to decided
        for events, relative to the true event window.  The event window is 
        considered to be window 0 in the plotting (it is considered window 1 
        the code).  Thus perhaps an event was flagged one window after it 
        occured, timeToFlagged = 1, and decided to be a true event two windows 
        after the true event,timeToDecided = 2

    ----------
    timeToFlagged : dict
        Dictionary keyed by customer ID containing the number of windows between 
        the event location and flagging location.
    timeToDecided : dict
        Dictionary keyed by customer ID containing the number of windows between
        the event location and the location of decision.
    savePath: str or pathlib object - the path to save the time duration curve
        parameters.  If no value is specified the file will be saved in the 
        current directory

    Returns
    -------
    None
    
    '''     
    # Note that the -1 here shifts the event window to be window 0 rather than window 1
    #   This was the notation we used for the paper, even though conceptually I think it makes more sense in the code to make the event window be window 1
    timeToFlaggedPlotting = []
    timeToDecidedPlotting = []
    for item in timeToFlagged:
        currValue = timeToFlagged[item][1]
        if currValue > 0:
            currValue = currValue - 1
        timeToFlaggedPlotting.append(currValue)
    for item in timeToDecided:
        currValue = timeToDecided[item][1]
        if currValue > 0:
            currValue = currValue -1 
                                  
        timeToDecidedPlotting.append(currValue)
    ## Plot TP Positives and Histogram of Time to Detection
    flaggedMin = np.min(timeToFlaggedPlotting)
    flaggedMax= np.max(timeToFlaggedPlotting)
    decidedMax = np.max(timeToDecidedPlotting)
    xMin = flaggedMin
    xMax = decidedMax + 2
    plt.figure(figsize=(12,6))
    xTickList=range(xMin,xMax)
    plt.hist(timeToFlaggedPlotting, bins=np.arange(xMin,xMax)-0.5,color = 'blue', edgecolor = 'black', label = 'time to flagged')
    plt.hist(timeToDecidedPlotting,bins=np.arange(xMin,xMax)-0.5, color = 'green', edgecolor = 'black', label = 'time to decided')
    plt.xticks(xTickList,fontweight='bold',fontsize = 25)
    plt.yticks(fontweight='bold',fontsize = 25)
    plt.xlabel('Number of Windows After Event',fontweight='bold', fontsize = 25)
    plt.ylabel('Number of Customers',fontweight='bold', fontsize = 25)
    plt.legend(fontsize=24,loc='upper left')
    plt.tight_layout()
    plt.show()
    figName =  '/TP_FlaggedDecided_HIST.png'
    filePath= str(savePath) + figName
    plt.savefig(filePath)
# End of plotTimeToFlaggedDecided_HIST function     
    


def plotTP_Flagged_Decided_LINE(realEventsIDs,totalWindows,windowSize,timeToFlagged,timeToDecided,savePath=-1):
    '''
    This function plots a line graph with number of customers on the y-axis,
        and windows through time on the x-axis.  The true positives, flagged
        customers, and decided customers are plotted as lines.

    ----------
    realEventsIDs: dict - dictionary keyed by customer ID, containing the 
        timestep of that customers real event
    totalWindows: int - the total number of windows evaluated by the algorithm
    windowSize: int - the number of measurements in each window 
    timeToFlagged : dict
        Dictionary keyed by customer ID containing the number of windows between 
        the event location and flagging location.
    timeToDecided : dict
        Dictionary keyed by customer ID containing the number of windows between
        the event location and the location of decision.
    savePath: str or pathlib object - the path to save the time duration curve
        parameters.  If no value is specified the file will be saved in the 
        current directory

    Returns
    -------
    None
    
    '''        
    
    tpCustWindows = []
    for index, cust in enumerate(realEventsIDs):
        #print(cust)
        timestep = realEventsIDs[cust][0]
        
        eventLocation = int(np.ceil((timestep / windowSize))) -1
        tpCustWindows.append(eventLocation)
    
    numTP=[]
    currentCount = 0
    for ctr in range(0,totalWindows):
        indices = np.where(np.array(tpCustWindows)==ctr)[0]
        currentCount = currentCount + len(indices)
        numTP.append(currentCount)
        
    flaggedWindows = []
    for index, custID in enumerate(timeToFlagged):
        currTuple = timeToFlagged[custID]
        flaggedWindows.append(currTuple[0])
        
    numFlagged= []
    currentCount = 0
    for ctr in range(0,totalWindows):
        indices = np.where(np.array(flaggedWindows)==ctr)[0]
        currentCount = currentCount + len(indices)
        numFlagged.append(currentCount)
        
        
    decidedWindows = []
    for index, custID in enumerate(timeToDecided):
        currTuple = timeToDecided[custID]
        decidedWindows.append(currTuple[0])
        
    numDecided= []
    currentCount = 0
    for ctr in range(0,totalWindows):
        indices = np.where(np.array(decidedWindows)==ctr)[0]
        currentCount = currentCount + len(indices)
        numDecided.append(currentCount)
             
    ## Plot True Positives Over Time
    xAxisTicks = range(0,len(numTP))
    plt.figure(figsize=(12,9))
    plt.xlabel('Windows',fontweight='bold', fontsize = 25)
    plt.ylabel('Number of True Events Flagged',fontweight='bold', fontsize = 25)
    plt.xticks(fontweight='bold',fontsize = 20)    
    plt.yticks(fontweight='bold',fontsize = 20)
    plt.plot(numFlagged[0:65], color = 'orange', marker = '.',linewidth=4, label = 'Flagged True Events')
    plt.plot(numDecided[0:65], color = 'red',linestyle='--', linewidth=4, label = 'Decided True Events')
    plt.plot(numTP[0:65], color = 'blue', linestyle = '--', linewidth=4, label = 'Number of True Events')
    plt.legend(fontsize=22,)
    plt.grid(axis="x")
    plt.show()
    plt.tight_layout()
    figName =  '/TPOverTime_LINE.png'
    filePath = str(savePath) + figName
    plt.savefig(filePath)
# End of plotTP_Flagged_Decided_LINE function























