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



This file contains the primary functions to accomplish the meter to transformer
pairing task.  

Function List:
    - RankFlaggingBySweepingThreshold
    - CCTransErrIdent
    - AdjustDistFromThresholdCompMatrix_Distance
    - CorrectFlaggedTransErrors
    



"""



import numpy as np
from copy import deepcopy


def RankFlaggingBySweepingThreshold(transLabelsInput,notMemberThresholdVector,ccMatrix):
    """     This function takes a vector of possible threshold values (probably correlation
            coefficients) and creates a ranked list of flagged transformers based on which
            had relatively lower cc values when they were flagged.
            
            Parameters
            ---------
                transLabelsInput: numpy array of float (1,customers) - the transformer
                    label for each customer
                notMemberThresholdVector: list of float - the list of possible
                    thresholds below which customers are considered not on 
                    the same transformer
                ccMatrix: numpy array of float (customers,customers) - the array
                    of pairwise correlation coefficients.
            Returns
            -------
                allflaggedTrans: list of list of int - each entry in the list
                    contains the list of flagged transformers for that 
                    threshold value.  This allows identification of the
                    threshold value responsible for flagging each 
                    transformer
                allNumFlagged: list of int - the number of flagged transformers
                    for each threshold value.  The length of this list will
                    match the length of notMemberThresholdVector
                rankedFlaggedTrans: list of int - a ranked list containing all
                    flagged transformers over all threshold values. Values at 
                    the beginning of the list had relatively lower cc threshold
                    values when they were flagged.
                rankedTransThresholds: list of float - a list of thresholds that
                    correspond in index to rankedFlaggedTrans denoting which
                    CC threshold flagged that transformer
            """    
    
    allFlaggedTrans = []
    allNumFlagged = []
    # Run through each threshold value
    for thresholdCtr in range(0,len(notMemberThresholdVector)):
        notMemberThreshold = notMemberThresholdVector[thresholdCtr]
        flaggedTrans = CCTransErrIdent(transLabelsInput,notMemberThreshold,ccMatrix)
        allFlaggedTrans.append(flaggedTrans)
        allNumFlagged.append(len(flaggedTrans))
    # End of thresholdCtr for loop
    
    rankedFlaggedTrans = []
    rankedTransThresholds = []
    # Create a ranking of the flagged results
    for thresholdCtr in range(0,len(notMemberThresholdVector)):
        currFlagging = allFlaggedTrans[thresholdCtr]
        currFlagging = set(currFlagging)
        # Remove transformers that were previously flagged and then update the ranked list with the newly flagged transformers
        currFlagging.difference_update(set(rankedFlaggedTrans))
        currFlagging = list(currFlagging)
        for newCtr in range(0,len(currFlagging)):
            rankedFlaggedTrans.append(currFlagging[newCtr])
            rankedTransThresholds.append(notMemberThresholdVector[thresholdCtr])
        
    return allFlaggedTrans, allNumFlagged, rankedFlaggedTrans, rankedTransThresholds
# End of RankFlaggingBySweepingThreshold function


def CCTransErrIdent(transLabelsInput,notMemberThreshold,ccMatrix):
    """ This function uses the correlation coefficients to detect errors in 
        transformer groupings.  This is simply used to flag customers as potential
        errors.
            
            Parameters
            ---------
                transLabelsInput: numpy array of float (1,customers) - the transformer
                    label for each customer
                notMemberThreshold: float - the threshold below which customers
                    are considered not on the same transformer
                ccMatrix: numpy array of float (customers,customers) - the array
                    of pairwise correlation coefficients.
            Returns
            -------
                flaggedCust: list of int - the list of indices of the flagged 
                    customers
                
            """    
            
    transUnique = np.unique(transLabelsInput)
    flaggedTrans = []
    for transCtr in range(0,transUnique.shape[0]):
        currentTrans = transUnique[transCtr]
        currentIndices = np.where(transLabelsInput==currentTrans)[1]
        if len(currentIndices)!=1:
            for indexCtr in range(0,len(currentIndices)):
                currentCust = currentIndices[indexCtr]
                currentCC=ccMatrix[currentCust,currentIndices]
                compIndices = np.where(currentCC<notMemberThreshold)[0]
                if compIndices.shape[0] > 0:
                    flaggedTrans.append(currentTrans)
    flaggedTrans=np.unique(flaggedTrans)
    return flaggedTrans
# End of CCTransErrIdent

##############################################################################
#
#       AdjustDistFromThreshold
#
def AdjustDistFromThreshold(compMatrix,matrix2Adjust,threshold,replacementValue):
    ''' Uses the specified threshold of the comparison matrix to replace values
        in a second matrix.Values less than 0 in the matrix2Adjust are replaced.
        For example, the way this is usually used is if the comparison matrix
        is the MSE matrix and the threshold is 0.2, then for any cell in the MSE 
        matrix with a higher mse than 0.2, the corresponding cell in the second matrix, 
        for example, the reactance distance matrix, is set to the replacement value which is 
        usually a very high value. Effectively this function filters a matrix, like the 
        reactance distance matrix, by high values in the MSE matrix. Values less
        than 0 in the reactance distance matrix are known to be bad values.
        
          Parameters
            ---------
                compMatrix: numpy array of float (1,customers) - the matrix 
                    containing the values used as a threshold to remove values
                    from matrix2Adjust.  This matrix must be a 'distance'-type
                    matrix.  Often this is the pairwise MSE matrix containing
                    the mean-squared error results from a pairwise regression
                    formulation
                matrix2Adjust: numpy array of float (1, customers) - the second matrix
                                that will have values replaced.
                threshold: float - value used to flag values in the comparison matrix above
                                    the threshold for which the corresponding cells in 
                                    matrix2Adjust will be replaced by a given replacement value.
                replacementValue: float - value used to replace flagged cells in matrix2Adjust
                    In the meter to transformer pairing task this is often set to the
                    max value in the matrix2Adjust field, so that the value is 
                    discarded, but remains within the range of the variable
                    
            Returns
            -------
                adjustedMatrix: numpy array of float (1, customers) - the final matrix adjusted
                                by replacing the cells in matrix2Adjust corresponding with the 
                                thresholded cells from compMatrix with a given replacement value.
            
        '''
    
    # Adjust matrix2Adjust (reactance distance) based on a threshold value for compMatrix (mse Matrix)
    #Intuitively this is setting pairs with mse values to high distance for better separation 
    
    adjustedMatrix = deepcopy(matrix2Adjust)
    for rowCtr in range(0,compMatrix.shape[0]):
        for colCtr in range(0,compMatrix.shape[0]):
            if compMatrix[rowCtr,colCtr] > threshold:
                adjustedMatrix[rowCtr,colCtr] =  replacementValue               
            if adjustedMatrix[rowCtr,colCtr] < 0:
                adjustedMatrix[rowCtr,colCtr] = replacementValue
    return adjustedMatrix
# End of AdjustDistFromThreshold
            


               
###############################################################################
#
#           CorrectFlaggedTransErrors

def CorrectFlaggedTransErrors(flaggedTrans,transLabelsInput,custIDInput,
                              ccMatrix,notMemberThreshold, mseMatrix,
                              xDistAdjusted,reactanceThreshold=0.046):
    """ This function takes a list of flagged transformers and produces a list
        of predicted customer labeling errors and a prediction for the correct
        labeling
            
            Parameters
            ---------
                flaggedTrans: list of flagged transformers
                transLabelsInput: numpy array of float (1,customers) - the transformer
                    label for each customer
                custIDInput: list of str - the customer ID's
                ccMatrix: numpy array of float (customers,customers) - the 
                    matrix of pairwise correlation coefficients
                notMemberThreshold: float - the threshold for flagging customers as not
                    being connected to the same transformer as another customer
                mseMatrix: numpy array of float (customers,customers) - the 
                    mse values from a pairwise linear regression
                xDistAdjusted: numpy array of float (customers,customers) - the
                    reactance distance, adjusted by the r-squared values
                reactanceThreshold: float - the threshold for considering a 
                    customer to be the only customer on a transformer. The
                    default value of 0.046 which was determined as an average
                    value for reactance across two transformers.
            Returns
            -------
                predictedTransLabels: numpy array of int (1,customers) - the 
                    predicted transformer labels for each customer. 'New' labels
                    are given by negative labels.  Those should reflect correct
                    transformer groupings but not the original labels themselves.
                allChangedIndices: list of lists of int - each entry in the 
                    list is the list of indices which changed under each
                    successive flagged transformer 
                allChangedOrgTrans: list of lists of int - each entry in the
                    list is the list of the original transformer label for 
                    customers which changed their label
                allChangedpredTrans: list of lists of int - each entry in the
                    list is the list of the predicted transformer label
                    for each customer which changed their label.  'New' 
                    transformer labels will be negative.  Thus the customer
                    groupings should match physical transformers but the 
                    utility label is to be determined.
                
            """    
            
    newTransLabel = -1
    predictedTransLabels = deepcopy(transLabelsInput)
    allChangedIndices = []
    allChangedOrgTrans = []
    allChangedPredTrans = []
    # Loop through each flagged transformer
    for flaggedCtr in range(0,len(flaggedTrans)):
        currentTrans = flaggedTrans[flaggedCtr]
        flaggedIndices = np.where(transLabelsInput == currentTrans)[1]
        pupMatrix = np.zeros((len(flaggedIndices),len(flaggedIndices)),dtype=int)
        
        # Loop through each customer in the transformer and create a paired/unpaired matrix for the transformer
        for custCtr in range(0,len(flaggedIndices)):
            currentIndex=flaggedIndices[custCtr]
            for colCtr in range(0,len(flaggedIndices)):
                if custCtr == colCtr:
                    pupMatrix[custCtr,colCtr] = 1
                else:
                    if ccMatrix[flaggedIndices[custCtr],flaggedIndices[colCtr]] < notMemberThreshold:
                        pupMatrix[custCtr,colCtr] = 0
                    else:
                        pupMatrix[custCtr,colCtr] = 1
                        
        # Evaluate each customer and assign a predicted transformer label
        for custCtr in range(0,len(flaggedIndices)):
            # If the predicted label is negative, then this customer has already been dealt with
            if predictedTransLabels[0,flaggedIndices[custCtr]] < 0:
                continue
            pairedIndices = np.where(pupMatrix[custCtr,:] == 1)[0]
            sortedXDist = np.sort(xDistAdjusted[flaggedIndices[custCtr],:])
            argsortedXDist = np.argsort(xDistAdjusted[flaggedIndices[custCtr],:])
            
            #If the customer is not paired with any of the customers on the transformer it was originally labeled on
            if len(pairedIndices) == 1:
                
                # This is the case where this customer is given its own single-customer transformer
                if sortedXDist[1] >= reactanceThreshold:
                    predictedTransLabels[0,flaggedIndices[custCtr]] = newTransLabel
                    newTransLabel = newTransLabel - 1
                
                # This is the case where this customer is moved to the closest fit transformer label
                else:
                    allPossibleMatchIndices = np.where(xDistAdjusted[flaggedIndices[custCtr],:]<reactanceThreshold)[0]
                    #pairedIndex = argsortedXDist[1]
                    predictedTransLabels[0,allPossibleMatchIndices] = newTransLabel
                    newTransLabel = newTransLabel - 1
                    
            # This is the case where the customer is paired with other customers on that transformer but they are in the minority
            # In the majority case we do nothing
            elif len(pairedIndices) <= (len(flaggedIndices)/2):
                continueFlag = True
                ctr = 1
                while continueFlag:
                    if argsortedXDist[ctr] not in set(flaggedIndices):
                        continueFlag = False
                        # If there is not another customer (not labeled on this transformer) which pairs well with the current customer
                        # Thus the paired, minority, customers on this transformer are given a new transformer label
                        if sortedXDist[ctr] >= reactanceThreshold:
                            predictedTransLabels[0,flaggedIndices[pairedIndices]] = newTransLabel
                            newTransLabel = newTransLabel-1
                        # If there is another customer (not labeled on this transformer) which pairs well with the current customer
                        # Thus the paired, minority customers on this transformer are given the transformer label of the closest paired customer
                        else:
                            allPossibleMatchIndices = np.where(xDistAdjusted[flaggedIndices[custCtr],:]<reactanceThreshold)[0]
                            predictedTransLabels[0,allPossibleMatchIndices] = newTransLabel
                            newTransLabel = newTransLabel - 1
                    ctr = ctr + 1
        changeIndices = np.where(predictedTransLabels != transLabelsInput)[1]
        allChangedIndices.append(changeIndices)
        allChangedOrgTrans.append(transLabelsInput[0,changeIndices])
        allChangedPredTrans.append(predictedTransLabels[0,changeIndices])

    return predictedTransLabels,allChangedIndices,allChangedOrgTrans,allChangedPredTrans
# End of CorrectFlaggedTransErrors

















