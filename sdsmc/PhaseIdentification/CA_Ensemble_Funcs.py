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


 CA_Ensemble_Funcs.py

This file contains the functions that implement the Co-Association Matrix
Ensemble Method for the phase identification task.

    Functions:
        -  SPClustering
        -  SPClustering_Precomp
        -  CAEnsemble
   
    
Publications related to this method:
    
L. Blakely and M. J. Reno, “Phase Identification Using Co-Association Matrix Ensemble Clustering,” IET Smart Grid, no. Machine Learning Special Issue, Jun. 2020.
B. D. Pena, L. Blakely, and M. J. Reno, “Parameter Tuning Analysis for Phase Identification Algorithms in Distribution System Model Calibration,” presented at the KPEC, Apr. 2021.
L. Blakely, M. J. Reno, and K. Ashok, “AMI Data Quality And Collection Method Consideration for Improving the Accuracy of Distribution System Models,” presented at the IEEE Photovoltaic Specialists Conference (PVSC), Chicago, IL, USA, 2019.

    
"""


# Import - Python Libraries
from sklearn.cluster import SpectralClustering
import numpy as np
from copy import deepcopy

# Import - Custom Libraries
if __package__ in [None, '']:
    import PhaseIdent_Utils as PIUtils
else:
    from . import PhaseIdent_Utils as PIUtils

###############################################################################
#
#                                  SPClustering
#

def SPClustering(features,k):
    """ This function takes a window of timeseries data for the total number of
         customers and the number of desired clusters and performs the spectral 
         clustering algorithm on that data, returning the cluster labels for each 
         customer.  This is the internal spectral clustering function which is
         called for each window (and each value in kVector).  These results
         are used to build the co-association matrix.  
         The kernel function has been hardcoded here to be the Radial
         Basis Function ('rbf') based on the results of this research.
            
            Parameters
            ---------
                features: numpy array of float (customers,measurments) - a 
                    'window' of time series measurements where customers with
                    missing data are removed.  Any NaN values in this matrix
                    will cause the SpectralClustering function to fail.  
                k:  int - Number of clusters
                
            Returns
            -------
                clusterLabels:  list of int - The resulting cluster label of 
                    each customer (1-k)
            """       
    
    sp = SpectralClustering(n_clusters=k,affinity='rbf')
    clusterLabels = sp.fit_predict(features)    
    return clusterLabels       
# End of SPClustering



###############################################################################
#
#                                  SPClustering_Precomp
#

def SPClustering_Precomp(aggWM,kFinal):
    """ This function takes a precomputed affinity matrix, in the form
        of a co-association matrix generated by CAEnsemble and will
        use that to construct the final clusters representing the three-phases.

            Parameters
            ---------
                aggWM: ndarray of float, shape (customers,customers) affinity 
                    matrix of paired/unpaired weights aggregated over all 
                    available windows.
                kFinal: int - the number of final clusters. This parameter 
                    should be set based on the feeder topology.  Setting this
                    parameter to 4 of 7 is a good place to start.  If the feeder
                    in question has voltage regulating devices a larger number 
                    of final clusters may be required.
                
            Returns
            -------
                clusterLabels:  list of int - The resulting cluster label of 
                    each customer (1-k)
            """       

    sp = SpectralClustering(n_clusters=kFinal,n_init=10,assign_labels='discretize',affinity='precomputed')
    clusterLabels = sp.fit_predict(aggWM)    
    return clusterLabels       
# End of SPClustering_Precomp



###############################################################################
#
#                       CAEnsemble
#

def CAEnsemble(voltage,kVector,kFinal,custID,windowSize,numPhases=-1,lowWindowsThresh=4,printLowWinWarningFlag=True):

    """ This function implements the ensemble of Spectral Clustering  for the
        task of phase identification task.  The ensemble size is determined by 
        the number of sliding windows available given the windowSize parameter.
        In each window the cluster labels are returned by the spectral clustering
        algorithm and that clustering is then used to update a co-association matrix
        based on pairwise paired/unpaired information in the cluster labels.  
        That weight matrix is then used for a final clustering into the final
        clusters which represent phase groupings.  The original utility phase
        labels are not used in this function.  The mapping of the final clusters
        to particular phases is left to a subsequent step.  
        For more details, please see this paper:  
        L. Blakely and M. J. Reno, “Phase Identification Using Co-Association Matrix Ensemble Clustering,” IET Smart Grid, no. Machine Learning Special Issue, Jun. 2020.
            
            Parameters
            ---------
                voltage:  numpy array of float (measurements,customers) -  
                    voltage timeseries for each customer.  The timeseries 
                    should be pre-processed into per-unit, difference (delta)
                    representation.  This pre-processing is an essential step.
                kVector: numpy array of int - a vector of the possible values of
                    k for the windows
                kFinal:  int - Number of clusters for the final clustering
                custID: list of str - list of customer ids
                windowSize:  int - The size (in number of measurements) of the 
                    sliding window
                numPhases: ndarray of int (1,customers) - the indicator for 
                    each customer if it is a single-phase, 2-phase, or 3-phase
                    customer.  This parameter should be supplied be the user
                    or will be estimated using the customer IDs and existing 
                    phase labels.  It is only marked as optional here for 
                    backwards compatibility purposes.                    
                lowWindowsThresh: int - the minimum number of windows before
                    printing a warning that some customers had few windows 
                    due to missing data.  The default value is set to 4 if 
                    this parameter is not specified.
                printLowWinWarningFlag: boolean - allows supression of the printout
                    if customer has only a few windows in the ensemble.  The
                    default value is True.  If a customer is only present in
                    a small number of windows the co-association matrix will not
                    be built adequately for that customer (although it will not
                    affect other customers).  Thus results for customers with
                    few windows should be considered low confidence predictions
                    and likely discarded
                
            Returns
            -------
                finalClusterLabels:  numpy array of int (1,customers) 
                    array of the final cluster labels representing the 
                    phases, but they will not match in number to the actual phases
                    Determining which cluster number goes with which real phase
                    is left for a future step.  This parameter is one that 
                    depends on the topology of the feeder.  For more discussion
                    see the paper by B.D. Pena listed above.  Starting values
                    to try for this parameter might be 4 or 7, topologies with
                    voltage regulators in the feeder may require a larger 
                    number of final clusters.  
                noVotesIndex:  list of int - list of customer indices that 
                    did not recieve any votes (i.e. were removed from all 
                    windows).  This occurs due to missing data for a customer.
                    If all windows for that customer contain missing data, then
                    that customer will be eliminated from the analysis.
                noVotesIDs:  list of str - list of customer ids that did not 
                    receive any votes (i.e. were removed from all windows due 
                    to missing data)
                clusteredIDs:  list of str (customers) - list of customers IDs 
                    that were clustered during the ensemble.  The length of 
                    clusteredIDs plus the length of noVotesIDs should equal
                    the total number of customers
                custWindowCounts: numpy array of int (customers) - the count,
                    for each customer, of the number of windows that were
                    included in the analysis, i.e. the number of windows that 
                    were not excluded due to missing data.  This count is 
                    significantly affected by the value chosen for the 
                    windowSize parameter.  Customers with a low number of 
                    windows in the ensemble should be considered low confidence
                    in the final prediction as they will not populate the 
                    co-association matrix properly.  
            """       
    
    ensTotal = int(np.floor(voltage.shape[0] / windowSize))  # This determines the total number of windows based on available data and window size
    ensPredictedPhases = np.zeros((1,len(custID)),dtype=int)
    aggWM = PIUtils.CreateAggWeightMatrix(custID) # This is the co-assocation matrix
    windowCtr = PIUtils.CreateAggWeightMatrix(custID) # This tracks the number of windows where each pair of customers was included together
    allClusterCounts = []
    custWindowCounts = np.zeros((len(custID)),dtype=int) # This tracks the number of windows used for each customer

    # Loop through each window in the available data
    for ensCtr in range(0,ensTotal):
        print('Ensemble Progress: ' + str(ensCtr) + '/' + str(ensTotal))
        #Select the next time series window and remove customers with missing data in that window
        windowDistances = PIUtils.GetVoltWindow(voltage,windowSize,ensCtr)
        currentDistances,currentIDs = PIUtils.CleanVoltWindowNoLabels(deepcopy(windowDistances), deepcopy(custID))
        custWindowCounts = PIUtils.UpdateCustWindowCounts(custWindowCounts,currentIDs,custID)
       
        # Check for the case where the entire distance matrix is nans
        if ((currentDistances.shape[0] == 1) and (currentDistances.shape[1] == 1)):
            continue
        currentDistances = currentDistances.transpose()
        
        # Loop through each value of k (number of clusters) to use multiple numbers of clusters in each available window
        for kCtr in range(0,len(kVector)):
            k = kVector[kCtr]
            #Check if the cleaning reduced the number of available customers to less than the number of clusters
            if (currentDistances.shape[0] <= k):
                 continue
            #Do the clustering
            clusterLabels = SPClustering(currentDistances,k)
            #Update the weight matrix
            aggWM, windowCtr = PIUtils.UpdateAggWM(clusterLabels,custID,currentIDs,aggWM,windowCtr)  
            #Update Cluster Sizes List
            clusterCounts=np.squeeze(PIUtils.CountClusterSizes(clusterLabels))
            for kCtr2 in range(0,k):
                allClusterCounts.append(clusterCounts[kCtr2])
        #End of kCtr for loop
    # End of ensCtr for loop

    
    # Zero entries for 2-phase and 3-phase customers with themselves so they cannot be clustered together
    # Note that 2-phase and 3-phase datastreams must be adjacent in the indexing!
    custCtr = 0
    while custCtr < len(custID):
        if numPhases[0,custCtr] == 2:
            aggWM[custCtr,(custCtr+1)] = 0
            aggWM[(custCtr+1),custCtr] = 0
            custCtr = custCtr + 2
        elif numPhases[0,custCtr] == 3:
            aggWM[custCtr,(custCtr+1)] = 0
            aggWM[custCtr,(custCtr+2)] = 0
            aggWM[(custCtr+1),custCtr] = 0
            aggWM[(custCtr+1),(custCtr+2)] = 0
            aggWM[(custCtr+2),custCtr] = 0
            aggWM[(custCtr+2),(custCtr+1)] = 0
            custCtr = custCtr + 3
        else:
            custCtr = custCtr + 1
    
    
    #Split customers into customers who had at least one window of data and those that did not
    # If a customer had missing data in all windows then they are not included in the algorithm results
    noVotesIndex = []
    noVotesIDs = []
    for custCtr in range(0,len(custID)):
        if (np.sum(aggWM[custCtr,:]) == 0):
            noVotesIndex.append(custCtr)
            noVotesIDs.append(custID[custCtr])
    clusteredIDs = np.delete(custID,noVotesIndex)
    ensPredictedPhases = np.delete(ensPredictedPhases,noVotesIndex,axis=1)
    aggWM = np.delete(aggWM,noVotesIndex,axis=0)
    aggWM = np.delete(aggWM,noVotesIndex,axis=1)
    windowCtr = np.delete(windowCtr,noVotesIndex,axis=0)
    windowCtr = np.delete(windowCtr,noVotesIndex,axis=1)
    
    if aggWM.shape == (0,0):
        print('Error!  All customers were eliminated from all windows, and the algorithm could not continue.  This is due to missing data in the customers datastreams.  The distribution of missing data was such that there were instances of missing data in every window for every customer.  You could try reducing the window size, but beware that there still may not be many viable windows ')
        return (-1,-1,-1,-1,-1,-1)
    #Normalize aggWM - This is done because each customer would have had different numbers of windows due to missing data, the normalization is done by dividing each cell by the number of windows that pair of customers was both present
    windowCtr[windowCtr==0]=0.0001 #This prevents divide by zero, the aggWM should already be zero in the locations where windowCtr is 0, so 0 will be the end result in that case anway
    aggWM_Norm = np.divide(aggWM,windowCtr)
    aggWM_Norm[np.isnan(aggWM_Norm)] = 0
    aggWM_Norm[aggWM_Norm==0]=0.00001 # The spectral clustering function does not allow zeros (the precomputed matrix must be fully-connected), so any zeros are set to a very small value
    finalClusterLabels = SPClustering_Precomp(aggWM_Norm,kFinal)
    
    # Few windows warning
    numLowWindows = np.where(custWindowCounts <= lowWindowsThresh)[0]
    if printLowWinWarningFlag:
        if len(numLowWindows) != 0:
            print('Warning!  ' + str(len(numLowWindows)) + ' customers had fewer than ' + str(lowWindowsThresh) + ' windows used in the phase identification ensemble.  The predictions for these customers should likely be considered low confidence predictions')
            print('Customer IDs for customers with fewer than ' + str(lowWindowsThresh) + ' windows:')
            for custCtr in range(0,len(numLowWindows)):
                print('Customer ID: ' + str(custID[numLowWindows[custCtr]]) + ' - ' + str(custWindowCounts[numLowWindows[custCtr]]) + ' Windows')
            print('')
        
    return finalClusterLabels,noVotesIndex,noVotesIDs,clusteredIDs,aggWM_Norm,custWindowCounts
# End of CAEnsemble






