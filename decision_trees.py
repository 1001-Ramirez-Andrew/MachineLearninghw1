#Author:Andrew Ramirez
#Date:9/28/2022
#programming Assignment 1 CS422

import numpy as np
from math import log2, floor

def DT_train_binary(X, Y, max_depth):
    #the binary dicision tree will be implemented using nested lists
    db = []
    #to make the tree will make recursive function calls since the problem is solved the same except on a smaller set of samples at each branch
    #for features in data
    numCols= np.shape(X)[1]
    Xbool = X.astype(bool)

    probYes = np.sum(Y)/Y.size
    probNo = 1 - probYes
    #if either of these has a probability of zero, then the etropy will be zero (because log throws error)
    try:
        entropy = -probYes*log2(probYes) - probNo*log2(probNo)
    except:
        entropy = 0

    #THe IG for each feature is stored in list and the max will give the index of the feature to use
    IGvals = []
    entropies = []
    subsets = []
    for i in range(numCols):
        featureVal = Xbool[ : , i]
        yesSubset = Y[featureVal]
        noSubset = Y[np.invert(featureVal)]

        try:
            probRightYes = float(np.sum(yesSubset))/float(yesSubset.size)
            probRightNo = 1 - probRightYes
        except:
            probRightYes = 0.0
            probRightNo = 0.0
        try:
            probLeftYes = float(np.sum(noSubset))/float(noSubset.size)
            probLeftNo = 1 - probLeftYes
        except:
            probLeftYes = 0.0
            probLeftNo = 0.0

        try:
            entropyRight = -probRightYes*log2(probRightYes) - probRightNo*log2(probRightNo)
        except:
            entropyRight = 0.0
        try:
            entropyLeft = -probLeftYes*log2(probLeftYes) - probLeftNo*log2(probLeftNo)
        except:
            entropyLeft = 0.0
        weightLeft = noSubset.size/Y.size
        weightRight = yesSubset.size/Y.size
        IGvals.append(entropy - weightLeft*entropyLeft - weightRight*entropyRight)
        entropies.append((entropyLeft, entropyRight))
        subsets.append((noSubset, yesSubset))

    maxIG = max(IGvals)
    fID = IGvals.index(maxIG)
    sampleSubsetRight = X[Xbool[ : , fID]]
    sampleSubsetLeft = X[np.invert(Xbool[ : , fID])]

    #Check for max depth, then make guess
    if max_depth == 1:
        try:
            probRightYes = float(np.sum(subsets[fID][1]))/float(subsets[fID][1].size)
        except:
            probRightYes = 0
        try:
            probLeftYes = float(np.sum(subsets[fID][0]))/float(subsets[fID][0].size)
        except:
            probLeftYes = 0
        db.append(fID)
        db.append(int(probLeftYes >= 0.5))
        db.append(int(probRightYes > 0.5))
        return db

    #recursive function called based on entropies of branches
    if entropies[fID][0] == 0 and entropies [fID][1] == 0:
        db.append(fID)
        db.append(subsets[fID][0][0])
        db.append(subsets[fID][1][0])
        return db
    elif entropies[fID][0] == 0:
        db.append(fID)
        db.append(int(subsets[fID][0][0]))
        db.append(DT_train_binary(sampleSubsetRight, subsets[fID][1], max_depth - 1))
        return db
    elif entropies[fID][1] == 0:
        db.append(fID)
        db.append(DT_train_binary(sampleSubsetLeft, subsets[fID][0], max_depth - 1))
        db.append(int(subsets[fID][1][0]))
        return db
    else:
        db.append(fID)
        db.append(DT_train_binary(sampleSubsetLeft, subsets[fID][0], max_depth - 1))
        db.append(DT_train_binary(sampleSubsetRight, subsets[fID][1], max_depth - 1))
        return db

def DT_make_prediction(x, DT):
    #if x is sample then each feature will index the sample
    try:
        isYes = bool(x[DT[0]])
    except:
        return 1
    #THe base case will occur when the value at branch is an int instead of list
    #THe first index of the DT will be the feature value
    if isYes:
        if type(DT[2]) == type(int(1)):
            return DT[2]
        else:
            return DT_make_prediction(x, DT[2])
    else:
        if type(DT[1]) == type(int(1)):
            return DT[1]
        else:
            return DT_make_prediction(x, DT[1])
    
def DT_test_binary(X, Y, DT):
    numSamples = len(X)
    numCorrect = 0
    for i in range(numSamples):
        prediction = DT_make_prediction(X[i], DT)
        if prediction == Y[i]:
            numCorrect = numCorrect + 1
    return numCorrect/numSamples

def RF_build_random_forest(X, Y, max_depth, num_of_trees):
    #calculate the numer of samples to randomly select
    numSamples = len(X)
    numSubSample = int(floor(.10*numSamples))
    #combine the sample values and labels into one
    cols = []
    for i in range(len(Y)):
        cols.append([Y[i]])
    X = np.append(X, cols, axis = 1)
    rf = []
    for i in range(num_of_trees):
        indecies = list(range(len(X)))
        subIndicies = np.random.choice(indecies, size = numSubSample, replace = False)
        W = X[subIndicies]
        try:
            dt = DT_train_binary(W[ : , :-1], W[ : , -1], max_depth)
            accuracy = DT_test_binary(W[ : , :-1], W[ : , -1], dt)
        except:
            dt = 0
            accuracy = 0
        rf.append(dt)
        print("DT", i, ": ", accuracy)
    return rf 

def RF_test_random_forest(X, Y, RF):
    
    isCorrect = []
    for i in range(len(X)):
        sample = X[i]
        predictions = []
        for trees in RF:
           prediction = DT_make_prediction(sample, trees)
           predictions.append(prediction)
        majority = sum(predictions)/len(predictions)
        if majority >=.5:
            guess = 1
        else:
            guess = 0
        isCorrect.append(int(guess == Y[i]))

    #calculate accuracy
    accuracy = sum(isCorrect)/len(isCorrect) 
    return accuracy
            


        

        


