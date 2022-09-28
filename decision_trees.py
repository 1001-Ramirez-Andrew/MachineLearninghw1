import numpy as np
from math import log2

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

        probRightYes = np.sum(yesSubset)/yesSubset.size
        probRightNo = 1 - probRightYes
        probLeftYes = np.sum(noSubset)/noSubset.size
        probLeftNo = 1 - probLeftYes

        try:
            entropyRight = -probRightYes*log2(probRightYes) - probRightNo*log2(probRightNo)
        except:
            entropyRight = 0
        try:
            entropyLeft = -probLeftYes*log2(probLeftYes) - probLeftNo*log2(probLeftNo)
        except:
            entropyLeft = 0 
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
        probRightYes = np.sum(subsets[fID][1])/subsets[fID][1].size
        probLeftYes = np.sum(subsets[fID][0])/subsets[fID][0].size
        return db.append(fID).append(int(probLeftYes >= 0.5)).append(int(probRightYes > 0.5))


    #recursive function called based on entropies of branches
    if entropies[fID][0] == 0 and entropies [fID][1] == 0:
        return db.append(fID).append(subsets[fID][0][0]).append(subsets[fID][1][0])
    elif entropies[fID][0] == 0:
        return db.append(fID).append(subsets[fID][0][0]).append(DT_train_binary(sampleSubsetRight, subsets[fID][1], max_depth - 1))
    elif entropies[fID][1] == 0:
        return db.append(fID).append(DT_train_binary(sampleSubsetLeft, subsets[fID][0], max_depth - 1)).append(subsets[fID][1][0])
    else:
        return db.append(fID).append(DT_train_binary(sampleSubsetLeft, subsets[fID][0], max_depth - 1)).append(DT_train_binary(sampleSubsetRight, subsets[fID][1], max_depth - 1))

