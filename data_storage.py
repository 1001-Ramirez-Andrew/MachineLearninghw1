#Author:Andrew Ramirez
#Date:9/28/2022
#programming Assignment 1 CS422


import math
import numpy as np

def build_nparray(data):
    #data is in 2D numpy array form, thus the header needs to be clipped and the rest needs to be cast into double type
    sampleValsString = data[1:, :-1]
    sampleVals = sampleValsString.astype(float)

    labelValsString = data[1: ,-1]
    labelVals = labelValsString.astype(float)

    return sampleVals, labelVals

#essentially the same function as before except changing the np array to a list
def build_list(data):
    sampleValsString = data[1:, :-1]
    sampleVals = sampleValsString.astype(float)

    labelValsString = data[1: ,-1]
    labelVals = labelValsString.astype(float)

    return sampleVals.tolist(), labelVals.tolist()

def build_dict(data):
    #first slice arrays into the correct formats
    features = data[0, :-1]
    sampleData = data[1:, :-1].astype(float)
    labelData = data[1:, -1].astype(int)

    #use zip to create iterator for n-tuples
    numRows= np.shape(data)[0]
    sampleDict = {}
    labelDict = dict(zip(np.array(list(range(numRows))), labelData))
    for i in range(numRows - 1):
        sampleDict[i] = dict(zip(features, sampleData[i]))

    return sampleDict, labelDict