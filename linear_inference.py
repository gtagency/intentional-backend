import numpy
import os
import pickle

PICKLE_FILE = "inference.p"

if not os.path.isfile(PICKLE_FILE):
    A = numpy.zeros((5,5))
    E = ['eat', 'go', 'food', 'class', 'chick-fil-a']
    H = ['directions', 'hours', 'menu', 'recommendations']
    evidenceMatrix = None
    probabilityVector = {}
    N = {}
    x = {}
    dataDict = {}
else:
    dataDict = pickle.load(open(PICKLE_FILE, "rb"))
    A = dataDict['A']
    E = dataDict['E']
    H = dataDict['H']
    evidenceMatrix = dataDict['EM']
    probabilityVector = dataDict['PV']
    N = dataDict['N']
    x = dataDict['x']
    
# Main Program

def getInputVector():
    global A, E, H
    query = raw_input("Please enter a query (enter to finish): ")
    query = query.split()
    vector = numpy.zeros((A.shape[0], 1))
    for word in query:
        if word.lower() not in E:
            E.append(word.lower())
            A = numpy.hstack((A, numpy.zeros((A.shape[0], 1))))
            vector = numpy.vstack((vector, numpy.zeros((1, 1))))

        vector[E.index(word.lower())][0] = 1.0
    
    return numpy.transpose(vector)
    
def vectorHash(array):
    vectorHash = 0
    vector = array.flatten()
    for i in range(0, vector.shape[0]):
        vectorHash += vector[i] * (2**i)
    return vectorHash
    
def matrixHash(array):
    vectorHash = 0
    for i in range(0, array.shape[1]):
        vectorHash += array[0,i] * (2**i)
    return vectorHash

def setProbability(row, index):
    global N, H, probabilityVector
    vHash = matrixHash(row)
    if vHash not in N.keys():
        N[vHash] = 1
    else:
        N[vHash] += 1
    for hypothesis in H:
        if hypothesis not in probabilityVector.keys():
            probabilityVector[hypothesis] = numpy.zeros((evidenceMatrix.shape[0],1))
        
        if index >= probabilityVector[hypothesis].shape[0]:
            probabilityVector[hypothesis] = numpy.vstack((probabilityVector[hypothesis], numpy.zeros((1, 1))))

        answer = raw_input(hypothesis + " [y/n]: ")
        if answer.lower() == "y":
            events = probabilityVector[hypothesis][index][0] * (N[vHash] - 1)
            probabilityVector[hypothesis][index][0] = (events + 1) / float(N[vHash])

def updateEvidenceMatrix(inputVector):
    global H, evidenceMatrix
    inputHash = vectorHash(inputVector)
    
    update = False

    if evidenceMatrix != None:
        for i in range(0, evidenceMatrix.shape[0]):
            if matrixHash(evidenceMatrix[i]) == inputHash:
                update = True
    
    if evidenceMatrix != None and not update:
        evidenceMatrix = numpy.vstack((evidenceMatrix, inputVector))
    elif not update:
        probabilityVector[0] = 0.0
        probabilityVector[1] = 1.0
        emptyMatrix = numpy.zeros((2, inputVector.shape[1]))
        emptyMatrix = numpy.vstack((emptyMatrix, inputVector))
        allOnes = numpy.zeros((1, inputVector.shape[1]))
        allOnes.fill(1)
        emptyMatrix = numpy.vstack((emptyMatrix, allOnes))
        evidenceMatrix = numpy.matrix(emptyMatrix)       
    
    for i in range(0, evidenceMatrix.shape[0]):
        if matrixHash(evidenceMatrix[i]) == inputHash:
            setProbability(evidenceMatrix[i], i)
            break


# Get input
keepLooping = True
while keepLooping:
    inputVector = getInputVector()
    if numpy.sum(inputVector) > 0:
        updateEvidenceMatrix(inputVector)
    else:
        keepLooping = False

# Solve system
for hypothesis in H:
    leftSide = numpy.transpose(evidenceMatrix) * evidenceMatrix
    rightSide = numpy.transpose(evidenceMatrix) * probabilityVector[hypothesis]
    x[hypothesis] = numpy.linalg.inv(leftSide) * (rightSide)

# Generate test example
print ("Enter a different query to check: ")
keepAsking = True
while keepAsking:
    checkVector = getInputVector()

    if numpy.sum(checkVector) > 0:
        resultVector = numpy.zeros((len(H), 1))
        for index, hypothesis in enumerate(H):
            resultVector[index] = checkVector * x[hypothesis]
    
        # Make Positive
        if numpy.min(resultVector) < 0:
            resultVector += -numpy.min(resultVector)

        # Normalize
        if numpy.linalg.norm(resultVector) != 0:
            resultVector /= numpy.linalg.norm(resultVector)

        print (resultVector)
    else:
        keepAsking = False

# Pickle the data

dataDict['A'] = A
dataDict['E'] = E
dataDict['H'] = H
dataDict['EM'] = evidenceMatrix
dataDict['PV'] = probabilityVector
dataDict['N'] = N
dataDict['x'] = x

pickle.dump(dataDict, open(PICKLE_FILE, "wb"))