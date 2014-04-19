import numpy

A = numpy.zeros((5,5))
E = ['eat', 'go', 'food', 'class', 'chick-fil-a']
H = ['directions', 'hours', 'menu', 'recommendations']
evidenceMatrix = None
probabilityVector = {}
N = {}
x = {}

def getInputVector():
    global A, E, H
    query = raw_input("Please enter a query: ")
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
    if evidenceMatrix != None:
        evidenceMatrix = numpy.vstack((evidenceMatrix, inputVector))
    else:
        probabilityVector[0] = 0.0
        probabilityVector[1] = 1.0
        emptyMatrix = numpy.zeros((2, inputVector.shape[1]))
        emptyMatrix = numpy.vstack((emptyMatrix, inputVector))
        allOnes = numpy.zeros((1, inputVector.shape[1]))
        allOnes.fill(1)
        emptyMatrix = numpy.vstack((emptyMatrix, allOnes))
        evidenceMatrix = numpy.matrix(emptyMatrix)        
    
    inputHash = vectorHash(inputVector)
    for i in range(0, evidenceMatrix.shape[0]):
        if matrixHash(evidenceMatrix[i]) == inputHash:
            setProbability(evidenceMatrix[i], i)
            break


# Get input
for i in range(0, 3):
    inputVector = getInputVector()
    updateEvidenceMatrix(inputVector)
    print (evidenceMatrix)
    for hypothesis in H:
        print (hypothesis + ": ")
        print (probabilityVector[hypothesis])

# Solve system
for hypothesis in H:
    leftSide = numpy.transpose(evidenceMatrix) * evidenceMatrix
    rightSide = numpy.transpose(evidenceMatrix) * probabilityVector[hypothesis]
    x[hypothesis] = numpy.linalg.inv(leftSide) * (rightSide)
    print (x[hypothesis])

# Generate test example
checkVector = getInputVector()

resultVector = numpy.zeros((len(H), 1))
for index, hypothesis in enumerate(H):
    resultVector[index] = checkVector * x[hypothesis]
    
    
# Make Positive
if numpy.min(resultVector) < 0:
    resultVector += -numpy.min(resultVector)

# Normalize
resultVector /= numpy.linalg.norm(resultVector)

print (resultVector)



    
    
