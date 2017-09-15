# cython: profile=True

import numpy as np
# standard SSA
def forwardSim(iniT, iniS, maxTime, rates):
    """ 
    simply forward sim of stoch model, 
    returning final state and number of reactions and the integrated props"""
    species = ["mRNA", "Protein"]
    V = np.array([[1, 0],
        [-1, 0],
        [ 0, 1],
         [0, -1]]);
    
    assert len(rates) == 4, 'Reaction rates have to be 4'

    propensities = lambda state: np.array([rates[0], rates[1]*state[0], rates[2]*state[0], rates[3]*state[1]])
    nReactions, nSpecies = V.shape
    
    reactionCounter = np.zeros(nReactions)

    time = iniT
    theState = iniS
    integrated_props = np.zeros(nReactions)

    #timeVector = [iniT]
    #stateVector = [iniS]
    while time < maxTime:

        props = propensities(theState)
        a0 = sum(props)
        a_norm = props/a0
        tau = np.random.exponential(1/a0)

        mu = np.where(np.cumsum(a_norm) > np.random.rand())[0][0]  # categorial rv witrh probs a_norm
        #mu = np.random.choice(range(nReactions), p=a_norm)
        #in case the next reaction happends beyond the intervakl of interest:
        if time + tau<maxTime:
            time = time + tau
            theState = theState + V[mu,:]
            reactionCounter[mu] += 1
            integrated_props += props*tau
        else:
            #dont upadte the state, no reaction happened before finished
            integrated_props += props*(maxTime-time)  # integrate over the remaining time in the interva;
            time = maxTime
        #timeVector.append(time)
        #stateVector.append(theState)

    G = integrated_props/rates
    return theState, reactionCounter, G   #  , timeVector,stateVector



# the cythonized version of  SSA
cimport numpy as np
cimport cython
@cython.profile(False)
cdef inline np.ndarray propensities(np.ndarray state, np.ndarray rates): return np.array([rates[0], rates[1]*state[0], rates[2]*state[0], rates[3]*state[1]])

cdef inline int getReaction(np.ndarray a, float r):
    cdef int i = 0
    cdef int nReactions = a.shape[0]

    cdef float accumulator = 0
    for i in range(nReactions):
        accumulator += a[i]
        if r < accumulator:
            return i

    return i

def forwardSim_cython(float iniT, np.ndarray iniS, float maxTime, np.ndarray rates):
    """
    simply forward sim of stoch model, 
    returning final state and number of reactions and the integrated props
    """
    cdef np.ndarray V = np.array([[1, 0],
        [-1, 0],
        [ 0, 1],
         [0, -1]], dtype=np.int);
    assert len(rates) == 4, 'Reaction rates have to be 4'

    #propensities = lambda state: np.array([rates[0], rates[1]*state[0], rates[2]*state[0], rates[3]*state[1]])

    cdef int nReactions = V.shape[0]
    cdef int nSpecies = V.shape[1]

    cdef np.ndarray reactionCounter = np.zeros(nReactions, dtype=np.int)
    cdef float time = iniT

    cdef np.ndarray theState = np.zeros(nSpecies, dtype=np.int)
    theState = iniS

    cdef np.ndarray integrated_props = np.zeros(nReactions, dtype=np.float)

    cdef np.ndarray props = np.zeros(nReactions, dtype=np.float)
    cdef np.ndarray a_norm = np.zeros(nReactions, dtype=np.float)
    cdef float a0
    cdef float tau
    cdef int mu
    cdef int mu2
    cdef float r
    #timeVector = [iniT]
    #stateVector = [iniS]
    while time < maxTime:
        props = propensities(theState, rates)
        a0 = np.sum(props)
        a_norm = props/a0
        tau = np.random.exponential(1/a0)

        r = np.random.rand()
#        mu = np.where(np.cumsum(a_norm) > r)[0][0]  # categorial rv witrh probs a_norm

        mu = getReaction(a_norm, r)

        # assert mu==mu2
        #mu = np.random.choice(range(nReactions), p=a_norm)
        #in case the next reaction happends beyond the intervakl of interest:
        if time + tau < maxTime:
            time = time + tau
            theState = theState + V[mu,:]
            reactionCounter[mu] += 1
            integrated_props += props*tau
        else:
            #dont upadte the state, no reaction happened before finished
            integrated_props += props*(maxTime-time)  # integrate over the remaining time in the interva;
            time = maxTime
        #timeVector.append(time)
        #stateVector.append(theState)

    cdef np.ndarray G = integrated_props/rates
    return theState, reactionCounter, G   #  , timeVector,stateVector


