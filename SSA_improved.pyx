# cython: profile=True

# the cythonized version of  SSA
import numpy as np
cimport numpy as np
cimport cython

#datatypes
DTYPE_int = np.int
ctypedef np.int_t DTYPE_int_t

DTYPE_float = np.float
ctypedef np.float_t DTYPE_float_t


@cython.boundscheck(False) # turn of bounds-checking for entire function
cdef inline np.ndarray[DTYPE_float_t, ndim=1] propensities(np.ndarray[DTYPE_int_t, ndim=1] state, np.ndarray[DTYPE_float_t, ndim=1] rates):
    cdef int N = rates.shape[0]  # number of reactions
    cdef np.ndarray[DTYPE_float_t, ndim=1] retVal = np.zeros(N)
    retVal[0] = rates[0]
    retVal[1] = rates[1] * state[0]
    retVal[2] = rates[2] * state[0]
    retVal[3] = rates[3] * state[1]
    return retVal

#@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.profile(False)
cdef np.ndarray[DTYPE_float_t, ndim=1] propensities_pointer(np.ndarray[DTYPE_int_t, ndim=1] state, np.ndarray[DTYPE_float_t, ndim=1] rates, np.ndarray[DTYPE_float_t, ndim=1] retVal):
    retVal[0] = rates[0]
    retVal[1] = rates[1] * state[0]
    retVal[2] = rates[2] * state[0]
    retVal[3] = rates[3] * state[1]
    return retVal

@cython.profile(False)
cdef inline int getReaction(np.ndarray[DTYPE_float_t, ndim=1] a, float r):
    cdef int i = 0
    cdef int nReactions = a.shape[0]
    cdef float accumulator = 0
    for i in range(nReactions):
        accumulator += a[i]
        if r < accumulator:
            return i
    return i


def forwardSim_cython(float iniT, np.ndarray[DTYPE_int_t, ndim=1] iniS,
                      float maxTime, np.ndarray[DTYPE_float_t, ndim=1] rates):
    """
    simply forward sim of stoch model, 
    returning final state and number of reactions and the integrated props
    """
    cdef int i
    cdef np.ndarray[DTYPE_int_t, ndim=2] V = np.array([[1, 0], [-1, 0], [ 0, 1], [0, -1]], dtype=DTYPE_int);

    assert rates.shape[0] == 4, 'Reaction rates have to be 4'

    cdef int nReactions = V.shape[0]
    cdef int nSpecies = V.shape[1]
    cdef float time = iniT

    cdef np.ndarray[DTYPE_int_t, ndim=1] reactionCounter = np.zeros(nReactions, dtype=DTYPE_int)
    cdef np.ndarray[DTYPE_int_t, ndim=1] theState = np.zeros(nSpecies, dtype=DTYPE_int)
    for i in range(nSpecies): #copy the content explicilty, otherwise modifing theState causes sideeffects
        theState[i] = iniS[i]

    cdef np.ndarray[DTYPE_float_t, ndim=1] integrated_props = np.zeros(nReactions, dtype=DTYPE_float)
    cdef np.ndarray[DTYPE_float_t, ndim=1] props = np.zeros(nReactions, dtype=DTYPE_float)
    cdef np.ndarray[DTYPE_float_t, ndim=1] a_norm = np.zeros(nReactions, dtype=DTYPE_float)

    cdef float a0
    cdef float tau
    cdef int mu
    cdef int mu2
    cdef float r

    while time < maxTime:
        # props = propensities(theState, rates)
        propensities_pointer(theState, rates, props)  # calculates new propensities, modifies props

        "a0 = np.sum(props)"
        a0 = 0
        for i in range(nReactions):
            a0 += props[i]

        " a_norm = props/a0"
        for i in range(nReactions):
            a_norm[i] = props[i]/a0

        tau = np.random.exponential(1/a0)
        r = np.random.rand()
        mu = getReaction(a_norm, r)

        #in case the next reaction happends beyond the intervakl of interest:
        if time + tau < maxTime:
            time = time + tau

            "theState = theState + V[mu,:]"
            for i in range(nSpecies):
                theState[i] = theState[i] + V[mu,i]

            reactionCounter[mu] += 1
            
            "integrated_props[i] += props[i]*tau"
            for i in range(nReactions):
                integrated_props[i] += props[i]*tau
        else:
            #dont upadte the state, no reaction happened before finished
            "integrated_props += props*(maxTime-time)"  # integrate over the remaining time in the interva;
            for i in range(nReactions):
                integrated_props[i] += props[i]*(maxTime-time)
            time = maxTime

    cdef np.ndarray[DTYPE_float_t, ndim=1] G = integrated_props/rates
    return theState, reactionCounter, G   #  , timeVector,stateVector
