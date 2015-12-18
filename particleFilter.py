import numpy as np
import pandas as pd
from matplotlib.mlab import normpdf

# importing the cython file and compile
import pyximport; pyximport.install()
from SSA import forwardSim, forwardSim_cython


class ParticlePopulation(object):

    def __init__(self):
        self.particles = []
        self.weights = []

    def addParticle(self, particle, weight):
        assert weight is not None
        if self.particles != []:
            assert len(particle.state) == len(self.particles[-1].state) # all particle have same dimension

        self.weights.append(weight)
        self.particles.append(particle)

    def get_number_of_particles(self):
        return len(self.particles)

    def sample_from_population_ix(self, nSamples):

        norm_weights = np.array(self.weights)/sum(self.weights)
        ix = np.random.multinomial(nSamples, norm_weights, size=1)

        assert np.sum(ix) == nSamples, 'sth went wrong probly all weights are zero!!'
        return ix

    def sample_from_population_generator(self, nSamples):
        "returns an iterator over samples from the distribution of particles"
        ix = self.sample_from_population_ix(nSamples)[0,:]
        return self.hist_to_list(ix, self.particles)

    def sample_from_population_matrix(self, nSamples):
        "returns a matrix over samples from the distribution of particles"
        nParas = len(self.particles[0].get_plotable_particle()[0])

        matrix = np.zeros((nSamples, nParas))
        for i,q in enumerate(self.sample_from_population_generator(nSamples)):
            matrix[i,:], _ = q.get_plotable_particle()
        return matrix

    def hist_to_list(self, histogram, aList):
        """given a histgram, e,g, [10,30,1] (first element seen 10 times, second 30...)
        turn into list of [a a a a ... b b b b b...]
        """
        assert len(histogram)==len(aList), (len(histogram),"  ", len(aList), "  ",histogram, aList)
        for i, counts in enumerate(histogram):
            for j in range(counts):
                yield aList[i]

    def plot_population_sample(self, sampleSize):
        matrix = self.sample_from_population_matrix(sampleSize)
        df = pd.DataFrame(matrix)
        pd.scatter_matrix(df,figsize=[20,20],marker='x') # c=df.Survived.apply(lambda x:colors[x])

    
class MyParticle(object):
    """encoding parameters and hidden states"""
    def __init__(self, theta, state):
        self.theta = theta
        self.state = state

    def get_plotable_particle(self):
        "returns a nuermic vector and names of its entries, used to plot"
        v = np.concatenate((self.theta, self.state),0)
        names = ['k_{}'.format(i) for i in range(len(self.theta))] + ['s_{}'.format(i) for i in range(len(self.state))]
        return v, names

    def get_weight_single_particle(particle, tau, datapoint):
        "returns a nuermic vector and names of its entries, used to plot"
        # the particle is acutally a gamma distribution over rates and encodes the starting state
        # sample from it!
        theta, state = particle.theta, particle.state

        # get the forward sim, yieling how to update the current particle
        finalState, counter, G = forwardSim(0, state, tau, np.exp(theta))  # see justins thesis for G: integrated props/rate

        #calc weight of particle based on the likelihood
        sigma_measure = 3
        weight = normpdf(datapoint, finalState[1], sigma_measure)

        return weight, finalState, counter, G


class MyGammaParticle(object):
    """encoding a Gamma-distribution over rates. also has hidden states"""
    def __init__(self, alpha, beta, state):
        self.alpha = alpha
        self.beta = beta
        self.state = state

    def get_plotable_particle(self):
        "returns a nuermic vector and names of its entries, used to plot"
        v = np.concatenate((self.alpha, self.beta, self.state), 0)
        names = ['a_{}'.format(i) for i in range(len(self.alpha))] \
                + ['b_{}'.format(i) for i in range(len(self.beta))] \
                + ['s_{}'.format(i) for i in range(len(self.state))]

        return v, names

    def get_weight_single_particle(self, tau, datapoint):
        """
        calculates the weight of the particle in the light of some datapoint
        """
        # the particle is acutally a gamma distribution over rates and encodes the starting state
        # sample from it!
        rates = self.sampleRates(N=1)

        # get the forward sim, yieling how to update the current particle
        finalState, counter, G = forwardSim(0, self.state, tau, rates)  # see justins thesis for G: integrated props/rate

        #calc weight of particle based on the likelihood
        sigma_measure = 5

        # assert  len(datapoint)==2 
        # weight = normpdf(datapoint, finalState[0], 1) #* normpdf(datapoint[1], finalState[1], sigma_measure)
        weight = normpdf(datapoint, finalState[1], sigma_measure)

        return weight, finalState, counter, G

    def sampleRates(self, N):
        """ create N sampels from the Gamma distributiosn this particle represents"""
        assert len(self.alpha) == len(self.beta)
        if N==1:
            return np.random.gamma(shape=self.alpha, scale=1/self.beta)        
        else:
            raise NotImplementedError('not yet implemented for N>1')

def plot_posterior_rates(population, nParticles):
    """a particle encodes a whole distrinbution over rates
    hence sample from this distribution to get the posterior over the rates"""

    samples_per_particle = 100
    ratesSamples = []
    colors = []
    for i,particle in enumerate(population.sample_from_population_generator(nParticles)):
        #sample the rates n tims from this distribution the particle encodes
        q = [particle.sampleRates(1) for i in range(samples_per_particle)]
        w = [i for j in range(samples_per_particle)]
        ratesSamples.append(q)
        colors.append(w)
    ratesSamples = np.vstack(ratesSamples)
    colors =  np.vstack(colors)  

    df = pd.DataFrame(ratesSamples)
    pd.scatter_matrix(df,figsize=[20,20],marker='x',c =colors) # c=df.Survived.apply(lambda x:colors[x])


def multivariate_uniform(N,lb,ub):
    thetas = np.zeros((N,len(lb)))
    for count, (l,u) in enumerate(zip(lb,ub)):
        thetas[:,count] = np.random.uniform(l,u, N)
    return thetas

