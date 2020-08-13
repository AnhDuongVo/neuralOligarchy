###############################################################################
# plot evolution of weights

import numpy as np
import matplotlib.pyplot as mpl
from scipy.optimize import curve_fit
import scipy.stats as ss
import pickle

def plotWeightsEvol(save_dir, ntype, weights, num_synapses, index):
    """
    This function plots the evolution of synaptic weights.
    Input: 
        save_dir = directory where to save the file ('<directory>')
        ntype = plot for EE or EI weights ('E'/'I')
        weights = array with weights of all synapses, shape is 
                  (<number of synapses>,<number of measurements>)
        num_synapses = number of synapses which weight evolution will be plotted
                       (<int>)
        index = index of serial plot ('<int>')
    Output:
        PDF file with plot of evolution of num_synapses synaptic weights. 
    """
    total_num_synapses = np.shape(weights)[0]
    rand_select = np.random.randint(0,total_num_synapses,num_synapses)
    time_ax = np.arange(0,5*np.shape(weights)[1],5)
    
    fig = mpl.figure(figsize=(12,6))
    for i in rand_select:
        mpl.plot(time_ax, weights[i],linewidth=0.5)
    
    mpl.xlabel("time [ms]")
    mpl.ylabel("weights")
    
    fig.savefig(save_dir + '/weights_evol_' + ntype + index + ".pdf",format="pdf")

with open('data/synee.p', 'rb') as pfile:
    weights_E=pickle.load(pfile)['a'].T

with open('data/synei.p', 'rb') as pfile:
    weights_I=pickle.load(pfile)['a'].T

for i in range(1):
    plotWeightsEvol('data','E',weights_E,5,str(i))
    plotWeightsEvol('data','I',weights_I,3,str(i))


