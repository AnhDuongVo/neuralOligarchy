###############################################################################
# plot spike raster plot

import numpy as np
from brian2 import *
from scipy.optimize import curve_fit
import scipy.stats as ss
import pickle
import matplotlib.pyplot as mpl
from SORN_plot import SORN_plot
p = SORN_plot(0)
     
###################################################################################################
                 
N_i = 50
save_dir = 'data'
filename = ''

with open(save_dir+'/gexc.p', 'rb') as pfile:
    spikes_e=pickle.load(pfile)

with open(save_dir+'/ginh.p', 'rb') as pfile:
    spikes_i=pickle.load(pfile)

RT = (spikes_e['t']/second)[-1]

CV_e = p.ISI_CV(spikes_e['i'][:],spikes_e['t']/second,'E',N_i,filename,save_dir)
CV_i = p.ISI_CV(spikes_i['i'][:],spikes_i['t']/second,'I',N_i,filename,save_dir)
p.spike_plot(spikes_e['i'][:], spikes_i['i'][:], spikes_e['t']/second, spikes_i['t']/second, N_i, RT-4, RT-1, CV_e, CV_i, filename, save_dir)