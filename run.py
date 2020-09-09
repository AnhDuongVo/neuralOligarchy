import sys
import os
from brian2 import *
set_device('cpp_standalone',directory='./builds',build_on_run=False)
import numpy as np
import time
import matplotlib.pyplot as mpl
from cpp_methods import syn_scale, syn_EI_scale
import pickle
from SORN_plot import SORN_plot
p = SORN_plot(0)
from random import randint, uniform

# set brian 2 and numpy random seeds
seed(578)
np.random.seed(589)

##########################################################
# Functions
##########################################################

time1 = time.time()

def set_active(*argv):
    """ Input: List argv with monitors
        Goal: Sets the elements in argv for the run active to be recorded """
    for net_object in argv:
        net_object.active=True

def set_inactive(*argv):
    """ Input: List argv with monitors
        Goal: Sets the elements in argv for the run inactive to not be 
              recorded """
    for net_object in argv:
        net_object.active=False
        
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
    time_ax = np.arange(0,20*np.shape(weights)[1],20)
    
    fig = mpl.figure(figsize=(12,6))
    for i in rand_select:
        mpl.plot(time_ax, weights[i],linewidth=0.5)
    
    mpl.xlabel("time [ms]")
    mpl.ylabel("weights")
    
    fig.savefig(save_dir + '/weights_evol_' + ntype + index + ".pdf",format="pdf")

def synnorm(connection_target,total_in):
    for i in range(shape(connection_target.W)[1]):
        sum_in = connection_target.W[:,i].sum()
        if not sum_in == 0:
            connection_target.W[:,i] = total_in * connection_target.W.todense()[:,i] / sum_in

def gaussian(x,u,s):
    g = (2/sqrt(2*pi*s*s))*exp(-(x-u)*(x-u)/(2*s*s))
    return g

def get_sepMatrix(N_src, N_tar,X_src,X_tar, width_T, size_T, selfConn):
    D = ndarray(shape=(N_src,N_tar))  # non-periodic e->e separation
    for i in range(N_src):
        for j in range(N_tar):
            Dx = X_src[i,0]-X_tar[j,0]
            Dy = X_src[i,1]-X_tar[j,1]
            D[i,j] = sqrt(Dx**2 + Dy**2)
    
    P = ndarray(shape=(N_src,N_tar))  #non-periodic e->e probability
    
    for i in range(N_src):
        for j in range(N_tar):
            P[i,j] = gaussian(D[i,j],0,width_T)
    if selfConn:
        for i in range(N_src):
            P[i,i] = 0  # prevent self-connections
    return(P)

def get_connections(N_tar, N_src, sparse, P):
    already_connected = []
    src = []
    tar = []
    n_new = int(round(N_src*N_tar*sparse))
    for i in range(n_new):
        addition_allowed = False
        new_i = randint(0, N_src-1)
        new_j = randint(0, N_tar-1)
        while not(addition_allowed):
            new_i = randint(0, N_src-1)
            new_j = randint(0, N_tar-1)
            if uniform(0,1) < P[new_i,new_j]:
                if not(new_i == new_j) and not([new_i,new_j] in already_connected):
                    addition_allowed = True
        already_connected.append([new_i,new_j])     
        src.append(new_i)
        tar.append(new_j)
        print(N_src, N_tar, i)
    return(src, tar)

##########################################################
# Parameter Values 
##########################################################

# Time parameters 
    # different runs i=1,2,3 with time Ti to record different parameters
    # see recorded parameters under section "Run Simulation"
T1 = 0                 # time of first run in s
number_T1 = int(T1/100)     # number of partial runs to decrease use of RAM
T2 = 0
number_T2 = int(T2/100) 
T3 = 500
number_T3 = int(T3/100)

RT = T1+T2+T3               # total runtime

# Base parameters
N_e=400
N_i=int(0.2*N_e)
size_T = 1000          # sheet length, microns
width_T = 200           # growth radius, microns
sparse_eTOe = 0.1     # recurrent excitatory sparseness
sparse_iTOe = 0.1     # inhibitory to excitatory sparseness
sparse_eTOi = 0.1     #  excitatory to inhibitory  sparseness
sparse_iTOi = 0.5     # inhibitory to inhibitory sparseness
wi_eTOe = 1.5 * mV     # max initial e->e weight
wi_eTOi = 1.5 * mV     # max initial e->i weight
wi_iTOe = -1.5 * mV    # max initial i->e weight
wi_iTOi = -1.5 * mV    # max initial i->e weight
delay_eTOe = 1.5 * ms  # e->e latency for 1 mm
delay_eTOi = 0.5 * ms  # e->i latency for 1 mm
delay_iTOe = 1.0 * ms  # i->e latency for 1 mm
delay_iTOi = 1.0 * ms  # i->i latency for 1 mm
syn_mod_dt = 100*ms       # dt between update of state variables


# Neuron parameters
sigma_noise = sqrt(5.0) * mV # noise amplitude
tau = 20 * ms          # membrane time constant
Vr_e = -70 * mV        # excitatory reset value
Vr_i = -60 * mV        # inhibitory reset value
El = -60 * mV          # resting value
Vti = 55 * mV          # minus maximum initial threshold voltage
Vtvar = 5 * mV         # maximum initial threshold voltage swing
Vvi = 50 * mV          # minus maximum initial voltage
Vvar = 20 * mV         # maximum initial voltage swing

# STDP
taupre = 15 * ms       # pre-before-post STDP time constant
taupost = taupre * 2.0 # post-before-pre STDP time constant
Ap = 15.0 * mV         # potentiating STDP learning rate
Ad = -Ap * 0.5         # depressing STDP learning rate

LTD_a = 0.000005 * mV

# normalization
eta_scaling = 0.25
total_in_eTOe = 40 * mV
total_in_iTOe = 12 * mV
total_in_eTOi = 60 * mV
total_in_iTOi = 60 * mV
dt_synEE_scaling = 1000*ms 

# monitoring
synEE_dt = 20000.*ms         # time step of EE recording of synaptic traces
synEI_dt = 20000.*ms         # time step of EI recording of synaptic traces

##########################################################
# Build neuron model 
##########################################################
noisylif = Equations('''
dV/dt = -(V-El)/tau + sigma_noise*xi/(tau**.5) : volt
Vt : volt 

AsumEE : volt
AsumEI : volt

ANormTar : volt
iANormTar : volt
''')

G_e = NeuronGroup(N = N_e, model = noisylif, threshold ='V > Vt', reset='V=Vr_e')  # excitatory group
G_i = NeuronGroup(N = N_i, model = noisylif, threshold ='V > Vt', reset='V=Vr_i')  # inhibitory group

### Randomize initial voltages
G_e.V = -(Vvi + rand(N_e) * Vvar) # starting membrane potential
G_i.V = -(Vvi + rand(N_i) * Vvar) # starting membrane potential

### Randomize initial thresholds
G_e.Vt = -(Vti + rand(N_e) * Vtvar)  # starting threshold
#G_i.Vt = -(Vti + rand(N_i) * Vtvar)  # starting threshold
G_i.Vt = -58 * mV

##########################################################
# Topology
##########################################################
X_e = size_T * rand(N_e,2)
X_i = size_T * rand(N_i,2)

P_eTOe = get_sepMatrix(N_e, N_e,X_e,X_e, width_T, size_T, True)
P_eTOi = get_sepMatrix(N_e, N_i,X_e,X_i, width_T, size_T, False)
P_iTOe = get_sepMatrix(N_i, N_e,X_i,X_e, width_T, size_T, False)
P_iTOi = get_sepMatrix(N_i, N_i,X_i,X_i, width_T, size_T, False)

print("Separation matrices initialized.")

##########################################################
# Build synapse model 
##########################################################

synEE_mod = '''
w : volt 
dApre/dt = -Apre/taupre  : volt (event-driven)
dApost/dt = -Apost/taupost : volt (event-driven)
AsumEE_post = w : volt (summed)
'''

synEE_pre_mod = '''
V_post += w
Apre += Ap
w = clip(w+Apost, 0, total_in_eTOe)
'''

synEE_post_mod = '''
Apost += Ad
w = clip(w+Apre, 0, total_in_eTOe)
'''

SynEE = Synapses(target=G_e, source=G_e, method='euler', model=synEE_mod,
on_pre=synEE_pre_mod, on_post=synEE_post_mod,
dt=syn_mod_dt)

synEI_mod = '''
w : volt 
dApre/dt = -Apre/taupre  : volt (event-driven)
dApost/dt = -Apost/taupost : volt (event-driven)
AsumEI_post = w : volt (summed)
'''

synEI_pre_mod = '''
V_post += w
Apre += Ap
w -= LTD_a
w = clip(w+Apost, 0, total_in_iTOe)
'''

synEI_post_mod = '''
Apost += Ad
w = clip(w+Apre, 0, total_in_iTOe)
'''

SynEI = Synapses(target=G_e, source=G_i, method='euler', model=synEI_mod,
on_pre=synEI_pre_mod, on_post=synEI_post_mod,
dt=syn_mod_dt)

SynIE = Synapses(target=G_i, source=G_e, method='euler', on_pre='V_post += wi_eTOi')
SynII = Synapses(target=G_i, source=G_i, method='euler', on_pre='V_post += wi_iTOi')

sEE_src, sEE_tar = get_connections(N_e, N_e, sparse_eTOe, P_eTOe)
SynEE.connect(i=sEE_src, j=sEE_tar)

sEI_src, sEI_tar = get_connections(N_e, N_i, sparse_eTOi, P_iTOe) 
SynEI.connect(i=sEI_src, j=sEI_tar)

sIE_src, sIE_tar = get_connections(N_i, N_e, sparse_iTOe, P_eTOi)
SynIE.connect(i=sIE_src, j=sIE_tar)

sII_src, sII_tar = get_connections(N_i, N_i, sparse_iTOi, P_iTOi)
SynII.connect(i=sII_src, j=sII_tar)

SynEE.delay = delay_eTOe
SynEI.delay = delay_iTOe
SynIE.delay = delay_eTOi
SynII.delay = delay_iTOi

SynEE.w = wi_eTOe
SynEI.w = wi_eTOi

print("Connections initialized.")
    
##########################################################
# Normalization
##########################################################

synEE_scaling = '''
                w = syn_scale(w, ANormTar, AsumEE_post, eta_scaling)
                '''

synEI_scaling = '''
                w = syn_EI_scale(w, iANormTar, AsumEI_post, eta_scaling)
                '''

G_e.ANormTar = total_in_eTOe
SynEE.summed_updaters['AsumEE_post']._clock = Clock(dt=dt_synEE_scaling)
SynEE.run_regularly(synEE_scaling,dt=dt_synEE_scaling,when='end')


G_e.iANormTar = total_in_eTOi
SynEI.summed_updaters['AsumEI_post']._clock = Clock(dt=dt_synEE_scaling)
SynEI.run_regularly(synEI_scaling,dt=dt_synEE_scaling,when='end')
    

##########################################################
# Monitoring 
##########################################################    

M_ee = StateMonitor(SynEE, ['w'],
                              record=range(50),
                              when='end', dt=synEE_dt)

M_ei = StateMonitor(SynEI, ['w'],
                          record=range(50),
                          when='end', dt=synEI_dt)

# monitor of spikes
GExc_spks = SpikeMonitor(G_e[:50])    
GInh_spks = SpikeMonitor(G_i[:50])

##########################################################
# Run Simulation
##########################################################  

net = Network(collect())  

print("Zeit0:",time.time()-time1)

time1 = time.time()

T1_recorders = [GExc_spks, GInh_spks, M_ee, M_ei] 
set_active(*T1_recorders)
set_inactive(*T1_recorders)
# T1: no active monitors to wait for stabilised network 
for index in range(number_T1):
    net.run(int(T1/number_T1)*second,report='text')


# T2: Record weights of EE and EI synapses 
T2_recorders = [M_ee, M_ei]
set_active(*T2_recorders)
for index in range(number_T2):
    net.run(int(T2/number_T2)*second,report='text')


# T3: Record weights of EE and EI synapses and spike times of exc and 
# inh neurons 
T3_recorders = [GExc_spks, GInh_spks, M_ee, M_ei]     
set_inactive(*T2_recorders)
set_active(*T3_recorders)
for index in range(number_T3):
    net.run(int(T3/number_T3)*second,report='text')


device.build(directory='builds',clean=True,compile=True, run=True, debug=False)

print("Zeit1:",time.time()-time1)

##########################################################
# Save spike times, synaptic connections, and EE and EI synaptic weights
##########################################################

time1 = time.time()

if not os.path.exists('data'):
    os.makedirs('data')
    
save_dir = 'data'

# save spike times
with open(save_dir+'/ginh.p','wb') as pfile:
    pickle.dump(GInh_spks.get_states(['i','t']),pfile)

with open(save_dir+'/gexc.p','wb') as pfile:
    pickle.dump(GExc_spks.get_states(['i','t']),pfile)

# save synaptic end weights end
with open(save_dir+'/synee_end.p','wb') as pfile:
    pickle.dump(M_ee.w.T[-1],pfile)

with open(save_dir+'/synei_end.p','wb') as pfile:
    pickle.dump(M_ei.w.T[-1],pfile)

# save synaptic connections
with open(save_dir+'/sEE_src.p','wb') as pfile:
    pickle.dump(sEE_src,pfile)

with open(save_dir+'/sEI_src.p','wb') as pfile:
    pickle.dump(sEI_src,pfile)
    
with open(save_dir+'/sEE_tar.p','wb') as pfile:
    pickle.dump(sEE_tar,pfile)

with open(save_dir+'/sEI_tar.p','wb') as pfile:
    pickle.dump(sEI_tar,pfile)
    
# save synaptic end weights
with open(save_dir+'/synee.p','wb') as pfile:
    pickle.dump(M_ee.get_states(['w']),pfile)

with open(save_dir+'/synei.p','wb') as pfile:
    pickle.dump(M_ei.get_states(['w']),pfile)

##########################################################
# Plot weights and spike plot
##########################################################

for i in range(3):
    plotWeightsEvol('data','E',M_ee.w,5,str(i))
    plotWeightsEvol('data','I',M_ei.w,5,str(i))
    
N_i_new = 50
save_dir = 'data'
filename = ''

with open(save_dir+'/gexc.p', 'rb') as pfile:
    spikes_e=pickle.load(pfile)

with open(save_dir+'/ginh.p', 'rb') as pfile:
    spikes_i=pickle.load(pfile)

RT = (spikes_e['t']/second)[-1]

CV_e = p.ISI_CV(spikes_e['i'][:],spikes_e['t']/second,'E',N_i_new,filename,save_dir)
CV_i = p.ISI_CV(spikes_i['i'][:],spikes_i['t']/second,'I',N_i_new,filename,save_dir)
p.spike_plot(spikes_e['i'][:], spikes_i['i'][:], spikes_e['t']/second, spikes_i['t']/second, N_i_new, RT-4, RT-1, CV_e, CV_i, filename, save_dir)

print("Zeit1:",time.time()-time1)
