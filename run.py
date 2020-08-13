import sys
import os
from brian2 import *
set_device('cpp_standalone',directory='./builds',build_on_run=False)
import numpy as np
import time
import br2models as mod
from utils import generate_connections, generate_full_connectivity
import datetime
from shutil import copyfile
import matplotlib.pyplot as mpl
from cpp_methods import syn_scale, syn_EI_scale
import pickle

##########################################################
# Functions
##########################################################

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

##########################################################
# Parameter Values 
##########################################################

# Time parameters 
    # different runs i=1,2,3 with time Ti to record different parameters
    # see recorded parameters under section "Run Simulation"
T1 = 100                  # time of first run in s
number_T1 = int(T1/100)     # number of partial runs to decrease use of RAM
T2 = 900
number_T2 = int(T2/100) 
T3 = 100
number_T3 = int(T3/100)

RT = T1+T2+T3               # total runtime

synEE_dt = 5000.*ms         # time step of EE recording of synaptic traces
synEI_dt = 5000.*ms         # time step of EI recording of synaptic traces

# Base parameters 
N_e = 800                   # number of excitatory neurons
N_i = int(0.2*N_e)          # number of inhibitory neurons
p_ee = 0.035                 # exc to exc sparseness
p_ie = 0.15                 # exc to inh sparseness
p_ei = 0.5                  # inh to exc sparseness
p_ii = 0.5                  # inh to inh sparseness
a_ee = 0.04                # initial exc to exc weight 
a_ie = 0.04                # exc to inh weight 
a_ei = 0.04                # initial inh to exc weight 
a_ii = 0.04                # inh to inh weight 
ascale = 1.0                # to scale to weights

# Neuron parameters 
syn_cond_mode = 'exp'       # Conductance mode of EE synapses (exp/alpha/biexp)
syn_cond_mode_EI = 'exp'    # Conductance mode of EI synapses (exp/alpha/biexp)

tau = 20.*ms                # membrane time constant
El = -60.*mV                # resting value
Ee = 0.*mV                  # reversal potential excitation
Ei = -80.*mV                # reversal potential inhibition
Vr_e = -60.*mV              # excitatory reset value
Vr_i = -60.*mV              # inhibitory reset value
tau_e = 5.*ms               # EPSP time constant
tau_i = 10.*ms              # IPSP time constant

tau_e_rise = 0.5*ms         # Used by biexp conductance model
tau_i_rise = 0.15*ms        # Used by biexp conductance model
norm_f_EE = 1.0             # Used by alpha, biexp conductance model
norm_f_EI = 1.0             # Used by alpha, biexp conductance model
invpeakEE = (tau_e / tau_e_rise) ** (tau_e_rise / (tau_e - tau_e_rise))
                            # Used by biexp conductance model
invpeakEI = (tau_i / tau_i_rise) ** (tau_i_rise / (tau_i - tau_i_rise))
                            # Used by biexp conductance model

# external noise parameters
external_mode = 'memnoise'  # poisson is not implemented yet
mu_e = 9.0*mV               # memnoise μ for excitatory neurons
mu_i = 8.5*mV               # memnoise μ for inhibitory neurons
sigma_e = 0.5**0.5*mV       # memnoise σ for excitatory neurons
sigma_i = 0.5**0.5*mV       # memnoise σ for inhibitory neurons

# Neuronal Thresholds 
Vt_e = -50.*mV              # initial Vt for excitatory neurons
Vt_i = -51.*mV              # initial Vt for inhibitory neurons

# STDP parameters (exc and inh) 
taupre = 15*ms              # pre-before-post STDP time constant
taupost = 30*ms             # post-before-pre STDP time constant
Aplus = 0.0015              # potentiating STDP learning rate
Aminus = -0.00075           # depressing STDP learning rate
amax = 2.0                  # maximum weight

stdp_active = 1             # enable recording of EE synapse spikes
ATotalMax = 1.1775
sig_ATotalMax = 0.05

istdp_active = 1
istdp_type = 'sym'          # (sym/dbexp)
taupre_EI = 20*ms
taupost_EI = 20*ms
LTD_a = 0.000005

# synaptic noise parameters 
syn_noise = 1               # enable/disable synapse noise
syn_noise_type = 'additive' # (additive/multiplicative)
syn_sigma = 1e-09/second    # synapse noise sigma
synEE_mod_dt = 100*ms       # dt between update of state variables


# escaling 
scl_active = 1              # EE synaptic scaling
dt_synEE_scaling = 25*ms    # time step for synaptic scaling
eta_scaling = 0.25

# iscaling 
iscl_active = 1
iATotalMax = 0.7/6
sig_iATotalMax = 0.025
syn_iscl_rec = 0

ext_to_E = 1.2 #5#7#5 # strength of external input (weight) to E neurons
ext_to_I = 1.2 # strength of external input (weight) to I neurons
inpRate = 5  # strength of external input (spike frequency)


##########################################################
# Build neuron model 
##########################################################

neuron_model = mod.condlif_memnoise

if syn_cond_mode=='exp':
    neuron_model += mod.syn_cond_EE_exp
elif syn_cond_mode=='alpha':
    neuron_model += mod.syn_cond_EE_alpha
elif syn_cond_mode=='biexp':
    neuron_model += mod.syn_cond_EE_biexp

if syn_cond_mode_EI=='exp':
    neuron_model += mod.syn_cond_EI_exp
elif syn_cond_mode_EI=='alpha':
    neuron_model += mod.syn_cond_EI_alpha
elif syn_cond_mode_EI=='biexp':
    neuron_model += mod.syn_cond_EI_biexp

GExc = NeuronGroup(N=N_e, model=neuron_model,
                   method='euler',
                   threshold='V > Vt',
                   reset='V = Vr_e')
GInh = NeuronGroup(N=N_i, model=neuron_model,
                   method='euler',
                   threshold ='V > Vt',
                   reset='V=Vr_i')

GExc.Vt, GInh.Vt = Vt_e, Vt_i
GExc.V , GInh.V  = np.random.uniform(Vr_e/mV, Vt_e/mV,
                                     size=N_e)*mV, \
                   np.random.uniform(Vr_i/mV, Vt_i/mV,
                                     size=N_i)*mV

##########################################################
# Build synapse model 
##########################################################

##### define synapse model of EI and EE synapses #########

# noise
if syn_noise:
    if syn_noise_type=='additive':
        synEE_mod = '''%s 
                       %s''' %(mod.synEE_noise_add, mod.synEE_mod)

        synEI_mod = '''%s 
                       %s''' %(mod.synEE_noise_add, mod.synEE_mod)

    elif syn_noise_type=='multiplicative':
        synEE_mod = '''%s 
                       %s''' %(mod.synEE_noise_mult, mod.synEE_mod)

        synEI_mod = '''%s 
                       %s''' %(mod.synEE_noise_mult, mod.synEE_mod)
else:
    synEE_mod = '''%s 
                   %s''' %(mod.synEE_static, mod.synEE_mod)
    
    synEI_mod = '''%s 
                   %s''' %(mod.synEE_static, mod.synEE_mod)

# add scaling
if scl_active:
    synEE_mod = '''%s
                   %s''' %(synEE_mod, mod.synEE_scl_mod)
    synEI_mod = '''%s
                   %s''' %(synEI_mod, mod.synEI_scl_mod)

# define on_pre rule for EE synapse
if syn_cond_mode=='exp':
    synEE_pre_mod = mod.synEE_pre_exp
elif syn_cond_mode=='alpha':
    synEE_pre_mod = mod.synEE_pre_alpha
elif syn_cond_mode=='biexp':
    synEE_pre_mod = mod.synEE_pre_biexp

# define on_post rule for EE synapse
synEE_post_mod = mod.syn_post
    
# add STDP to EE synapse
if stdp_active:
    synEE_pre_mod  = '''%s 
                        %s''' %(synEE_pre_mod, mod.syn_pre_STDP)
    synEE_post_mod = '''%s 
                        %s''' %(synEE_post_mod, mod.syn_post_STDP)

# define on_pre and on_post rule for EI synapse (iSTDP type dbexp)
if istdp_active and istdp_type=='dbexp':
    if syn_cond_mode_EI=='exp':
        EI_pre_mod = mod.synEI_pre_exp
    elif syn_cond_mode_EI=='alpha':
        EI_pre_mod = mod.synEI_pre_alpha
    elif syn_cond_mode_EI=='biexp':
        EI_pre_mod = mod.synEI_pre_biexp
        
    synEI_pre_mod  = '''%s 
                        %s''' %(EI_pre_mod, mod.syn_pre_STDP)
    synEI_post_mod = '''%s 
                        %s''' %(mod.syn_post, mod.syn_post_STDP)

# define on_pre and on_post rule for EI synapse (iSTDP type dbexp)
elif istdp_active and istdp_type=='sym':

    if syn_cond_mode_EI=='exp':
        EI_pre_mod = mod.synEI_pre_sym_exp
    elif syn_cond_mode_EI=='alpha':
        EI_pre_mod = mod.synEI_pre_sym_alpha
    elif syn_cond_mode_EI=='biexp':
        EI_pre_mod = mod.synEI_pre_sym_biexp

    synEI_pre_mod  = '''%s 
                        %s''' %(EI_pre_mod, mod.syn_pre_STDP)
    synEI_post_mod = '''%s 
                        %s''' %(mod.synEI_post_sym, mod.syn_post_STDP)
    
    
# define EE synapse
SynEE = Synapses(target=GExc, source=GExc, method='euler', model=synEE_mod,
                     on_pre=synEE_pre_mod, on_post=synEE_post_mod,
                     dt=synEE_mod_dt)

# define EI synapse 
if istdp_active:        
    SynEI = Synapses(target=GExc, source=GInh, method='euler', model=synEI_mod,
                     on_pre=synEI_pre_mod, on_post=synEI_post_mod,
                     dt=synEE_mod_dt)    
else:
    model = '''a : 1
               syn_active : 1'''
    SynEI = Synapses(target=GExc, source=GInh, model=model,
                     on_pre='gi_post += a')
    
 
##### define synapse model of IE and II synapses #########

SynIE = Synapses(target=GInh, source=GExc, method='euler', on_pre='ge_post += a_ie')
SynII = Synapses(target=GInh, source=GInh, method='euler', on_pre='gi_post += a_ii')


##### generate connections and connects the synapses #####
sEE_src, sEE_tar = generate_connections(N_e, N_e, p_ee, same=True)
SynEE.connect(i=sEE_src, j=sEE_tar)

sEI_src, sEI_tar = generate_connections(N_e, N_i, p_ei) 
SynEI.connect(i=sEI_src, j=sEI_tar)

sIE_src, sIE_tar = generate_connections(N_i, N_e, p_ie)
SynIE.connect(i=sIE_src, j=sIE_tar)

sII_src, sII_tar = generate_connections(N_i, N_i, p_ii,same=True)
SynII.connect(i=sII_src, j=sII_tar)

##########################################################
# Add Noise
##########################################################

if syn_noise:
    SynEE.run_regularly('a = clip(a,0,amax)', when='after_groups',dt=dt_synEE_scaling) 

if syn_noise and istdp_active:
    SynEI.run_regularly('a = clip(a,0,amax)', when='after_groups',dt=dt_synEE_scaling,) 
 
SynEE.syn_active, SynEE.a = 1, a_ee
SynEI.syn_active, SynEI.a = 1, a_ei

##########################################################
# Scaling 
##########################################################

if scl_active:

    if sig_ATotalMax==0.:
        GExc.ANormTar = ATotalMax
    else:
        GExc.ANormTar = np.random.normal(loc=ATotalMax,
                                         scale=sig_ATotalMax,
                                         size=N_e)
    
    SynEE.summed_updaters['AsumEE_post']._clock = Clock(dt=dt_synEE_scaling)
    SynEE.run_regularly(mod.synEE_scaling,
                                        dt=dt_synEE_scaling,
                                        when='end')

if istdp_active and iscl_active:
    if sig_iATotalMax==0.:
        GExc.iANormTar = iATotalMax
    else:
        GExc.iANormTar = np.random.normal(loc=iATotalMax,
                                           scale=sig_iATotalMax,
                                           size=N_e)
        
    SynEI.summed_updaters['AsumEI_post']._clock = Clock(
        dt=dt_synEE_scaling)

    SynEI.run_regularly(mod.synEI_scaling,
                                        dt=dt_synEE_scaling,
                                        when='end')
    
##########################################################
# External Input
##########################################################  

inputToEx = PoissonInput(target= GExc,target_var = 'ge', N=N_e, rate=inpRate*Hz,weight = ext_to_E)
inputToInh = PoissonInput(target = GInh, target_var = 'ge', N=N_i, rate=inpRate*Hz, weight = ext_to_I)
##########################################################
# Monitoring 
##########################################################    

# monitor of synaptic weights
M_ee = StateMonitor(SynEE, ['a'],
                              record=range(50),
                              when='end', dt=synEE_dt)

M_ei = StateMonitor(SynEI, ['a'],
                          record=range(50),
                          when='end', dt=synEI_dt)

# monitor of spikes
GExc_spks = SpikeMonitor(GExc[:50])    
GInh_spks = SpikeMonitor(GInh[:50])

##########################################################
# Run Simulation
##########################################################  

net = Network(collect())  

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

##########################################################
# Save spike times, synaptic connections, and EE and EI synaptic weights
##########################################################
if not os.path.exists('data'):
    os.makedirs('data')
    
save_dir = 'data'

# save spike times
with open(save_dir+'/ginh.p','wb') as pfile:
    pickle.dump(GInh_spks.get_states(['i','t']),pfile)

with open(save_dir+'/gexc.p','wb') as pfile:
    pickle.dump(GExc_spks.get_states(['i','t']),pfile)

# save synaptic weights
with open(save_dir+'/synee.p','wb') as pfile:
    pickle.dump(M_ee.get_states(['a']),pfile)

with open(save_dir+'/synei.p','wb') as pfile:
    pickle.dump(M_ei.get_states(['a']),pfile)

# save synaptic connections
with open(save_dir+'/sEE_src.p','wb') as pfile:
    pickle.dump(sEE_src,pfile)

with open(save_dir+'/sEI_src.p','wb') as pfile:
    pickle.dump(sEI_src,pfile)
    
with open(save_dir+'/sEE_tar.p','wb') as pfile:
    pickle.dump(sEE_tar,pfile)

with open(save_dir+'/sEI_tar.p','wb') as pfile:
    pickle.dump(sEI_tar,pfile)