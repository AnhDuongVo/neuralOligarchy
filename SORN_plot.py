import numpy as np
#from brian2 import *

class SORN_plot():
    ''' This class defines plotting for SORN results.
    Many plots share similar code and features, that is why I put them together in this class.
    Also useful to keep plotting consistent
    '''
    def __init__(self, fs):
        self.fs = fs
        
    @classmethod # this method is accessed by other methods within SORN_plot.
    def BinSpikesFast(self,spiketrain,binsize,run_time):
        #loopless version for binning spikes. Requires numpy
        binlabels = np.arange(0,(run_time/second)+binsize,binsize)
        binvalues = np.zeros([len(binlabels)],float)
    
        mat = np.tile(binlabels, (len(spiketrain),1))    
        mat_err = np.absolute(mat - np.transpose(np.tile(spiketrain,(len(binlabels),1))))
    
        # find index of bin with smallest error. If two neighouring bins have equal error,
        # then the numpy.argmin() function always chooses the lowest bin.
        bin_ind = np.argmin(mat_err,axis=1)
    
        # some spikes end up in the same bin.
        # to avoid loop argument, 
        # use histogram to find total spikes that go in each bin.
        # the histogram classifies just integers (the indices)
        hst = np.histogram(bin_ind, bins = np.arange(len(binlabels))-0.5)
    
        return hst[0] # how many spikes in each bin    
        
    @classmethod # this method is accessed by other methods within SORN_plot.
    def BinSpikesFastLimit(self,spiketrain,binsize,run_time,lim):
        #loopless version for binning spikes. Requires numpy
        
        # take the last part of the spikes only: with spike times 
        # starting with 'lim' (seconds)
        
        spikes = spiketrain[np.where(spiketrain>lim)]
        spikes -= lim #subtract the unused time (before lim)
        
        #binlabels = np.arange(lim,(run_time/second)+binsize,binsize)
        binlabels = np.arange(0,(run_time/second)-lim+binsize,binsize)
        #binvalues = np.zeros([len(binlabels)],float)
    
        mat = np.tile(binlabels, (len(spikes),1))    
        mat_err = np.absolute(mat - np.transpose(np.tile(spikes,(len(binlabels),1))))
    
        # find index of bin with smallest error. If two neighouring bins have equal error,
        # then the numpy.argmin() function always chooses the lowest bin.
        bin_ind = np.argmin(mat_err,axis=1)
    
        # some spikes end up in the same bin.
        # to avoid loop argument, 
        # use histogram to find total spikes that go in each bin.
        # the histogram classifies just integers (the indices)
        hst = np.histogram(bin_ind, bins = np.arange(len(binlabels)+1)-0.5) # 0.5 is for making int bincenters binedges
    
        #allbins = hst[0]    
        return hst[0]#allbins#allbins[np.floor(lim/binsize):len(allbins)] # how many spikes in each bin    
        
    def plotWeightsFinal(self,ws,ntype,filename,save_dir):
        # plot the distribution of weights at the end of the simulation.
        
        # since none of the weights are zero at the first timepoint, this is
        # used to find the presence of an actual connection in the SORN instance.
        
        import matplotlib.pyplot as mpl
        
        N_e = 400
        N_i = 80
        
        fs = 24
        ll = 12
        ww = 5
        
        if ntype == 'E':
            N = N_e**2
            cc = 'g'
        else:
            N = N_e*N_i
            cc = [0.56,0.15,0.56]#'r'
            
        figW = mpl.figure(figsize=(ll,ww))
        mpl.rcParams['xtick.major.pad']='15'
        mpl.rcParams['ytick.major.pad']='15'
    
        start = ws[0].reshape(N) 
        finish = ws[-1].reshape(N)
        
        #print np.max(finish)
        
        finish_inc0 = finish#[np.where(start>0)]
        finish_non0 = finish_inc0[np.where(finish_inc0>0)]
    
        if ntype == 'E':
            
            binedges = np.arange(0,0.00201,0.00001)
        else:
            binedges = np.arange(0,0.031,0.0005)
        
        #mpl.hist(finish_inc0,bins=binedges,facecolor='white',edgecolor = cc,linewidth=2)
        mpl.hist(finish_non0,bins=binedges,facecolor=cc,linewidth=2)
        
        #mpl.xticks([0.000,0.002,0.004,0.006])#,0.008, 0.01])
        #mpl.xticks([0,0.01,0.02,0.03,0.04,0.05])
        if ntype == 'E':
            #mpl.xticks([0,0.002,0.004,0.006],[0,2.0,4.0,6.0],fontsize=fs)
            mpl.xlim(0,0.002)
        else:
            mpl.xticks([0,0.01,0.02,0.03],[0,10,20,30],fontsize=fs)
            mpl.xlim(0,0.03)
            
        mpl.yticks(fontsize=fs)
        mpl.tight_layout()
        mpl.title(str(np.max(finish)))
        #mpl.show()
        figW.savefig(save_dir + '/LIFSORN_wfinal' + filename + '_' + ntype + ".eps",format="eps") 
   
    def plotWeightsEvol(self,ws,ntype,nn,filename,save_dir):
        
        import matplotlib.pyplot as mpl
        import matplotlib.colors as colors
        import matplotlib.cm as cm
        
        # plotting parameters
        fs = 34
        ll = 12
        ww = 6
                
        fig1 = mpl.figure(figsize=(ll,ww))
        mpl.rcParams['xtick.major.pad']='18'
        mpl.rcParams['ytick.major.pad']='18'
        ax = fig1.add_subplot(1,1,1)
    
        mpl.xticks(fontsize=fs)
        mpl.yticks(fontsize=fs)
        weightspertimei = np.zeros([len(ws),nn])
        meanweight = np.zeros([len(ws)])
    
        b = ws[0,:,:]
        b_end = ws[-1,:,:]
        
        # get indices (i, j) in the connectivity matrix, where an actual connection is present.
        true_iw = np.where(b>0) #get connections that exist at the beginning of simulation, with nonzero weights.
        true_iw_i = true_iw[0] # the i index in the matrix (rows). inhibitory or excitatory neurons.
        true_iw_j = true_iw[1] # the j index (columns). excitatory neurons.
            
        rand_select = np.sort(np.random.randint(0,len(true_iw_i)-1,nn)) # randomly select nn 'existing' weights
        ipos = true_iw_i[rand_select] # get the i indices in the connec. matrix.
        jpos = true_iw_j[rand_select] # get the j indices.

        for t in range(0,len(ws)):

            b = ws[t,:,:]
            weightspertimei[t,:] = b[ipos,jpos] # get time evolution of the selected weights.
            meanweight[t] = np.mean(b[true_iw]) # mean I weight over all connections.
            # for structural plasticity:   
            #if t>0: # skip first step in which there are no connections.          
            #  meanweight[t] = np.mean(b[np.where(b>0)]) # mean E weight over all nonzero connections.
                
        
        # get different colours for each individual plotted weight.
        # colours are sampled evenly spaced from colormap.
        myMap = mpl.get_cmap('jet')
        cNorm  = colors.Normalize(vmin=0, vmax=nn)
        scalarMap = cm.ScalarMappable(norm=cNorm, cmap=myMap)
        
        for weight in range(nn):
            colorVal = scalarMap.to_rgba(weight)
            mpl.plot(weightspertimei[:,weight],linewidth=2,color=colorVal)
            
        mpl.plot(meanweight,linewidth=5,color='k')
        
        #if ntype == 'E':
        #    mpl.yticks([0,0.01,0.02],['0','.01','.02'],fontsize=fs)
        #else:
        #    mpl.yticks([0,0.005,0.01],['0','.005','.01'],fontsize=fs)

        # remove unnecessary axes & ticks
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left') 
        ax.xaxis.set_ticks_position('bottom')
        mpl.tight_layout()                
        fig1.savefig(save_dir + '/Wevol' +  filename + '_' + ntype + ".eps",format="eps")

    def plotWeightsEvolSTD(self,ws,ntype,filename,save_dir):
        
        # get evolution of all weights, but only plot the mean over all weights, and the STD over all weights.
        # if you include structural plasticity: ! important! edit so that you also include 0 weigths
        
        import matplotlib.pyplot as mpl
        import matplotlib.colors as colors
        import matplotlib.cm as cm
        
        fs = 34
        ll = 12
        ww = 6
        
        if ntype == 'E':
            c = [0,0.7,0]
        else:
            c = [0.56,0.15,0.56]
            
        fig1 = mpl.figure(figsize=(ll,ww))
        ax = fig1.add_subplot(1,1,1)
        mpl.rcParams['xtick.major.pad']='18'
        mpl.rcParams['ytick.major.pad']='18'
        mpl.xticks(fontsize=fs)
        mpl.yticks(fontsize=fs)
        
        b_start = ws[0]
        b_end = ws[-1]
        
        # get indices (i, j) in the connectivity matrix, where an actual connection is made (all existant synapses are initiated above 0.):
       
        true_iw = np.where(b_start>0) # the weights that represent existent connections.
               
        a = np.shape(true_iw)
        true_iw_i = true_iw[0] # the i index in the matrix (rows). inhibitory or excitatory neurons.
        true_iw_j = true_iw[1] # the j index (columns). excitatory neurons.
    
        #rand_select = np.sort(np.random.randint(0,a[1]-1,nn)) # randomly select weights
        #rand_select = np.sort(np.random.randint(0,a[1]-1,nn)) # select ALL weights
    
        ipos = true_iw_i#[rand_select] # get the i indices in the connec. matrix.
        jpos = true_iw_j#[rand_select] # get the j indices.
        
        weightspertimei = np.zeros([len(ws),len(true_iw_i)])
        meanweight = np.zeros([len(ws)])
        sdweight = np.zeros([len(ws)])

        for t in range(0,len(ws)):
            #a = W_eTOe[t]
            #weightspertime[t,:] = a[0:nr_neurons,0]
            b = ws[t]
            weightspertimei[t,:] = b[ipos,jpos] # get time evolution of the selected weights.
            
            if ntype == 'E':
                meanweight[t] = np.mean(b) 
                sdweight[t] = np.std(b)
            else:
                meanweight[t] = np.mean(b[true_iw])   
                sdweight[t] = np.std(b[true_iw])
               
        mpl.plot(meanweight+sdweight,'--',linewidth=2,color=c)
        mpl.plot(meanweight-sdweight,'--',linewidth=2,color=c)
  
        mpl.plot(meanweight,linewidth=4,color=c)

        
        if ntype == 'E':
            mpl.yticks(fontsize=fs)
            mpl.yticks([0,0.01,0.02],['0','.01','.02'],fontsize=fs)
        else:
            mpl.yticks([0,0.005,0.01],['0','.005','.01'],fontsize=fs)
        
        # remove unnecessary axes & ticks
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left') 
        ax.xaxis.set_ticks_position('bottom')
        mpl.tight_layout()    
        #mpl.show()
        fig1.savefig(save_dir + '/LIFSORN_wevol_STD' +  filename + '_' + ntype + ".eps",format="eps")
    
    def plotWeightsScatter(self,ws,Nin,Nout,wtype,start,endpt,timing,filename,save_dir):
        # W - delta W plots with linear regression.
        
        import matplotlib.pyplot as mpl
        from scipy import stats
        fs = 24
        
        figdw = mpl.figure(figsize = (6,6))
        mpl.rcParams['xtick.major.pad']='15'
        mpl.rcParams['ytick.major.pad']='15'
        
        if wtype == 'E':
            c=[0,0.7,0]
        else:
            c=[0.65,0.15,0.65]
        
        xx = []
        yy = []
        for x in range(Nin):
            for y in range(Nout):
                # check each weight
                m = ws[0]
                if m[x,y]>0: 
                    # which are the existing weights?
                    w = np.zeros([len(ws)],float)
                    for i in range(len(ws)):
                        m  = ws[i]
                        w[i] = m[x,y]

                    xx.append(w[start])
                    yy.append(w[endpt])  
        
        mpl.scatter(xx,yy,marker='+',color=c)            
        slope, intercept, r_value, p_value, std_err = stats.linregress(xx,yy)
        mpl.title(r_value**2)

        # regression line
        mpl.plot([0,0.02],[intercept,intercept+(0.02*slope)],linewidth = 1, color='k')
                
        if wtype == 'E':
            mpl.xlim(0,0.05) # weight.
            mpl.ylim(0,0.05)
            #mpl.xticks([0,0.001,0.002,0.003,0.004],fontsize=fs)
                    
            # unit line
            mpl.plot([0,0.004],[0,0.004],color='y')
            
        elif wtype =='I':
            mpl.xlim(0,0.05) # weight.
            mpl.ylim(0,0.05)           
            mpl.xticks([0,0.01,0.02,0.03,0.04,0.05],['0','.01','.02','.03','.04','.05'],fontsize=fs)
            mpl.yticks([0,0.01,0.02,0.03,0.04,0.05],['0','.01','.02','.03','.04','.05'],fontsize=fs)           
            #mpl.yticks([-0.01,0,0.01,0.02,0.03],[-10,0,10,20,30],fontsize=fs)  
            
            # unit line
            mpl.plot([0,0.05],[0,0.05],color='y')      
        
        #mpl.show()
        figdw.savefig(save_dir + '/DeltaW_' + wtype + '_len' + str(float(endpt+1)) + filename + '_' + timing + '.eps',format = "eps")     
        #save R^2 value.
        #np.save(save_dir + '/R2_' + wtype + '_len' + str(float(endpt+1)) + filename + '.npy',r_value**2)     
  
    def plotSTDPcontrib(self,nn,ws2,ws3,wtype,N_e,run_time,start,filename,save_dir):
        
        import matplotlib.pyplot as mpl
        figwdw = mpl.figure(figsize = (6,6))
        import numpy.random as nr
              
        totaltime = int(run_time/second)
        #w_t0 = np.array([])
        #w_t1 = np.array([])
        
        # weight changes on a timeline, per weight
        tlen = totaltime-start
        wt = np.zeros([tlen,nn],float)
        
        # get indices (i, j) in the connectivity matrix, where an actual connection is made from the start. (for struct. plast.: 
        # connections that exist at the end of simulation, with nonzero weights.
        if wtype == 'E':
           true_iw = np.where(ws2[-1]>0) #get connections that exist at the end of simulation, with nonzero weights.
        else:
           true_iw = np.where(ws2[0]>0) #get existing connections (weights are always nonzero at the beginning of the simulation).     
        a = np.shape(true_iw)
        true_iw_i = true_iw[0] # the i index in the matrix (rows). inhibitory or excitatory neurons.
        true_iw_j = true_iw[1] # the j index (columns). excitatory neurons.

        rand_select = np.sort(np.random.randint(0,a[1],nn)) # randomly select nn 'existing' weights
        ipos = true_iw_i[rand_select] # get the i indices in the connec. matrix.
        jpos = true_iw_j[rand_select] # get the j indices.
        
        for t in range(start,totaltime):
            ws = ws2[t]
            ws_sc = ws3[t-1]
            
            # matrix of all weight changes through STDP in this step
            STDPcontrib  = ws[ipos,jpos] - ws_sc[ipos,jpos]
            wt[t-start,:] = STDPcontrib
            
        for w in range(nn):
            mpl.plot(wt[:,w],linewidth=2)  
                    
        mpl.show()
        figwdw.savefig(save_dir + '/W_STDPcontrib_' + filename + '_' + wtype +'.eps',format = "eps")        
        
    def plotRegSlope(self,ws,Nin,Nout,start,filename,save_dir):
        
        import matplotlib.pyplot as mpl
        from scipy import stats
        
        fs = 24
        
        end = np.size([ws])/(Nin*Nout)
        #get regression slopes
        sl=np.zeros([end-start],float) 
        
        #WeTOe 
        w = ws[0] #starting weights
               
        for timestep in range(end-start):
            
            wstart = ws[start]
            wsend = ws[start+timestep]
                        
            w_exist_start = wstart[np.where(w>0)]
            w_exist_end = wsend[np.where(w>0)]
            
            w_exist_start.reshape(np.size(w_exist_start))
            w_exist_end.reshape(np.size(w_exist_start))
                        
            slope, intercept, r_value, p_value, std_err = stats.linregress(w_exist_start,w_exist_end) 
            sl[timestep]=slope
        
        
        figslope = mpl.figure(figsize=(6,5))
        mpl.rcParams['xtick.major.pad']='15'
        mpl.rcParams['ytick.major.pad']='15'
        
        mpl.plot(np.arange(end-start),sl,linewidth=2)
        #mpl.plot(np.mean(sl,axis=0)-np.std(sl,axis=0),'--')
        #mpl.plot(np.mean(sl,axis=0)+np.std(sl,axis=0),'--')
        mpl.ylim(0,1.0)
        
        mpl.xticks(fontsize=fs)
        mpl.yticks(fontsize=fs)
        figslope.savefig(save_dir + '/DeltaW_' + filename + '_slope.eps',format = "eps")
               
    def plotPopRate(self,MR_e,MR_i,filename, save_dir):    
        
        import matplotlib.pyplot as mpl
        fs = 24
        
        fig1 = mpl.figure(figsize=(5,6))
        mpl.rcParams['xtick.major.pad']='15'
        mpl.rcParams['ytick.major.pad']='15'
        mpl.plot(MR_e.times,MR_e.smooth_rate(1000 * ms, 'gaussian'), color = [0,0.7,0],linewidth=3)
        mpl.plot(MR_i.times, MR_i.smooth_rate(1000 * ms, 'gaussian'), color = [0.56,0.15,0.56],linewidth=3)
        mpl.xlabel('Time(s)',fontsize=fs)
        mpl.ylabel('Mean Rate (Hz)',fontsize=fs)
        mpl.xticks(fontsize=fs)
        mpl.yticks(fontsize=fs)
        #mpl.title('Excitatory Population Rate in Green, Inhibitory Population Rate in Red')
        
        #mpl.xlim(5.0,45.0)
        
        #print np.mean(MR_e.smooth_rate(1000 * ms, 'gaussian'))
        #print 'i:'
        #print np.mean(MR_i.smooth_rate(1000 * ms, 'gaussian'))
        #mpl.show()
        fig1 = savefig(save_dir + '/LIFSORN_rate' + filename+ ".png",format="png")

    def spike_plot(self, S_e, S_i, S_et, S_it, N_i, start,stop, CV_e, CV_i, filename, save_dir):

        import matplotlib.pyplot as mpl
        fs = 12 

        nshow = 50 # show how many neurons of each type?
        
        if nshow>N_i:
            print("error: nshow is larger than number of inhibitory neurons!")
        
        # sort the spike times and the spike indices.
        spikeFig = mpl.figure(figsize=(8,8))
        font = {'family' : 'Arial',
            'weight' : 'normal',
            'size'   : fs}
        mpl.rc('font', **font)
        mpl.rcParams['xtick.major.pad']='18'
        mpl.rcParams['ytick.major.pad']='18'
        mpl.plot(S_et,S_e+N_i,'o',color=[0,110/255.0,65/255.0],markersize=5,markeredgecolor=[0,110/255.0,65/255.0], markeredgewidth=0.0, label = "excitatory")
        mpl.plot(S_it,S_i,'o',color=[0.65,0.15,0.65],markersize=5,markeredgecolor=[0.65,0.15,0.65], markeredgewidth=0.0, label="inhibitory")
        print(S_it)
        print(S_i)        
        mpl.xlim(start,stop)
        mpl.ylim(N_i-nshow,N_i+nshow)
        mpl.xlabel("time[s]")
        mpl.ylabel("Neuron number")
        mpl.title('CV E: ' + str(round(CV_e,2)) + '  CV I: ' + str(round(CV_i,2)))
        mpl.tight_layout()
        legend = mpl.legend(loc= "upper right")
        legend.get_frame().set_facecolor('white')
        mpl.rcParams["legend.framealpha"] = 0.0
        #mpl.show()
        spikeFig.savefig(save_dir + "/SpikePlot" + filename + ".pdf",format = "pdf")
        
    def membranePlot(self,MS_e,G_e,MV_e,MV_e_ge,MV_e_gi,Ee,Ei,tau,run_time,filename,save_dir):
        
        # show membrane voltage, and E and I currents onto a neuron.
        # the last x seconds of the simulation are shown.
        
        import matplotlib.pyplot as mpl
        import scipy.signal as sp
        
        fs = 18 # fontsize for plotting
        nrn = 0 # chosen neuron:  [0~N_e]
        x = 2 # the last x seconds of the simulation are shown.

        MV_epsp = np.multiply((np.tile(Ee,[len(MV_e.times)])-MV_e[nrn]),MV_e_ge[nrn])/tau
        MV_ipsp = np.multiply((np.tile(Ei,[len(MV_e.times)])-MV_e[nrn]),MV_e_gi[nrn])/tau

        figPOT = mpl.figure(figsize=(9,6))
        mpl.rcParams['xtick.major.pad']='15'
        mpl.rcParams['ytick.major.pad']='15'
    
        tt = int(((run_time/second)-x)*1000)

        mpl.subplot(311)

        mpl.title('Membrane Voltage',fontsize=fs)
        thresh = G_e[nrn].Vt
        plot([(run_time/second)-x,(run_time/second)],[thresh,thresh],color='b',linewidth=2)
        mem_times = MV_e.times[tt::]
        mem_val = MV_e[nrn][tt::]
        plot(mem_times, mem_val,color='k',linewidth=1)
        spiketimes = MS_e[nrn]
        lastspikes = spiketimes[np.where(spiketimes > tt/1000)]
        for bb in range(len(lastspikes)):
            plot([lastspikes[bb],lastspikes[bb]],[-50*mV,-45*mV],color='k') # plot spikes
        mpl.xticks(fontsize=fs)
        mpl.yticks([-0.08,-0.07,-0.06,-0.05],[-80,-70,-60,-50],fontsize=fs)
        mpl.xlim(int(((run_time/second)-x)),int(((run_time/second))))
        mpl.tight_layout()

        mpl.subplot(312)

        mpl.title('Excitatory and Inhibitory conductances',fontsize=fs)
        #plot(MV_e_ge.times[tt::],MV_e_ge[nrn][tt::],color =[0,110/255.0,65/255.0],linewidth=2)
        #plot(MV_e_gi.times[tt::],-MV_e_gi[nrn][tt::],color = [0.56,0.15,0.56],linewidth=2)
        mpl.xticks(fontsize=fs)
        mpl.yticks([0.01,0.00,-0.01,-0.02],fontsize=fs)
        mpl.tight_layout()

        mpl.subplot(313)

        MV_epsp = np.multiply((np.tile(Ee,[len(MV_e.times)])-MV_e[nrn]),MV_e_ge[nrn])
        MV_ipsp = np.multiply((np.tile(Ei,[len(MV_e.times)])-MV_e[nrn]),MV_e_gi[nrn])
        #current * resistance (R=1/conductance)
        cur_resist_e = MV_epsp[tt::]
        cur_resist_i = MV_ipsp[tt::]
        #conductance
        g_e = MV_e_ge[nrn][tt::]
        g_i = MV_e_gi[nrn][tt::]
        # current
        cur_e = cur_resist_e * g_e #positive value
        cur_i = cur_resist_i * g_i #negative value
    
        #mpl.title('Excitatory and Inhibitory current * resistance')
        #plot(MV_e_ge.times[tt::],MV_epsp[tt::],color =[0.4,1,0.4],linewidth=2)
        #plot(MV_e_gi.times[tt::],MV_ipsp[tt::],color = [1,0.5,0.5],linewidth=2)
        mpl.title('Excitatory and Inhibitory currents',fontsize=fs)
        plot(MV_e_ge.times[tt::],cur_e,color =[0.4,1,0.4],linewidth=2)
        plot(MV_e_gi.times[tt::],cur_i,color = [1,0.5,1],linewidth=2)
        mpl.xticks(fontsize=fs)
        #mpl.yticks(fontsize=fs)
        #mpl.yticks([0.0000025,0.00000000,-0.0000025,-0.000005, -0.0000075],[25,0,-25,-50,-75],fontsize=fs)
        mpl.yticks([0.000005,0.00000000,-0.000005],[50,0,-50],fontsize=fs)
        mpl.xlabel('Time (second)',fontsize=fs)
        mpl.tight_layout()

        #mpl.show()    
        figPOT.savefig(save_dir + '/Fig_potential' + filename+ ".eps",format="eps")
        mpl.close()
        
        ''''
        # get smooth currents:
        winlen = 20 # steps
        e_cur = cur_e
        i_cur = cur_i
        #for l in range(len(cur_e)/winlen):
        area_e = np.zeros([len(e_cur)-winlen],float)
        area_i = np.zeros([len(i_cur)-winlen],float)
        for tt in range(len(e_cur)-winlen):
            win_e = e_cur[tt:(tt+winlen)]
            area_e[tt] = np.sum(win_e)

            win_i = i_cur[tt:(tt+winlen)]
            area_i[tt] = np.sum(win_i) 
             
        # smooth current trace and replot.
        figCUR = mpl.figure()
        mpl.title('Excitatory and Inhibitory currents',fontsize=fs)
        #plot(MV_e_ge.times[tt::],sp.savgol_filter(cur_e,91,5)/np.sum(cur_e),color =[0.4,1,0.4],linewidth=2)
        #plot(MV_e_gi.times[tt::],sp.savgol_filter(cur_i,91,5)/(-np.sum(cur_i)),color = [1,0.5,1],linewidth=2)
        plot(np.arange(len(area_e)),area_e/np.sum(area_e),color =[0.4,1,0.4],linewidth=2)
        plot(np.arange(len(area_i)),area_i/-np.sum(area_i),color = [1,0.5,1],linewidth=2)
        mpl.xticks(fontsize=fs)
        #mpl.yticks([0.0000025,0.00000000,-0.0000025,-0.000005],fontsize=fs)
        #mpl.yticks([0.0000025,0.00000000,-0.0000025,-0.000005, -0.0000075],[25,0,-25,-50,-75],fontsize=fs)
        
        mpl.xlabel('Time (second)',fontsize=fs)
        mpl.tight_layout()

        #mpl.show()    
        figCUR.savefig(save_dir + '/LIFSORN_CUR' + filename+ ".eps",format="eps")
        #mpl.close()
        
        #ax.yaxis.tick_left()
        #ax.xaxis.tick_bottom()
        '''
    def ISI_CV(self,spikes,spiketimes,ntype,nn,filename,save_dir):

        # calculate ISIs, and CV of ISIs for nn random neurons.
        #% inter-spike intervals
        
        import matplotlib.pyplot as mpl
        fs = 50
        
        #limit = 800 # seconds. count ISIs from here!
        #spikes  = spikes[np.where(spikes>limit)]-limit
        
        resol = 0.05    

        if ntype == 'E':
            #a = np.random.randint(400,size = nn)  # select nn random neurons
            colour=[0,110/255.0,65/255.0]
            ytix = [0,20,40,60,80]
        else:
            #a = np.random.randint(80,size = nn)  # select nn random neurons  
            colour=[0.56,0.15,0.56]
            ytix = [0,6,12,18,24]

        CV = np.zeros([nn], float)

        all_isi = []
                
        for i in range(nn):
            
            # get all spikes from this neuron
            spikes_neuron = spiketimes[spikes==i]
                        
            #spikes_neuron = spikes[a[i]]
            pernrn_isi = []
            for l in range(len(spikes_neuron)-1):
                dt = spikes_neuron[l+1]-spikes_neuron[l]
                pernrn_isi.append(dt)
                all_isi.append(dt)
            CV[i] = np.std(pernrn_isi)/np.mean(pernrn_isi)
            
        CV = CV[np.where(CV>0)] # remove zero values    
        print('mean ISI is:')
        print(np.mean(all_isi))    
        print('mean CV of nrn type ' + ntype + ' is:')
        print(np.mean(CV))
        
        figISI = mpl.figure(figsize=(7,6))
        mpl.rcParams['xtick.major.pad']='20'
        mpl.rcParams['ytick.major.pad']='20'
        ax = mpl.subplot(1,1,1)

        binlims = np.arange(0+resol,2.0+resol,resol)
        ax.hist(CV,bins=binlims,linewidth=2,facecolor=colour)
        #ax.hist(CV,bins=binlims,ec = colour,fc = 'None',lw=3,normed=True, histtype='step') 
        
        mpl.yticks(ytix,fontsize = fs)
        mpl.xticks([0.2,0.6,1.0,1.4],fontsize = fs)
        #mpl.xlim(0.2,1.4)
        
        # remove unnecessary axes & ticks
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left') 
        ax.xaxis.set_ticks_position('bottom')
        mpl.tight_layout()
        
        #figISI.savefig(save_dir + '/Fig_ISI' + filename + '_' + ntype + ".eps",format="eps")
        return np.mean(CV)

    def corr_mat(self,spikes_nrn,spikes_t, ntype,nn,run_time,lim,filename,save_dir):
        # display a matrix and a histogram with the pairwise correlations between E or between I neurons.
        # spikes: MS_e or MS_i. (BRIAN object from SpikeMonitor)
        # ntype (neurontype) is 'E' or 'I'. To save under the right filename. 
        # nn is the number of neurons shown.
        
        #for profiling /optimizing code:
        
        #import time
        # the code to time goes here
        #print(t1 - t0)
        
        fs = 34
        
        import matplotlib.pyplot as mpl
        import matplotlib.cm as cm # colormaps
        # calculate pairwise correlation coefficients.   
        
        #limit = 800 # seconds. count correlations from here!
        #spikes  = spikes[np.where(spikes>limit)]-limit # normalize the spike times to start at 0.
        r_time = run_time #- limit*second
        
        binsiz=0.005#0.05 # binsize in seconds
        if ntype == 'E':
            #a = np.random.randint(400,size = nn)  # select nn random neurons
            ccc=[0,0.7,0]
        else:
            #a = np.random.randint(80,size = nn)  # select nn random neurons    
            ccc=[0.56,0.15,0.56]
            
        #corr = np.zeros([nn,nn],float) # pairwise correlation matrix.
        corrvec = np.zeros([int((0.5*(nn-1)*nn))],float)
        count=0
           
        for i in range(nn):
            #print i
            
            spikes = spikes_t[np.where(spikes_nrn==i)]
            t0 = time.time()  # start time
            sp_nrn1 = SORN_plot_struct.BinSpikesFastLimit(spikes[i],binsiz,r_time,lim)
            #print 'just binned spikes'
            #print time.time()
            
            for k in range((i+1),nn):
        
                ### cov = np.dot(sp_nrn1-np.mean(sp_nrn1),sp_nrn2-np.mean(sp_nrn2)) # calc covariance
                ### corr[i,k] = cov / ((np.std(sp_nrn1))*(np.std(sp_nrn2))) # divide by variances or STD
                ### covr = np.cov(sp_nrn1,sp_nrn2) # calc covariance
                ### corr[i,k] = covr / ((np.std(sp_nrn1)**2)*(np.std(sp_nrn2)**2)) # divide by variances
                
                spikes = spikes_t[np.where(spikes_nrn==k)]
                sp_nrn2 = SORN_plot_struct.BinSpikesFastLimit(spikes[k],binsiz,r_time,lim)
                #corr[i,k] = np.corrcoef(sp_nrn1,sp_nrn2)[0,1]
                corrvec[count] = np.corrcoef(sp_nrn1,sp_nrn2)[0,1]
                count +=1
        
            t1 = time.time()
        
            #print t1-t0
        
        np.save(save_dir + '/corrs_' + filename + '_' + ntype + ".npy",corrvec)
            
        # histogram of the pairwise correlations.
        figCorrHist = mpl.figure(figsize=(7,6))
        mpl.rcParams['xtick.major.pad']='15'
        mpl.rcParams['ytick.major.pad']='15'
        
        #binlims = np.arange(0,0.105,0.005)
        binlims = np.arange(-0.1,0.1+binsiz,binsiz)
        
        
        #aa = np.histogram(corrvec, bins=binlims) 
        #bincenters = binlims[0:-1]+((binlims[1]-binlims[0])*0.5)
        #mpl.plot(bincenters,aa[0],linewidth=3,color='b')
        
        mpl.hist(corrvec, bins=binlims,facecolor=ccc,linewidth=2) 
        
        #xlabel('Pairwise Spike Correlations') 
        #ylabel('Frequency')
        
        mpl.xlim(-0.1,0.1)
        mpl.yticks(fontsize = fs)
        mpl.xticks([-0.1,0,0.1],fontsize = fs)
        mpl.tight_layout()
        #mpl.show()
        figCorrHist.savefig(save_dir + '/LIFSORN_corrhist' + filename + '_' + ntype + ".eps",format="eps")
        
    def corr_mat_EI(self,spikes_nrn1,spikes_t1,spikes_nrn2,spikes_t2, n1,n2,run_time,lim,filename,save_dir):
        # display a matrix and a histogram with the pairwise correlations between E and I neurons.
        # spikes1,2: MS_e and MS_i. (BRIAN object from SpikeMonitor)
        # n1,n2 is the number of neurons shown.
        
        fs = 34

        r_time = run_time #- limit*second
                
        import matplotlib.pyplot as mpl
        import matplotlib.cm as cm # colormaps
        # calculate pairwise correlation coefficients.   
        binsiz=0.005#0.05
        a1 = np.random.randint(400,size = n1)  # select n1 random neurons
        a2 = np.random.randint(80,size = n2)  # select n2 random neurons    
            
        corr = np.zeros([n1,n2],float) # pairwise correlation matrix.
        
        #corrvec = np.zeros([int((0.5*(nn-1)*nn))],float)
        corrvec = np.zeros([n1*n2],float)
        
        count=0
        for i in range(n1):
            #print 'now at:'
            #print i
            sp_nrn1 = SORN_plot_struct.BinSpikesFastLimit(spikes1[a1[i]],binsiz,r_time,lim)
            for k in range(n2):
        
                sp_nrn2 = SORN_plot_struct.BinSpikesFastLimit(spikes2[a2[k]],binsiz,r_time,lim)
                corr[i,k] = np.corrcoef(sp_nrn1,sp_nrn2)[0,1]
                corrvec[count] = corr[i,k] #np.corrcoef(sp_nrn1,sp_nrn2)[0,1]
                count +=1
 
        np.save(save_dir + '/corr_EI_' + filename + ".npy",corrvec)
            
        # histogram of the pairwise correlations.
        figCorrHist = mpl.figure(figsize=(7,6))
        mpl.rcParams['xtick.major.pad']='15'
        mpl.rcParams['ytick.major.pad']='15'
        
        binlims = np.arange(-0.1,0.1+binsiz,binsiz)
        
        mpl.hist(corrvec, bins=binlims,facecolor=[0.2,0.2,1],linewidth=2) 

        mpl.yticks(fontsize = fs)
        mpl.xticks([-0.1,0,0.1],fontsize = fs)
        mpl.xlim(-0.1,0.1)
        mpl.tight_layout()
        #mpl.show()
        figCorrHist.savefig(save_dir + '/LIFSORN_corrhist_' + filename + "EI.eps",format="eps")    
 
    def plotWdeltaW(self,startt,totaltime,Scfac,ws_rc2,ws_rc3,wtype,filename,save_dir):  
        
        import matplotlib.pyplot as mpl
        figW = mpl.figure(figsize=(5,10))
        mpl.rcParams['xtick.major.pad']='15'
        mpl.rcParams['ytick.major.pad']='15'
        
        N_e=400
        
        if wtype=='I':
            wmax = 0.015
            ymin = -0.0008
            ymax = 0.0008
            ymin_SN = -0.0008
            ymax_SN = 0.0008
        else:
            wmax = 0.05
            ymin = -0.00025
            ymax = 0.0004
            ymin_SN = -0.0001
            ymax_SN = 0.0001
        
        totalt = totaltime
        
        wdw_x = []
        wdw_stdp = []
        wdw_scs = []
        
        for t in range(startt,totalt):
            ws = ws_rc2[t]
            ws_sc = ws_rc3[t-1]

            Fs = Scfac[t,:]
            for i in range(N_e):
                #get x value. if not zero
                xvals = ws[:,i]
                yfcs = np.ones([len(xvals)])*Fs[i]
                
                #by how much is the weight decreased? (delta-W)
                yvals =  (xvals * yfcs) -xvals
                
                # get scaling contrib
                mpl.subplot(312) 
                mpl.plot(xvals,yvals,'o',markerfacecolor = 'none', markeredgecolor='b') 
                wdw_x.append(xvals)
                wdw_scs.append(yvals)
                
                
                #get STDP contribution
                STDPcontrib  = ws[:,i] - ws_sc[:,i]
                mpl.subplot(311)   
                mpl.plot(xvals,STDPcontrib,'o',markerfacecolor = 'none', markeredgecolor ='r') 
                wdw_stdp.append(STDPcontrib)
        
                mpl.subplot(313)    
                mpl.plot(xvals,yvals+STDPcontrib,'o',markerfacecolor = 'none', markeredgecolor =[1,0.4,1])       
        
               
        ax1 = figW.add_subplot(3,1,1)
        ax1.plot([0,wmax],[0,0],'--',color='k',linewidth=2)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.yaxis.set_ticks_position('left') 
        ax1.xaxis.set_ticks_position('bottom')
        ax1.tick_params(labelbottom='off')
        ax1.set_xlim([0,wmax])
        ax1.set_ylim([ymin,ymax])
        
        ax2 = figW.add_subplot(3,1,2)
        ax2.plot([0,wmax],[0,0],'--',color='k',linewidth=2) 
        ax2.set_xlim([0,wmax])
        ax2.set_ylim([ymin_SN,ymax_SN])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.yaxis.set_ticks_position('left') 
        ax2.xaxis.set_ticks_position('bottom')
        ax2.tick_params(labelbottom='off')
        ax3 = figW.add_subplot(3,1,3)
        ax3.plot([0,wmax],[0,0],'--',color='k',linewidth=2) 
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.yaxis.set_ticks_position('left') 
        ax3.xaxis.set_ticks_position('bottom')
        
        ax3.set_xlim([0,wmax])
        ax3.set_ylim([ymin,ymax])
        
        # prop10000
         
        np.save(save_dir + '/WdeltaW_x_' + filename +'_' + wtype + '.npy',wdw_x)
        np.save(save_dir + '/WdeltaW_stdp_' + filename +'_' + wtype + '.npy',wdw_stdp)
        np.save(save_dir + '/WdeltaW_scs_' + filename +'_' + wtype + '.npy',wdw_scs)
        