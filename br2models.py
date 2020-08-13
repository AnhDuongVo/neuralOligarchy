condlif_memnoise = '''
              dV/dt = (El-V + ge*(Ee-V) + gi*(Ei-V))/tau : volt
              Vt : volt 

              AsumEE : 1
              AsumEI : 1

              ANormTar : 1
              iANormTar : 1
              '''
              
syn_cond_EE_exp = '''
                  dge /dt = -ge/tau_e : 1
                  '''

syn_cond_EI_exp = '''
                  dgi /dt = -gi/tau_i : 1
                  '''

syn_cond_EE_alpha = '''
                    dge /dt = (xge-ge)/tau_e : 1
                    dxge /dt = -xge/tau_e : 1
                    '''

syn_cond_EE_biexp = '''
                    dge/dt = (invpeakEE*xge-ge)/tau_e_rise : 1
                    dxge/dt = -xge/tau_e                   : 1
                    '''

syn_cond_EI_alpha = '''
                    dgi /dt = (xgi-gi)/tau_i : 1
                    dxgi /dt = -xgi/tau_i : 1
                    '''

syn_cond_EI_biexp = '''
                    dgi/dt = (invpeakEI*xgi-gi)/tau_i_rise : 1
                    dxgi/dt = -xgi/tau_i                   : 1
                    '''

synEE_static = 'a : 1'

synEE_noise_add   = '''da/dt = syn_active*syn_sigma**0.5*xi : 1'''


synEE_noise_mult  = '''da/dt = syn_active*a*syn_sigma**0.5*xi : 1'''

synEE_mod = '''            
            syn_active : integer

            dApre  /dt = -Apre/taupre  : 1 (event-driven)
            dApost /dt = -Apost/taupost : 1 (event-driven)
            '''

synEE_scl_mod = 'AsumEE_post = a : 1 (summed)'

synEI_scl_mod = 'AsumEI_post = a : 1 (summed)'


synEE_pre_exp   = '''
                  ge_post += syn_active*a
                  Apre = syn_active*Aplus
                  '''

synEE_pre_alpha = '''
                  xge_post += syn_active*a/norm_f_EE
                  Apre = syn_active*Aplus
                  '''

synEE_pre_biexp = '''
                  xge_post += syn_active*a/norm_f_EE
                  Apre = syn_active*Aplus
                  '''

synEI_pre_exp   = '''
                  gi_post += syn_active*a
                  Apre = syn_active*Aplus
                  '''

synEI_pre_alpha = '''
                  xgi_post += syn_active*a/norm_f_EI
                  Apre = syn_active*Aplus
                  '''

synEI_pre_biexp = '''
                  xgi_post += syn_active*a/norm_f_EI
                  Apre = syn_active*Aplus
                  '''

synEI_pre_sym_exp   = '''
                      gi_post += syn_active*a
                      Apre = syn_active*Aplus
                      a = a - stdp_active*LTD_a
                      '''

synEI_pre_sym_alpha = '''
                       xgi_post += syn_active*a/norm_f_EI
                       Apre = syn_active*Aplus
                       a = a - stdp_active*LTD_a
                       '''

synEI_pre_sym_biexp = '''
                       xgi_post += syn_active*a/norm_f_EI
                       Apre = syn_active*Aplus
                       a = a - stdp_active*LTD_a
                       '''

syn_pre_STDP = '''
                 a = syn_active*clip(a+Apost*stdp_active, 0, amax)
                 '''

syn_post = '''
           Apost = syn_active*Aminus
           '''

synEI_post_sym = '''
                 Apost = syn_active*Aplus
                 '''

syn_post_STDP = '''
                a = syn_active*clip(a+Apre*stdp_active, 0, amax)
                '''

synEE_scaling = '''
                a = syn_active*syn_scale(a, ANormTar, AsumEE_post, eta_scaling, t, syn_active, i, j)
                '''

synEI_scaling = '''
                a = syn_active*syn_EI_scale(a, iANormTar, AsumEI_post, eta_scaling, t, syn_active, i, j)
                '''
                
# poisson noise is not implemented yet

condlif_poisson = '''
              dV/dt = (El-V + (gfwd+ge)*(Ee-V) + gi*(Ei-V))/tau : volt
              Vt : volt 
              dge /dt = -ge/tau_e : 1
              dgfwd /dt = -gfwd/tau_e : 1
              dgi /dt = -gi/tau_i : 1

              AsumEE : 1
              AsumEI : 1

              ANormTar : 1
              iANormTar : 1
              '''