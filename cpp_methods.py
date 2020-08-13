
from brian2.units import ms,mV,second
from brian2 import implementation, check_units

# Implementation of synaptic scaling
#
#
@implementation('cpp', code=r'''
   
    double syn_scale(double a, double vANormTar, double Asum_post, double veta_scaling, double t, int syn_active, int i, int j) {
      
      double a_out;

      if (Asum_post==0.){
          a_out = 0.;
      }
      else{
          a_out = a*(1 + veta_scaling*(vANormTar/Asum_post-1));
      }

      return a_out;
    } ''')
@check_units(a=1, vANormTar=1, Asum_post=1, eta_scaling=1, t=second, syn_active=1, i=1, j=1, result=1)
def syn_scale(a, vANormTar, Asum_post, eta_scaling, t, syn_active, i, j):
    return -1.


# Implementation of E<-I synaptic scaling
@implementation('cpp', code=r'''
   
    double syn_EI_scale(double a, double vANormTar, double Asum_post, double veta_scaling, double t, int syn_active, int i, int j) {
      
      double a_out;

      if (Asum_post==0.){
          a_out = 0.;
      }
      else{
          a_out = a*(1 + veta_scaling*(vANormTar/Asum_post-1));
      }

      return a_out;
    } ''')
@check_units(a=1, vANormTar=1, Asum_post=1, eta_scaling=1, t=second, syn_active=1, i=1, j=1, result=1)
def syn_EI_scale(a, vANormTar, Asum_post, eta_scaling, t, syn_active, i, j):
    return -1.