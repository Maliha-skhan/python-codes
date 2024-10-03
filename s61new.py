# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 08:58:08 2019

@author: Maliha
"""

#clear #all; close all; clc
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
#
#%  Stommel's 1961 model of convection in coupled boxes. 
#%
#%  Compute the temperature and salinity difference, here represented
#%  by y and x, between two well-stirred boxes that are both
#%  in contact with a reservoir that has (y, x) = (1, 1).  Both
#%  boxes conduct y and x at rates 1 and delta (delta < 1), and the 
#%  density difference between the boxes is given by d = -y + R*x, where
#%  we are assuming R > 1, typically.  There can be advection (or 
#%  flushing) between the boxes at a rate d*lambda, where the 
#%  flushing does not depend upon the sign of the density anomaly.  
#%  
#%  This code also has 1) flushing with inertia (not especially
#%  interesting), 2) a flickering or random temperature
#%  perturbation (slightly interesting), and 3) an oscillating 
#%  reservoir temperature (hoping for but not yet finding chaos;
#%  may be present in parameter regimes not checked).  
#%
#%  This code has not been fully tested, but seems to reproduce 
#%  S61's two cases fairly well when all three of the 'new' things 
#%  are turned off, of course.
#%
#%  Written by Jim Price, April 28, 1999.  Public domain.
#%

#set(0,'DefaultLineLineWidth',1.2)
#set(0,'DefaultTextFontSize',14)
#set(0,'DefaultAxesLineWidth',1.6)
#set(0,'DefaultAxesFontSize',14)
# 
#clear; format compact; 
nn = 0;

#%  set the model parameters (follows S61, aside from 'new')

R = 2.0;       #%  abs of the ratio of the expansion coefficients, x/y
delta = 1/6;   #%  conduction rate of salinity wrt temperature
#% delta = 1;   #%  conduction rate of salinity wrt temperature
lmbda = 0.2;  #%  inverse non-d flushing rate; = inf for no flushing
q = 0.;        #%  initial flushing rate (0 to 1) 'new'
qdelta = 100.; #%  time constant (inertia) for flushing; 'new'
               #%    set = 1/dtau for equilibrium flushing as in S61
               #%    set = 0.2 for slowly responding flushing 
yres = 1.;     #%  steady reservoir y, = 1 for S61 case  'new'
resosc = 0.;   #%  amplitude of reservoir y oscillation  'new'            
dtau = 0.01;   #%  the time step of non-d time; 0.01 seems OK
nstep = 1500;  #%  number of time steps; 1500 is usually 
               #%    enough to insure convergence to a steady state
#%                 
#%  This model version is set up to do integration over a range
#%  of initial T,S or y,x from 0 to 1.  The increment of T and S 
#%  are set by delT and delS = 1/ni,  where ni is the number of
#%  integrations (n1 = 1 to 20 is reasonable).

yres0 = yres;

ni = 6;
delT = 1/ni;   #%  make ni = 1 or 2 to reduce the number of
               #%     integrations to be done
delS = delT;

#x=[1]*nstep#np.ones((1,nstep))
#y=[1]*nstep#np.ones((1,nstep))
#tau=[1]*nstep#np.ones((1,nstep))
#d=[1]*nstep

for n1 in np.arange(0,1,delT): #n1=0:delT:1 # %  n1 and n2 are used to set the initial T,S
    for n2 in np.arange(0,1,delS):
#        x=np.ones((1,nstep))
#        y=np.ones((1,nstep))
#        tau=np.ones((1,nstep))
        x=[0]*nstep#np.ones((1,nstep))
        y=[0]*nstep#np.ones((1,nstep))
        tau=[0]*nstep#np.ones((1,nstep))
#        d=[0]*nstep
#        d=[1]*nstep
        if (n1==0, n1==1, n2==0, n2==1):  #%  skip all non-boundary initial points
            x[0] = n1;   #%  set the initial temperature
            y[0] = n2;   #%  set the initial salinity
#            tau=np.ones((1,nstep))
            for m in range(2,nstep): #= 2:nstep
                tau[m] = (m)*dtau;  #%  the non-d time
#                tau[m]=tau[m]
#                print(tau)
#                tau[m] = m*dtau;  #%  the non-d time

#%  evaluate the reservoir temperature (y); note that
#%   this temperature is steady if resosc = 0. (the S61 case)
#                yres = yres0 + resosc*np.sin(tau[m]*pi);
                yres = yres0 + resosc*np.sin(tau[m]*np.pi);
#                yres = yres + resosc*np.sin(tau[m]*np.pi);

#% the first part of a second order R-K method; time step forward
#%    by half a time step to make a first guess at the new times
                dr = abs(R*x[m-2]   - y[m-2]);        #%  the density anomaly
#                dr = abs(R*x   - y);        #%  the density anomaly
                qequil = dr/lmbda;                   #%  the equilibrium flushing
                yh = y[m-2] + dtau*(yres - y[m-2])/2 -dtau*y[m-2]*q/2;      #%  time step the temperature 
                xh = x[m-2] + dtau*delta*(1 - x[m-2])/2 -dtau*x[m-2]*q/2;                   #%  time step the salinity
#                yh = y + dtau*(yres - y)/2 -dtau*y*q/2;      #%  time step the temperature 
#                xh = x + dtau*delta*(1 - x)/2 -dtau*x*q/2;  
                qh = q + dtau*qdelta*(qequil - q)/2;  #%  time step the flushing
    
    #% the second part; use the half time step values to make a full step
    
                dr = abs(R*xh  - yh);
                qequil = dr/lmbda;
                y[m] = y[m-2] + dtau*(yres - yh) - dtau*qh*yh;  
                x[m] = x[m-2] + dtau*delta*(1 - xh) - dtau*qh*xh; 
#                y = y + dtau*(yres - yh) - dtau*qh*yh;  
#                x = x + dtau*delta*(1 - xh) - dtau*qh*xh; 
                q = q + dtau*qdelta*(qequil - qh);
                
#                Rx=[i * int(R) for i in x]
#                d[m-2]=R*x[m]-y[m]
            d=[0]*nstep
            for m2 in range(nstep):
                d[m2]=R*x[m2]-y[m2]
#d=d.tolist()

#% now add on a flickering temperature if you want to (or comment out)

#% tflickamp = 0.01;   %  set the amplitude here
#% tflick = tflickamp*unifrnd(-1., 1.);
    #% y(m) = y(m) + tflick;
     
    #end  #%  end of time step loop
    
#            d = R*x  - y;   #%  evaluate the density
    
#            if nn == 0:
#                nn=1
    #        nn = 1;
#Rx=[i * int(R) for i in x]
#d=np.array(Rx)-np.array(y)
#d=d.tolist()
#d=np.array(Rx)-np.array(y)
    #%  make a time series plot of the first case only 

#plt.figure(1)
#clf reset

plt.subplot(211)
#ptau=tau.tolist();
#px=x.tolist();
#lines = plt.plot(tau, x, tau, y)
#plt.setp(lines[0], linewidth=4)
#plt.setp(lines[1], linewidth=2)
plt.plot(tau, x,'b',label='salinity')
plt.plot(tau, y,'r',label='temperature')
plt.xlim(0,np.max(tau))
plt.ylim(0,0.6)
plt.ylabel('T, S diff, non-d')
plt.legend()
plt.title('Experiment 1,1')
plt.show()

plt.subplot(212)
plt.plot(tau, d,'g', label='density')
plt.xlim(0,np.max(tau))
plt.ylim(-0.3,0)
plt.xlabel('time, non-d')
plt.ylabel('density diff')
plt.legend()
plt.show()
#  
#%  contour the density (or flushing rate), and add the (T,S)
#%  trajectories on top of the contours

#%%   
ym = np.arange(0,1.1,0.1)
xm = np.arange(0,1.1,0.1)
dm=np.zeros((11, 11))
for k1 in range(11):
    for k2 in range(11):
        dm[k1][k2] = (1/lmbda)*(R*xm[k2]  - ym[k1]);
#end
#end
dc = np.arange(-10,22,2)#-10:2:20;
#%%
#plt.figure(2)
#clf reset

#dc = -10:2:20;
#c = plt.contour(xm, ym, dm, dc,'k');
#clabel(c);
fig, ax = plt.subplots()
contours=plt.contour(xm, ym, dm, dc, colors='black');
#contours=plt.contour(xm, ym, dm, dc);
ax.clabel(contours, inline=True, fontsize=8, colors='black');#plt.colorbar();
plt.xlabel('salinity diff, non-d')
plt.ylabel('temp diff, non-d')

#hold on

#end   #%  the if on nn = 0 

#[m1 m2] = size(x); [1 1500]
m1 = len(x); m2=m1-1
#
#%  plot the individual trajectories. 
#%  color code according to which equilibrium point
#%  the trajectory ends up on. this will likely have be 
#%  reset if the model parameters (R, delta, lambda) are changed.
   
#if d[m2] >= 0:
#    plt.plot(x, y,'r')
#    plt.plot(x[m2], y[m2], '*r')
#else:
#    plt.plot(x, y, 'g')
#    plt.plot(x[m2], y[m2], '*g')
    
#end

#clear x y d

#end #%  or to gate out non-boundary values of n1, n2
#end #%  loop on n1
#end #%  loop on n2

#%%
#%  make some plots to show where roots (equil. points) are
f=[0]*60;
lhs=[0]*60;
rhs=[0]*60;
for k in range(0,60): #k=1:60
    f[k] = (k+1-30)*0.1;
    lhs[k] = lmbda*f[k];
    rhs[k] = (R/(1 + abs(f[k])/delta)) - 1/(1 + abs(f[k]));
#end

plt.figure(3)
#clf reset
plt.plot(f, rhs, f, lhs)
plt.xlabel('f, flow rate')
plt.ylabel('\phi (f,R,\delta), lhs(f), rhs(f)')
plt.title('roots of S61 model')
plt.grid()

#%   end of the script
