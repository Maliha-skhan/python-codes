# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 15:36:54 2019

@author: Maliha
"""
# Link to the webpage: 
#https://nbviewer.jupyter.org/github/brian-rose/climlab/blob/master/courseware/Snowball%20Earth%20in%20the%20EBM.ipynb

#"Ice - Albedo Feedback and runaway glaciation"
#"Here we will use the 1-dimensional diffusive Energy Balance Model (EBM) to"
#"explore the effects of albedo feedback and heat transport on climate sensitivity."
#
#"Load packages"
from __future__ import division, print_function
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import climlab
from climlab import constants as const
from climlab import legendre

"Annual-mean model with albedo feedback: adjustment to equilibrium"
"A version of the EBM in which albedo adjusts to the current position "
"of the ice line, wherever T<Tf"
model1 = climlab.EBM_annual( num_points = 180, a0=0.3, a2=0.078, ai=0.62)
print(model1)

model1.integrate_years(5)
Tequil = np.array(model1.Ts)
ALBequil = np.array(model1.albedo)
OLRequil = np.array(model1.OLR)
ASRequil = np.array(model1.ASR)

#Let's look at what happens if we perturb the temperature -- make it 20ºC colder everywhere!

model1.Ts -= 20.
model1.compute_diagnostics()

# Let's take a look at how we have just perturbed the absorbed shortwave:

my_ticks = [-90,-60,-30,0,30,60,90]
lat = model1.lat

fig = plt.figure( figsize=(12,5) )

ax1 = fig.add_subplot(1,2,1)
ax1.plot(lat, Tequil, label='equil') 
ax1.plot(lat, model1.Ts, label='pert' )
ax1.grid()
ax1.legend()
ax1.set_xlim(-90,90)
ax1.set_xticks(my_ticks)
ax1.set_xlabel('Latitude')
ax1.set_ylabel('Temperature (degC)')

ax2 = fig.add_subplot(1,2,2)
ax2.plot( lat, ASRequil, label='equil') 
ax2.plot( lat, model1.ASR, label='pert' )
ax2.grid()
ax2.legend()
ax2.set_xlim(-90,90)
ax2.set_xticks(my_ticks)
ax2.set_xlabel('Latitude')
ax2.set_ylabel('ASR (W m$^{-2}$)')

plt.show()

#So there is less absorbed shortwave now, because of the increased albedo. 
#The global mean difference is:
print('The global mean difference is')
print(climlab.global_mean( model1.ASR - ASRequil ))

