### This code will generate an LG_p,l image of the specified size


from __future__ import division
import scipy
import scipy.special
import matplotlib
import matplotlib.pyplot as plt
import sys
import os

os.chdir(os.path.dirname(sys.argv[0]))

# Specify LG mode:

p = 1
l = 1

# Specify image size

nx = 1024
ny = 768

centerX = int(scipy.ceil((nx)/2))
centerY = int(scipy.ceil((ny)/2))


# Define the generalized Laguerre Polynomial:

lag = scipy.special.genlaguerre(p,l)

# Last root of the LG polynomial:

d = (scipy.roots(scipy.special.genlaguerre(max(p,1),l)))[0]

if p == 0:
    w = 1.7 * (p+1)/(p+3) * scipy.sqrt((l+1)/(l+2)) * round(min(centerX,centerY) * scipy.sqrt(2/d))
    
else:
    w = 0.9 * (p+1)/(p+3) * scipy.sqrt((l+1)/(l+2)) * round(min(centerX,centerY) * scipy.sqrt(2/d))


matrix = [[((i-centerX)+(j-centerY)*1j) for i in range(nx)] for j in range(ny)]
matrix = scipy.asarray(matrix)
phase = l*scipy.angle(matrix)


r = abs(matrix)
r = scipy.asarray(r)

amplitude = r**l * scipy.exp(-(r**2)/(w**2)) * lag(2*(r**2)/(w**2))

LG = amplitude * scipy.exp(1j*phase)

matplotlib.image.imsave('LG_'+str(p)+','+str(l)+'_phase.jpg', scipy.angle(LG))
matplotlib.image.imsave('LG_'+str(p)+','+str(l)+'_amp.jpg', abs(LG), cmap=plt.cm.gray)