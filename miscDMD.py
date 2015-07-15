from __future__ import division
import scipy
import scipy.fftpack as fft
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import sys
import os
#import misc
from scipy.constants import pi

#os.chdir(os.path.dirname(sys.argv[0]))


def make_field_from_image(amplitudeimage, phaseimage):

    # Take two images, one for intensity and one for phase, and combine them
    # into a complex field:
    
    I = scipy.array(mpimg.imread("Images/"+amplitudeimage))
    if len(scipy.shape(I)) == 3:
        I = scipy.dot(I[:,:,0:3], [0.299, 0.587, 0.144])
        
    Igs = I / I.max()    
    #Igs = scipy.minimum(1, Igs*1.3) # QUESTION
    
    maxIntensity = Igs.max()/Igs.sum()
    
    phase = scipy.array(mpimg.imread("Images/"+phaseimage))
    if len(scipy.shape(phase)) == 3:
        phase = phase[:,:,0]/(phase[:,:,0].max())
    else:
        phase = phase/phase.max()

    E_target = scipy.sqrt(Igs)*scipy.exp((1.9*pi*phase-0.95*pi)*1j)
    
    return [E_target, maxIntensity]
    
    
    
def fourier_mask(ny,nx,resolution):

    # Create circular aperture around the center

    maskCenterX = int(scipy.ceil((nx+1)/2))
    maskCenterY = int(scipy.ceil((ny+1)/2))
    
    ### Code optimization purposes
    angle = ny/nx
    nres = (ny/resolution/2)**2
    ###
    
    return [[(( i+1 - maskCenterY)**2 + (angle*( j+1 - maskCenterX))**2 < nres) for j in range(nx)] for i in range(ny)]
    


def inner_product(E1,E2):
    
    norm_E1 = scipy.sqrt((abs(E1)**2).sum())
    norm_E2 = scipy.sqrt((abs(E2)**2).sum())
    
    gamma = (E1*E2.conj()).sum()
    return gamma / (norm_E1*norm_E2)
    
    
    
def phase_amplitude_to_DMDpixels_lookup_table(E):

    m=4
    
    # Load lookup table
    targetFields = scipy.loadtxt("lookup_table/targetFields", dtype="complex", delimiter=",")
    lookupTable = scipy.loadtxt("lookup_table/lookupTable", dtype="int", delimiter=",")
    gridParameters = scipy.loadtxt("lookup_table/gridParameters", dtype="float", delimiter=",")
    
    maxAmplitude = gridParameters[0] # Maximum amplitude the superpixel can generate
    stepSize = gridParameters[1]
    lookupTable_x0 = (len(lookupTable)+1)/2
    
    ny,nx = scipy.shape(E)
    
    #DMDpixels = [[0 for i in range(nx*m)] for j in range(ny*m)]
    #DMDpixels = scipy.asarray(DMDpixels)
    DMDpixels = scipy.zeros((ny*m,nx*m))
    
    # Decrease maximum amplitude to 1 if needed
    # E = scipy.exp(scipy.angle(E)*1j)*scipy.minimum(abs(E), 1) # QUESTION isn't E already normalized
    
    # Correct for overall phase offset
    E = E*scipy.exp(11/16*(pi)*1j) # QUESTION WHY?
    
    # Choose normalization: maxAmplitude for highest efficiency, highRes to
    # restrict the modulation to a smaller and denser disk in the complex plane
    
    E = E * maxAmplitude
    
    # Loop over superpixels. Find correct combination of pixels to turn on in the lookup table and put
    # them into the 'DMDpixels' matrix that contains the DMD pattern
    E = E / stepSize

    for j in range(ny):
        for i in range(nx):
            idx = lookupTable[round(E[j,i].imag+lookupTable_x0)-1,round(E[j,i].real+lookupTable_x0)-1]
            pixels = targetFields[idx-1,1:m**2+1]
            shift = (m*j) % (m**2)
            pixels = scipy.hstack((pixels[shift:m**2+1], pixels[0:shift]))
            DMDpixels[m*j:m*j+m, m*i:m*i+m] = scipy.transpose(pixels.reshape(4,4))
            
    phaseFactor = [[scipy.exp(1j*pi*((k+1)+4*(l+1))/8) for l in range(nx*m)] for k in range (ny*m)]

    DMDpixels = DMDpixels * phaseFactor
    
    return DMDpixels
    
    
    
def rescale_target_superpixel_resolution(E_target):
    '''Rescale the target field to the superpixel resolution (currently only 4x4 superpixels implemented)'''
    
    superpixelSize = 4
    
    ny,nx = scipy.shape(E_target)
    
    maskCenterX = scipy.ceil((nx+1)/2)
    maskCenterY = scipy.ceil((ny+1)/2)
    
    nSuperpixelX = int(nx/superpixelSize)
    nSuperpixelY = int(ny/superpixelSize)
    
    FourierMaskSuperpixelResolution = fourier_mask(ny,nx,superpixelSize)
    
    
    E_target_ft = fft.fftshift(fft.fft2(fft.ifftshift(E_target)))
    
    #Apply mask
    E_target_ft = FourierMaskSuperpixelResolution*E_target_ft
    
    #Remove zeros outside of mask
    E_superpixelResolution_ft = E_target_ft[(maskCenterY - scipy.ceil((nSuperpixelY-1)/2)-1):(maskCenterY + scipy.floor((nSuperpixelY-1)/2)),(maskCenterX - scipy.ceil((nSuperpixelX-1)/2)-1):(maskCenterX + scipy.floor((nSuperpixelX-1)/2))]
    
    # Add phase gradient to compensate for anomalous 1.5 pixel shift in real
    # plane
    
    phaseFactor = [[(scipy.exp(2*1j*pi*((k+1)/nSuperpixelY+(j+1)/nSuperpixelX)*3/8)) for j in range(nSuperpixelX)] for k in range(nSuperpixelY)] # QUESTION
    E_superpixelResolution_ft = E_superpixelResolution_ft*phaseFactor
    
    # Fourier transform back to DMD plane
    E_superpixelResolution = fft.fftshift(fft.ifft2(fft.ifftshift(E_superpixelResolution_ft)))

    return E_superpixelResolution
    
    
    
def spatial_filter(E_target):

    superpixelSize = 4
    
    ny,nx = scipy.shape(E_target)

    FourierMaskSuperpixelResolution = fourier_mask(ny,nx,superpixelSize)

    E_target_ft = fft.fftshift(fft.fft2(fft.ifftshift(E_target)))

    #Apply mask
    E_target_ft = FourierMaskSuperpixelResolution*E_target_ft

    # Fourier transform back to DMD plane
    E_systemResolution = fft.fftshift(fft.ifft2(fft.ifftshift(E_target_ft)))
    
    return E_systemResolution