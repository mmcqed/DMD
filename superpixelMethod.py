from __future__ import division
import scipy
import scipy.fftpack as fft
import miscDMD

def superpixelMethod(E_target, resolution):
    # for simplicity we take number of DMD pixels equal to the size of E_target
    # resolution is given in DMDpixels
    
    ny,nx = scipy.shape(E_target)
    
    # Create Fourier mask
    FourierMaskSystemResolution = miscDMD.fourier_mask(ny, nx, resolution)
    
    # Rescale the target to the superpixel resolution
    E_superpixelResolution = miscDMD.rescale_target_superpixel_resolution(E_target)
    
    # Normalize such that the maximum amplitude is 1
    E_superpixelResolution = E_superpixelResolution / ((abs(E_superpixelResolution)).max())
    
    # Calculate which DMD pixels to turn on according to the superpixel method
    DMDpixels = miscDMD.phase_amplitude_to_DMDpixels_lookup_table(E_superpixelResolution)
    DMDpattern = abs(DMDpixels)
    
    # First lens
    DMDpixels_ft = fft.fftshift(fft.fft2(fft.ifftshift(DMDpixels)))
    
    # Spatial filter
    DMDpixels_ft = DMDpixels_ft * FourierMaskSystemResolution
    
    # Second lens
    E_obtained = fft.fftshift(fft.fft2(fft.ifftshift(DMDpixels_ft)))
 
    # Manually rotate the image back   
    E_obtained=scipy.rot90(E_obtained,2)
    
    # Intensity efficiency (total intensity in E_obtained / total incident
    #intensity)
    efficiency = (abs(E_obtained / scipy.sqrt(nx*ny))**2).sum()
    
    # Normalize fields
    E_obtained = E_obtained / scipy.sqrt((abs(E_obtained)**2).sum())
    E_target = E_target / scipy.sqrt((abs(E_target)**2).sum())

    # Calculate the fidelity
    g = miscDMD.inner_product(E_target,E_obtained);
    fidelity = abs(g)**2;
    
    # Return the output
    return [DMDpattern, E_obtained, fidelity, efficiency]