from __future__ import division
import scipy
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import superpixelMethod
import miscDMD
#change
# Diffraction limited spot size in target plane in units of DMD pixels:
resolution = 8

# Specify files from Images folder to be encoded:

amplitude = "dog.jpg"
phase = "cat.jpg"

# Create a target field of the size of the DMD:

E_target, maxIntensity = miscDMD.make_field_from_image(amplitude, phase)

# Apply the superpixel method to the target field. Here superpixels of size
# 4x4 are used

DMDpattern_superpixel, E_superpixel, fidelity_superpixel, efficiency_superpixel = superpixelMethod.superpixelMethod(E_target, resolution)


E_target_lowres = miscDMD.spatial_filter(E_target)
fidelity_superpixel_lowres = abs(miscDMD.inner_product(E_superpixel,E_target_lowres))**2

# Plot the results:

f1 = plt.figure()
a = f1.add_subplot(1,2,1)
plt.imshow(abs(E_target/ scipy.sqrt((abs(E_target)**2).sum())),"gray")
a.set_title('target intensity') 
a = f1.add_subplot(1,2,2)
plt.imshow(scipy.angle(E_target/ scipy.sqrt((abs(E_target)**2).sum())),"gray")
a.set_title('target phase')
 
f2 = plt.figure()
a = f2.add_subplot(1,1,1)
plt.imshow(DMDpattern_superpixel, "gray")
a.set_title('DMD pattern superpixel')

f3 = plt.figure()
a = f3.add_subplot(1,2,1)
plt.imshow(abs(E_superpixel/ scipy.sqrt((abs(E_target_lowres)**2).sum())),"gray")
a.set_title('superpixel intensity') 
a = f3.add_subplot(1,2,2)
plt.imshow(scipy.angle(E_superpixel/ scipy.sqrt((abs(E_target_lowres)**2).sum())),"gray")
a.set_title('superpixel phase')

plt.show()