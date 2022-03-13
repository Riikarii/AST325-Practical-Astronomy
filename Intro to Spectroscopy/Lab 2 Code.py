#!/usr/bin/env python
# coding: utf-8

# In[8]:


#imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.constants as sc
from scipy.optimize import curve_fit as cf
from astropy.io import fits


# In[9]:


#Changing the look of plots
# Changing plotting aesthetics
plt.style.use('bmh')
mpl.rcParams['figure.figsize'] = (9, 5)
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.labelsize'] = 10

red = '#ff4d4a'
green = '#2ffa7d'
blue = '#34bdf7'


# In[10]:


# Importing data
Bb = np.loadtxt('Group_I_BB.dat') # Blackbody data
neon = np.loadtxt('Ne_calib.dat') # Neon calibration data


# In[11]:


# Wavelengths of bright neon lines (from lab document) (Q1)
wavelengths = np.array([
    540.056, 576.441, 582.015, 585.249, 588.189, 594.483, 597.553, 602.000,
    607.433, 609.616,612.884, 614.306, 616.359, 621.728, 626.649, 630.479,
    633.442, 638.299, 640.225, 650.653, 653.288, 659.895, 667.828, 671.704,
    692.947, 703.241, 717.394, 724.512, 743.890, 747.244, 748.887, 753.577,
    754.404
    ])


# In[160]:


# Plotting the blackbody spectrum
plt.figure(1)
plt.plot(Bb, color = blue)

plt.title('Intensity vs Pixel Position')
plt.xlabel('Pixel Position')
plt.ylabel('Intensity')

#Plotting the Neon spectrum
plt.figure(2)
plt.plot(neon, color = blue)

plt.title('Intensity of Neon vs Pixel Position')
plt.xlabel('Pixel Position')
plt.ylabel('Intensity')


# In[158]:


#Identifying centroids (Q2)

def find_peaks(data, threshold):
    """
    Arguments are an array and a threshold, with which values of the array 
    are compared against the threshold to determine whether or not it is a peak.
    This is done by comparing a datapoint to the point before and after it.
    If that point is larger than both points before and after it,
    then it is a peak. Values of the peak and its index are then stored
    in their respective lists.
    """
    
    peaks = []
    indices = []
    for i in range(1, len(data)-1): # Start at 2nd data point, end at 2nd last.
        if data[i] >= threshold:
            if data[i] > data[i-1] and data[i] > data[i+1]:
                #proceeds iff the current point is greater than the points before and after
                peaks.append(data[i])
                indices.append(i)
    return peaks, indices

peaks, pos = find_peaks(neon, 21) # Peaks and pixel positions

pos = np.array(pos) # Converting to an array

# Plotting the neon calibration and peaks
plt.figure(2)
plt.plot(neon, color = blue, label = 'Neon')
plt.plot(pos, peaks, 'X', color = "black", label = 'Peaks') # Plotting position of peaks

plt.title('Intensity of Neon vs Pixel Position')
plt.xlabel('Pixel Position')
plt.ylabel('Intensity')

plt.legend()


# In[28]:


#Obtaining least square fitting (Q3)

# Function to fit
def line(x, m, b):
    return m * x + b

# Optimal parameters and covariance matrices
popt, pcov = cf(line, pos, wavelengths)

### Best fit using least squares (from lab document)

data_points = 33 # Number of data points
plt.figure(5)
plt.plot(pos, wavelengths, 'o', label = 'data', color = blue) # Data
plt.xlabel('Pixel')
plt.ylabel('Wavelength (nm)')

# Construct the matrices
ma = np.array([[np.sum(pos**2), np.sum(pos)], [np.sum(pos), data_points]])
mc = np.array([[np.sum(pos * wavelengths)], [np.sum(wavelengths)]])

# Compute the gradient and intercept
mai = np.linalg.inv(ma)
md = np.dot(mai, mc)

# Overplot the best fit
mfit = md[0, 0]
cfit = md[1, 0]
plt.plot(pos, line(pos, mfit, cfit), label = 'Least Square Fit', color = red) # Fit

plt.title('Wavelength vs Pixel Position')
plt.xlabel('Pixel Position')
plt.ylabel('Wavelength (nm)')
plt.legend()

plt.savefig("pre_least_square_fitting.png")

# # lab method vs curve_fit
# print(mfit, popt[0])
# print(cfit, popt[1])


# In[157]:


#Applying the wavelength solution (Q4)

xs = range(len(BB)) # Range of pixel positions (0 - 1024)

plt.figure(7)
plt.plot(line(xs, mfit, cfit), BB, color = blue) # Applying the wavelength solution
plt.title('Application of Wavelength Solution to Blackbody Spectrum')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity')
plt.savefig("application.png")

peak_index = np.where(BB == max(BB)) # Index of peak
peak_wavelength = line(peak_index[0], mfit, cfit)[0] * 1.e-9 # nanometers
print("Peak Wavelength:", peak_wavelength)

b = sc.Wien # Wien's displacement constant in meter * Kelvin (m times K)
print("Displacement constant:", b)

temp = b/peak_wavelength # The Blackbody Temperature in Kelvin (K)
print("Blackbody temperature:", temp)

#Error calculation for slope and intercept

uncertain = (1/len(wavelengths))*np.sum((wavelengths - (mfit*pos + cfit))**2)

slope_error = (len(wavelengths)*uncertain)/(len(wavelengths)* np.sum(pos**2) - np.sum(pos)**2) #error in slope (mfit)
print("Slope error:", slope_error)

intercept_error = (uncertain*np.sum(pos**2))/(len(wavelengths)*np.sum(pos**2) - np.sum(pos)**2) #error in intercept (cfit)
print("Intercept error:", intercept_error)

first_error = np.sqrt((peak_index[0]*slope_error)**2 + intercept_error**2) * 1.e-9
second_error = (first_error/(peak_wavelength * 1.e-9))*temp


# In[29]:


#Part 2
#Loading in the data

near_inf = fits.open("Near-Infrared.fits")
new_data = near_inf[0].data
r_data = new_data[0]


# In[30]:


plt.imshow(r_data, aspect='auto', cmap='gray', vmax=50000)
plt.colorbar()
plt.title("2D Dispersed Image of Fe II Gas")
plt.xlabel("Spatial Direction")
plt.ylabel("Spectral Direction")


# In[31]:


#Question 1

oh_lines = np.array([16128.608, 16194.615, 16235.376, 16317.161, 16530.650, 16360.385, 
                     16388.492, 16442.155, 16475.648, 16502.365, 16692.380])

print(len(oh_lines))


# In[110]:


#Question 2

sample = np.transpose(np.array(r_data))[139]
plt.plot(sample)
plt.xlabel("Y-Axis Pixels")
plt.ylabel("OH Line Intensity")
plt.title("Spectrum of OH Telluric Sky Lines")

def find_more_peaks(data, cutoff):
    peaks = []
    indices = []
    for num in range(1, len(data) - 1):
        if data[num] >= cutoff:
            if data[num] > data[num-1] and data[num] > data[num+1]:
                peaks.append(data[num])
                indices.append(num)
    return peaks, indices

total_peaks, pixelnum = find_more_peaks(sample, 2500)

#Convert pixelnum into an array
pixelnum = np.array(pixelnum)

#Replot the given graph
plt.figure(3)
plt.plot(sample)

# #Plot the position of the peaks
plt.plot(pixelnum, total_peaks, 'X', color='black', label='Peaks')
plt.xlabel("Y-Axis Pixels")
plt.ylabel("OH Line Intensity")
plt.title("Spectrum of OH Telluric Sky Lines with Peaks")
plt.savefig("spectrum_peaks.png")


# In[171]:


def cubic(x, a, b, c, d):
    return a*(x**3) + b*(x**2) + c*x + d

plt.figure(4)
plt.plot(pixelnum[2:], oh_lines, 'o', color='blue')
plt.xlabel("Y-Axis Pixels")
plt.ylabel("OH Lines Wavelengths")
plt.title("Peak Positions vs. OH Wavelengths")
plt.savefig("unfit_OH.png")
#Two-Dimensional Curve Fit

c1, c2 = cf(cubic, pixelnum[2:], oh_lines)
error = np.sqrt(np.diag(c2))
print(error)
print("The uncertainty in the wavelength solution is:", (error[0]) + (error[1]) + (error[2]) + (error[3]), "m.")

plt.figure(5)
plt.plot(pixelnum[2:], oh_lines, 'o', color='blue')
plt.title("Peak Positions vs. OH Lines with Curve Fit")
plt.xlabel("Y-Axis Pixels")
plt.ylabel("OH Line Wavelengths")
plt.plot(pixelnum[2:], cubic(pixelnum[2:], c1[0], c1[1], c1[2], c1[3]), '-', color='red', label='Curve Fitting')
plt.savefig("fit_OH.png")


# In[169]:


#Question 3

y_s = np.array(range(len(sample))) #Range of pixel positions

plt.figure(6)
plt.title('Spectrum of OH Lines in Sample of 139 x-Pixels')
plt.xlabel('Wavelength ($\AA$)')
plt.ylabel('Intensity')
plt.plot(cubic(y_s, c1[0], c1[1], c1[2], c1[3]), sample)

estimate = cubic(181, c1[0], c1[1], c1[2], c1[3])

intrin_wave = 16439e-10

Delta = estimate*1.e-10 - intrin_wave

print("The estimated wavelength of the [Fe II] emission is:", round(estimate, 3))

print("The velocity of the gas is:", abs(((Delta / intrinsic) * (3.e8)) / 1000), "km/s")


# In[ ]:




