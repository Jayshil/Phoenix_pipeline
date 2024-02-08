import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as clr
import os
import re

def save_image(image, name = 'Image', path = os.getcwd()):
    """
    image = image data
    name = name of the image, default is Image
    path = path of the resultant image,
        default is the the present working directory
    """
    fig, ax = plt.subplots(1, 1, figsize = (5,10))
    im = ax.imshow(image, origin = 'lower', norm = clr.LogNorm(vmin=3, vmax=20000),cmap='PuBu_r')
    fig.colorbar(im, ax = ax, fraction = 0.046, pad = 0.04)
    fig.savefig(path, name + '.png')



def find_nearest_dark_exposure(image, dark_exposure_times, tolerance=0.5):
    """
    Find the nearest exposure time of a dark frame to the exposure time of the image,
    raising an error if the difference in exposure time is more than tolerance.

    This function is taken from the CCD Data reduction guide:
    https://mwcraig.github.io/ccd-as-book/00-00-Preface.html
    
    Parameters
    ----------
    
    image : astropy.nddata.CCDData
        Image for which a matching dark is needed.
    
    dark_exposure_times : list
        Exposure times for which there are darks.
    
    tolerance : float or ``None``, optional
        Maximum difference, in seconds, between the image and the closest dark. Set
        to ``None`` to skip the tolerance test.
    
    Returns
    -------
    
    float
        Closest dark exposure time to the image.
    """

    dark_exposures = np.array(list(dark_exposure_times))
    idx = np.argmin(np.abs(dark_exposures - image.header['exptime']))
    closest_dark_exposure = dark_exposures[idx]

    if (tolerance is not None and 
        np.abs(image.header['exptime'] - closest_dark_exposure) > tolerance):
        return None
        #raise RuntimeError('Closest dark exposure time is {} for flat of exposure '
                           #'time {}.'.format(closest_dark_exposure, image.header['exptime']))
        
    
    return closest_dark_exposure


def inverse_median(a):
    return 1/np.median(a)

#------------------------------------------------------------------------------------------
#-------------------------------Natural Sorting--------------------------------------------
#------------------------------------------------------------------------------------------
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

#------------------------------------------------------------------
#-----------------------Special Maxima-----------------------------
#------------------------------------------------------------------
"""
This function finds a maximum among a dataset which
is not a Dirac-delta.
"""
def special_maxi(x_array):
    x_ary = x_array
    yy = np.max(x_ary)
    abc = np.where(x_ary == yy)
    if abc[0][0] == 255:
        cd = x_ary[abc[0][0]-1]
    else:
        cd = (x_ary[abc[0][0]-1]+x_ary[abc[0][0]+1])/2
    if (yy-cd)>5:
        x_ary[abc[0][0]] = 0
        yy = np.max(x_ary)
    return yy, x_ary

def special_maximum(x_array):
    x_ary = x_array
    maxi = True
    while maxi:
        aaa, xnew = special_maxi(x_ary)
        aaa1, xnew1 = special_maxi(xnew)
        if aaa == aaa1:
            maxi = False
        else:
            aaa = aaa1
            xnew = xnew1
    return aaa

#------------------------------------------------------------------
#--------------Polynomial of arbitraty degree----------------------
#------------------------------------------------------------------

def arbi_poly(x, *params):
    """
    Parameters
    ----------
    x : numpy.ndarray
        takes the values of x where the value of polynomial is needed
    *params: tuple, numpy.ndarray
        values of coefficients of the polynomial
        lowest degree first
    ----------
    returns
    ----------
    numpy.ndarray : values of polynomial at desired x-values
    """
    return sum([p*(x**i) for i, p in enumerate(params)])

#------------------------------------------------------------------
#---------------------Negative Gaussian----------------------------
#------------------------------------------------------------------

def neg_gaus(x, mu, sig, const, aa):
    yy = np.exp(-0.5*((x-mu)/sig)**2)
    zz = -aa*yy + const
    return zz

# Defining a Gaussian to create Spatial Image
def gaus(x, mu, sigma):
    a1 = np.sqrt(2*np.pi*sigma*sigma)**-1
    a2 = np.exp(-0.5*((x-mu)/sigma)**2)
    return a1*a2

#------------------------------------------------------------------
#---------------------Line----------------------------
#------------------------------------------------------------------
# Defining line and inverse line
def line(x, m, c):
    return m*x + c

def inv_line(x, m, c):
    bc = (x-c)/m
    return bc

