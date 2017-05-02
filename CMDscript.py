# -*- coding: utf-8 -*-
"""
Created on Sun May 01 20:28:46 2016

@author: Anna
"""
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import pylab as pl
import sys
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator

#plot_title = sys.argv[1]
#redfilter = sys.argv[2]
#bluefilter = sys.argv[3]
plot_title = "insert title"
redfilter = "insert red filter name"
bluefilter = "insert blue filter name"

#Real Data
"""Here you will need to change the path to your phot.dat file"""
dat = np.genfromtxt("phot.dat")
V = np.array(dat[:,5])
Verr = np.array(dat[:,6])
I = np.array(dat[:,13])
Ierr = np.array(dat[:,14])
VmI = V-I
VmIerr = (((Verr)**2)+(((Ierr)**2)))**.5
print VmIerr

#fake error stuff
"""Here you will need to change the path to your fake.dat file"""
fdat = np.loadtxt('fake.dat')

fdat = np.asarray([d for d in fdat if not 99.999 in d])
fVerr = np.array(fdat[:,2])
fIerr = np.array(fdat[:,3])
fI = np.array(fdat[:,1])
fV = np.array(dat[:,0])
fVmIerr = (fVerr**2 + fIerr**2)**0.5
print fVmIerr

#Here I am finding the max and min values of the data
#We will use this to automate the figsize thing 
maxV = np.amax(V) + .3
print max(V)
maxI = np.amax(I) + .3
minV = np.amin(V) + 1.3
minI = np.amin(I) + 1.3
meanVmI = np.mean(VmI)
maxVmI = (meanVmI) + 5*np.std(VmI)
minVmI = (meanVmI) - 4*np.std(VmI)
Ierrup = np.around(maxI - .5)
Verrup = np.around(maxV - .5)
Ierrlow = np.around(minI + .5)
Verrlow = np.around(minV + .5)
errx = maxVmI -.3
print maxVmI
print errx
#maxVmI = np.amax(VmI) + .2
#minVmI = np.amin(VmI) - .2
#print minI
#print min(I)


#Making plain scatter plot
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1) 
ax.scatter(VmI, I, color = 'black', s = 2)
ax.set_ylim(maxI, minI)
ax.set_xlim(minVmI, maxVmI)
ax.set_xlabel(bluefilter + '-' + redfilter + ' (mag)')
ax.set_ylabel(redfilter + ' (mag)')
fig.text(.18, .85, 'Number of stars: '+str(len(V)))

majorLocator = MultipleLocator(1)
majorFormatter = FormatStrFormatter('%d')
minorLocator = AutoMinorLocator(10)
ax.yaxis.set_major_locator(majorLocator)
ax.yaxis.set_major_formatter(majorFormatter)
ax.yaxis.set_minor_locator(minorLocator)

plt.suptitle(plot_title)
#ax.set_title(plot_title)

#adding error to the plot
#c = 3
c = errx
#location of the line
errlist = []
ylist = []
Verrlist = []
VmIerrlist = []
#Ilist = range(18, 27)
Ilist = range(int(Ierrlow), int(Ierrup))

for a in Ilist:
    Iw = np.where(I > a)
    Iwr = np.where(I[Iw] < a+1)
    fIw = np.where(fI > a)
    fIwr = np.where(fI[fIw] < a+1)
    Ierravg = ((np.mean(Ierr[Iwr]))**2 + (np.mean(abs(fIerr[fIwr]))**2))**.5
    errlist.append(Ierravg)
    ylist.append(a+.5)
    VmIerravg = (np.mean(VmIerr[Iwr])**2 + np.mean(abs(fVmIerr[fIwr]))**2)**.5
    VmIerrlist.append(VmIerravg)
    
xlist = c*np.ones_like(ylist)
plt.errorbar(xlist, ylist, xerr = VmIerrlist, yerr= errlist, fmt = '.', capsize=0)


#adding in the contour script
def multidigitize(VmI,I,binsVmI,binsV):
    dVmI = np.digitize(VmI.flat, binsVmI)
    dI = np.digitize(I.flat, binsV)
    return dVmI,dI

def linlogspace(xmin,xmax,n):
    return np.logspace(np.log10(xmin),np.log10(xmax),n)

#here's the contour actual values
def adaptive_param_plot(VmI,I,
                        bins=10,
                        threshold=5,
                        marker='.',
                        marker_color=None,
                        ncontours=5,
                        fill=False,
                        mesh=False,
                        contourspacing=linlogspace,
                        mesh_alpha=0.5,
                        norm=None,
                        axis=None,
                        cmap=None,
                        **kwargs):
    if axis is None:
        axis = pl.gca()
        axis.set_ylim(28, 18)
    ok = np.isfinite(VmI)*np.isfinite(I)
    
    if hasattr(bins, 'ndim') and bins.ndim == 2:
        nbinsVmI, nbinsI = bins.shape[0]-1, bins.shape[1]-1
    else:
        try:
            nbinsVmI = nbinsI = len(bins)-1
        except TypeError:
            nbinsVmI = nbinsI = bins
    H, bVmI, bI = np.histogram2d(VmI[ok], I[ok], bins = bins)
    
    dVmI, dI = multidigitize(VmI[ok], I[ok], bVmI, bI)
    
    plottable = np.ones([nbinsVmI+2, nbinsI+2], dtype = 'bool')
    plottable_hist = plottable[1:-1, 1:-1]
    assert H.shape == plottable_hist.shape
    
    plottable_hist[H > threshold] = False
    
    H[plottable_hist] = 0
    
    toplot = plottable[dVmI, dI]
    
    cVmI = (bVmI[1:]+bVmI[:-1])/2
    cI = (bI[1:]+bI[:-1])/2
    levels = contourspacing(threshold-0.5, H.max(), ncontours)
    
    if cmap is None:
        cmap = plt.cm.get_cmap()
        cmap.set_under((0,0,0,0))
        cmap.set_bad((0,0,0,0))
    
    if fill:
        con = axis.contourf(cVmI, cI, H.T, levels= levels, norm = norm, cmap = cmap,  **kwargs)
    else: 
        con = axis.contour(cVmI, cI, H.T,levels=levels,norm=norm,cmap=cmap,**kwargs) 
    if mesh: 
        mesh = axis.pcolormesh(bVmI, bI, H.T, **kwargs)
        mesh.set_alpha(mesh_alpha)
        #Is there a way to add lines w the contour levels?
    
    if 'linestyle' in kwargs:
        kwargs.pop('linestyle')
        
    #if i wanted to plot the scatter from this script intstead, but I can't make it look as nice
   # axis.plot(VmI[ok][toplot],
    #          I[ok][toplot],
     #         linestyle='none',
      #        marker=marker,
       #       markerfacecolor=marker_color,
        #      markeredgecolor=marker_color,
         #     **kwargs)
    
    return cVmI, cI, H, VmI[ok][toplot], I[ok][toplot]

adaptive_param_plot(VmI, I, bins = 100, fill = True, ncontours = 7, threshold = 12, axis = ax)





#SECOND PLOT




#Making plain scatter plot
#fig = plt.figure()
ax = fig.add_subplot(1, 2, 2) 
ax.scatter(VmI, V, color = 'black', s = 2)
ax.set_ylim(maxV, minV)
ax.set_xlim(minVmI, maxVmI)
#ax.set_ylim(29, 18)
#ax.set_xlim(-.98, 3.48)
ax.set_xlabel(redfilter + '-' + bluefilter + ' (mag)')
ax.set_ylabel(bluefilter + ' (mag)')
fig.text(.6, .85, 'Number of stars: '+str(len(V)))
majorLocator = MultipleLocator(1)
majorFormatter = FormatStrFormatter('%d')
minorLocator = AutoMinorLocator(10)
ax.yaxis.set_major_locator(majorLocator)
ax.yaxis.set_major_formatter(majorFormatter)
ax.yaxis.set_minor_locator(minorLocator)
#ax.set_title(plot_title)


#adding error to the plot
#c = 3
c = errx
#location of the line
errlist = []
ylist = []
Verrlist = []
VmIerrlist = []
#Ilist = range(18, 28)
Ilist = range(int(Verrlow), int(Verrup))

for a in Ilist:
    Iw = np.where(I > a)
    Iwr = np.where(I[Iw] < a+1)
    fIw = np.where(fI > a)
    fIwr = np.where(fI[fIw] < a+1)
    Ierravg = ((np.mean(Ierr[Iwr]))**2 + (np.mean(abs(fIerr[fIwr])))**2)**.5
    errlist.append(Ierravg)
    ylist.append(a+.5)
    VmIerravg = (np.mean(VmIerr[Iwr])**2 + np.mean(fVmIerr[fIwr])**2)**.5
    VmIerrlist.append(VmIerravg)

xlist = c*np.ones_like(ylist)
plt.errorbar(xlist, ylist, xerr = VmIerrlist, yerr= errlist, fmt = '.', capsize = 0)


#adding in the contour script
def multidigitize(VmI,V,binsVmI,binsV):
    dVmI = np.digitize(VmI.flat, binsVmI)
    dV = np.digitize(V.flat, binsV)
    return dVmI,dV

def linlogspace(xmin,xmax,n):
    return np.logspace(np.log10(xmin),np.log10(xmax),n)

#here's the contour actual values
def adaptive_param_plot(VmI,V,
                        bins=10,
                        threshold=5,
                        marker='.',
                        marker_color=None,
                        ncontours=5,
                        fill=False,
                        mesh=False,
                        contourspacing=linlogspace,
                        mesh_alpha=0.5,
                        norm=None,
                        axis=None,
                        cmap=None,
                        **kwargs):
    if axis is None:
        axis = pl.gca()
        axis.set_ylim(28, 18)
    ok = np.isfinite(VmI)*np.isfinite(V)
    
    if hasattr(bins, 'ndim') and bins.ndim == 2:
        nbinsVmI, nbinsV = bins.shape[0]-1, bins.shape[1]-1
    else:
        try:
            nbinsVmI = nbinsV = len(bins)-1
        except TypeError:
            nbinsVmI = nbinsV = bins
    H, bVmI, bV = np.histogram2d(VmI[ok], V[ok], bins = bins)
    
    dVmI, dV = multidigitize(VmI[ok], V[ok], bVmI, bV)
    
    plottable = np.ones([nbinsVmI+2, nbinsV+2], dtype = 'bool')
    plottable_hist = plottable[1:-1, 1:-1]
    assert H.shape == plottable_hist.shape
    
    plottable_hist[H > threshold] = False
    
    H[plottable_hist] = 0
    
    toplot = plottable[dVmI, dV]
    
    cVmI = (bVmI[1:]+bVmI[:-1])/2
    cV = (bV[1:]+bV[:-1])/2
    levels = contourspacing(threshold-0.5, H.max(), ncontours)
    
    if cmap is None:
        cmap = plt.cm.get_cmap()
        cmap.set_under((0,0,0,0))
        cmap.set_bad((0,0,0,0))
    
    if fill:
        con = axis.contourf(cVmI, cV, H.T, levels= levels, norm = norm, cmap = cmap,  **kwargs)
    else: 
        con = axis.contour(cVmI, cV, H.T,levels=levels,norm=norm,cmap=cmap,**kwargs) 
    if mesh: 
        mesh = axis.pcolormesh(bVmI, bV, H.T, **kwargs)
        mesh.set_alpha(mesh_alpha)
        #Is there a way to add lines w the contour levels?
    
    if 'linestyle' in kwargs:
        kwargs.pop('linestyle')

    
    return cVmI, cV, H, VmI[ok][toplot], I[ok][toplot]

adaptive_param_plot(VmI, V, bins = 100, fill = True, ncontours = 7, threshold = 12, axis = ax)

plt.savefig('CMD.pdf')




