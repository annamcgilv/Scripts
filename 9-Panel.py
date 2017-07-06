#original code Hess_plot.py, revised by Anthony Pahl 11/30/2016
#revised by Anna McGilvray 7/5/2017
#syntax: CMD_res_SF_Z_plot.py out2.cmd out2.final output2.png
#
#plots observed CMD, modelled CMD, and residual based on calcsfh
#outputs. also plots SF, SFR, and Z vs. time for said fits.
#attempts to recreate 'plot_cmd' output images from match.

import numpy as np
import matplotlib
#uncomment line below for TACC usage
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import sys

cmdfile_name = sys.argv[1]
outfile_name = sys.argv[2]
plot_name = sys.argv[3]
figsize=(8, 8)
subsize = 0.25


subx1, suby1 = 0.05, 0.05
subx2, suby2 = 0.38, 0.38


subx3, suby3 = 0.72, 0.72
interpolation = 'hanning'
#you dont need to change these to change the label, see line 368
#xlabel, ylabel = 'F606W - F814W', 'F606W'
xlabel, ylabel = 'F555W - F814W', 'F555W'

def makecmap(arr):
    x = np.linspace(arr.min(), arr.max(), 50)
    #x = np.arange(arr.min(), arr.max(), 1)
    steps = (x - x.min()) / (x.max() - x.min())
    numerator = np.arcsinh(x) - np.arcsinh(x.min())
    denominator = np.arcsinh(x.max()) - np.arcsinh(x.min())
    mapping = 1 - numerator / denominator
    cdict = {}
    for key in ('red', 'green', 'blue'):
        cdict[key] = np.vstack([steps, mapping, mapping]).transpose()
    return cl.LinearSegmentedColormap('new_colormap', cdict, N=1024)

#In this function, you might want to change the colormap ('cm') or maybe the xlim and ylim for scaling
def plot_HessD(fig, arr, subx1, suby2, subsize, extent, cmin, cmax, interpolation,
               title, xlabel, ylabel,col_bool):
    if col_bool:
        cm = matplotlib.cm.jet#makecmap(arr)
        nm = None
        cmap_ofs=0.
    else:
        cm=makecmap(arr)
        nm=None
        cmap_ofs=0.05
    #cm.set_gamma(0.2)
    ax = fig.add_axes([subx1, suby2, subsize, subsize])
    #Change cm to gist_yarg if you want grayscale
    ax.imshow(arr, cmap=cm, aspect='auto', extent=extent,
              interpolation=interpolation, norm=nm, clim = (cmin, cmax))
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.get_xticks()
    #ax.set_xticklabels(ax.get_xticks(), size=8)
    #ax.set_yticklabels(ax.get_yticks(), size=8)
    lowx = extent[0]*0.9
    highx = extent[1]*0.7
    #ax.set_xlim(np.array(extent[0])-lowx, np.array(extent[1])-highx)
    ax.set_ylim(maxdepth+.1, mindepth+1)
    #ax.set_xlim(lowx, highx)
    ax.set_xlim(minshift+.3, maxshift - 1)
    #ax.set_xticks(np.array([-1, 0, 1, 2, 3]))
    #ax.set_xticklabels(np.array([-1.0, 0, 1.0, 2.0, 3.0]), size = 8)
    ax.set_yticklabels(ax.get_yticks(), size = 8)
    ax.set_xticklabels(ax.get_xticks(), size = 8)
    #cax = fig.add_axes([subx1+0.13, suby2+0.35, subsize-0.15, subsize-0.365])
    cax = fig.add_axes([subx1+0.07+cmap_ofs, suby2+0.225,
                        subsize-0.10-cmap_ofs, subsize-0.235])
    cb_arr = np.linspace(arr.min(), arr.max(), 100).reshape(1, 100)
    cax_limits = [np.floor(arr.min()), np.ceil(arr.max()), 0, 1]
    #Change cmap = cm to cmap = 'gist_yarg' if you want grayscale
    cax.imshow(cb_arr, cmap=cm, aspect='auto', extent=cax_limits, interpolation='nearest', norm=nm)
    cax.plot([arr.min(), arr.min()], [0, 1], 'k-')
    cax.plot([arr.max(), arr.max()], [0, 1], 'w-')
    cax.axis(cax_limits)
    cax.yaxis.set_visible(False)
    cax.xaxis.tick_bottom()
    cax.xaxis.set_major_locator(plt.LinearLocator(5))
    cax.set_xticklabels(cax.get_xticks(), size=5)
    #cax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    return ax, cax

def plot_Scatter(fig, x, y, subx1, suby2, subsize, title, xlabel, ylabel,linestyle, errlow=[], errhigh=[], bound=False):    
      x = np.array(x)
      y = np.array(y)
      
      if errlow == []:
          errlow = np.zeroes_like(x)
      else:
          errlow = np.array(errlow)
      
      if errhigh == []:
          errhigh = np.zeroes_like(x)
      else:
          errhigh = np.array(errhigh)    
          
      nonnan = np.logical_not(np.isnan(x)) & np.logical_not(np.isnan(y))
      x = x[nonnan]
      y = y[nonnan]
      errlow = errlow[nonnan]
      errhigh = errhigh[nonnan]
    
      ax = fig.add_axes([subx1, suby2, subsize, subsize])
      if len(errlow) == 0:
		ax.plot(x, y,'k', linestyle = linestyle)
      else:
		ax.errorbar(x, y, yerr=[errlow, errhigh], fmt='k', capsize=0, ecolor='k', elinewidth=0.7)
      ax.set_title(title, fontsize=8)
      ax.set_xlabel(xlabel, fontsize=8)
      ax.set_ylabel(ylabel, fontsize=8)
      if bound == True:
		ax.set_xlim(14.0, min(x))
		ax.set_xticks(np.array([14, 12, 10, 8, 6, 4, 2, 0]))
		ax.set_xticklabels(ax.get_xticks(), size=8)
		ax.set_yticklabels(ax.get_yticks(), size=8)
      else:
		ax.set_xlim(max(x),min(x))
		ax.set_xticklabels(ax.get_xticks(), size=8)
		ax.set_yticklabels(ax.get_yticks(), size=8)
		#ax.set_ylim(min(y)*0.8, max(y)*1.2)
      #ax.set_xticklabels(ax.get_xticks(), size=8)
      #ax.set_yticklabels(ax.get_yticks(), size=8)
      return ax
 
def plot_Hist(fig, x, y, subx1, suby2, subsize, title, xlabel, ylabel, linestyle, errlow=[], errhigh=[], bound = False):
   
    ax = fig.add_axes([subx1, suby2, subsize, subsize]) 
    
    a = lage_arr
    b = lage2_arr
    ab = list(a) + list (b)
    abc = list(set(ab))
    abcd = sorted(abc)
    bins = np.array(abcd)
    
    widths = bins[1:] - bins[:-1]
    
    ax.step(bins[1:], sfr_arr, color='k')
    #ax.bar(bins[:-1], sfr_arr, width = widths, edgecolor = 'k', color = 'none', alpha = 0.4)
    ax.set_xlim(6.6, 10.15)
    ax.invert_xaxis()
    
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xticks(np.array([10, 9.5, 9, 8.5, 8, 7.5, 7])) 
    ax.set_xticklabels(ax.get_xticks(), size=8)
    ax.set_yticklabels(ax.get_yticks(), size=8)
    #ax.set_yticklabels(ax.get_yticks(), size=8)
   
    #This tells the error bars to be in the center of the graph, and also plots them
    bincenters = .5*(bins[1:] + bins[:-1])
    ax = fig.add_axes([subx1, suby2, subsize, subsize])
    ax.errorbar(bincenters, sfr_arr, yerr = [sfr_low, sfr_high], fmt = ' ', capsize = 0, color = 'r')
    
    return ax
    
def plot_LinHist(fig, x, y, subx1, suby2, subsize, title, xlabel, ylabel, linestyle, errlow=[], errhigh=[], bound = False):
   
    ax = fig.add_axes([subx1, suby2, subsize, subsize]) 
    
    a = age_arr
    b = age2_arr
    ab = list(a) + list (b)
    abc = list(set(ab))
    abcd = sorted(abc)
    bins = np.array(abcd)
    
    widths = bins[1:] - bins[:-1]
    
    ax.step(bins[1:], sfr_arr, color='k')
    #ax.bar(bins[:-1], sfr_arr, width = widths, edgecolor = 'k', color = 'w', alpha = 0.4)
    ax.set_xlim(0, 14.0)
    ax.invert_xaxis()
    
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xticks(np.array([10, 9.5, 9, 8.5, 8, 7.5, 7])) 
    ax.set_xticks(np.array([14, 12, 10, 8, 6, 4, 2, 0]))
    ax.set_xticklabels(ax.get_xticks(), size=8)
    ax.set_yticklabels(ax.get_yticks(), size=8)
    #ax.set_xticklabels(ax.get_xticks(), size=8)
    #ax.set_yticklabels(ax.get_yticks(), size=8)
   
    bincenters = .5*(bins[1:] + bins[:-1])
    ax = fig.add_axes([subx1, suby2, subsize, subsize])
    ax.errorbar(bincenters, sfr_arr, yerr = [sfr_low, sfr_high], fmt = ' ', capsize = 0, color = 'r')
    
    return ax
     
     

def plot_fill(fig, x, y, subx1, suby2, subsize, title, xlabel, ylabel,linestyle, errlow=[], errhigh=[]):
    ax = fig.add_axes([subx1, suby2, subsize, subsize])
    ax.plot(x, y,'k', linestyle = linestyle)
    ax.fill_between(x, errlow, errhigh, facecolor='k', alpha=0.5)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlim(max(x),min(x))
    #ax.set_xticklabels(ax.get_xticks(), size=8)
    #ax.set_yticklabels(ax.get_yticks(), size=8)
    return ax

cmdfile = open(cmdfile_name, 'r')
line = cmdfile.readline()
shape = [int(i) for i in cmdfile.readline().split()]
shape = shape[1:3]
line_col, line_mag = cmdfile.readline(), cmdfile.readline()
magbins, colbins = np.zeros(shape[0]), np.zeros(shape[1])
obs_arr, mod_arr = np.zeros(shape), np.zeros(shape)
res_arr, sig_arr = np.zeros(shape), np.zeros(shape)
for i in range(shape[0]):
    for j in range(shape[1]):
        line = cmdfile.readline().split()
        mag, col, obs, mod, res, sig = [float(n) for n in line[:6]]
        colbins[j] = col
        obs_arr[i, j], mod_arr[i, j] = obs, mod
        res_arr[i, j], sig_arr[i, j] = res, sig
    magbins[i] = mag
cmdfile.close()
cmdat = np.genfromtxt(cmdfile_name, unpack = True, skip_header = 4)
maxdepth = max(cmdat[0])
mindepth = min(cmdat[0])
maxshift = max(cmdat[1])
minshift = min(cmdat[1])

outfile = np.genfromtxt(outfile_name, unpack = True, skip_header = 1)
lage_arr=outfile[0]
lage_arr1=np.append(lage_arr, 10.15)
lage2_arr=outfile[1]
sfr_arr=outfile[3]*100
sfr_low=outfile[5]
sfr_high=outfile[4]
#csfr_arr=outfile[12]
#To_arr = outfile[5]
#Tf_arr = outfile[6]
met_arr = np.array(outfile[6])
for i,m in enumerate(met_arr):
    if m == 0:
        met_arr[i] = np.nan   
met_arr1 = np.append(met_arr, -1.9)  
met_up = outfile[7]
met_up1 = np.append(met_up, 0)
met_low = outfile[8]
met_low1 = np.append(met_low, 0)
csf_arr = outfile[12]
#csf_arr1 is just csf_arr with a zero appended to it, I wanted to make a plot work - 
# without messing up other plots
csf_arr1 = np.append(csf_arr, 0)
csf_up = outfile[13]
csf_up1 = np.append(csf_up, 0)
csf_low = outfile[14]
csf_low1 = np.append(csf_low, 0)
outfile =open(outfile_name, 'r')
csfr_arr = np.zeros(len(lage_arr))
line =outfile.readline()
totalSF = float(line.split()[1])
outfile.close()

age_arr = [10**(n)/1e9 for n in lage_arr]
age_arr1 = np.append(age_arr, 14)
age2_arr = [10**(n)/1e9 for n in lage2_arr]
#age2_arr1 = np.append(age2_arr, 14)

for i in range(len(csfr_arr)):
    if i==0:
        csfr_arr[i]=0.
    else:
        csfr_arr[i]=csfr_arr[i-1]-sfr_arr[i]*(10**(lage_arr[i])-10**(lage_arr[i]-0.05))
#csfr_arr = csfr_arr + totalSF
csfr_arr = [(n+totalSF) / totalSF for n in csfr_arr]



xlabel = ' - '.join(line_col.replace('WFC','F').split('-'))
ylabel = line_mag.replace('WFC','F')

#Here we are specifying the color map min and max values, basically it adjusts the scale
#If you want to change the plots to include more of the lower intensity 
fig = plt.figure(figsize=figsize)
obs_cmin = min(obs_arr.flatten())
obs_cmax = .5*max(obs_arr.flatten())
mod_cmin = min(mod_arr.flatten())
mod_cmax = .5*max(mod_arr.flatten())
sig_cmin = min(sig_arr.flatten())
sig_cmax = max(sig_arr.flatten())

extent = [colbins[0], colbins[-1], magbins[-1], magbins[0]]
ax_obs, cax_obs = plot_HessD(fig, obs_arr, subx1, suby3, subsize, extent, obs_cmin, obs_cmax, interpolation, '(a) Observed CMD', xlabel, ylabel,1)

ax_mod, cax_mod = plot_HessD(fig, mod_arr, subx1, suby2, subsize, extent, mod_cmin, mod_cmax, interpolation, '(d) Modeled CMD', xlabel, ylabel,1)

#ax_res, cax_res = plot_HessD(fig, res_arr, subx1, suby1, subsize, extent, interpolation, '(c) Residual (Obs. - Mod.)', xlabel, ylabel)

ax_sig, cax_sig = plot_HessD(fig, sig_arr, subx1, suby1, subsize, extent, sig_cmin, sig_cmax, interpolation, '(g) Residual Significance', xlabel, ylabel,0)

ax_2 = plot_Hist(fig, lage_arr, sfr_arr, subx2, suby2, subsize, '(e) Star Formation Rate', 'log(age)','SFR (10$^{-2}$ M$\odot$ yr$^{-1}$)','steps', sfr_low, sfr_high)

ax5 = plot_LinHist(fig, age_arr, sfr_arr, subx3, suby2, subsize, '(f) Star Formation Rate', 'age (Gyr)','SFR (10$^{-2}$ M$\odot$ yr$^{-1}$)','steps', sfr_low, sfr_high)

ax_1 = plot_Scatter(fig, lage_arr1, csf_arr1, subx2, suby3, subsize, '(b) Cumulative Star Formation', 'log(age)','Fraction of Stellar Mass','solid', csf_low1, csf_up1)

ax_4 = plot_Scatter(fig, age_arr1, csf_arr1, subx3, suby3, subsize, '(c) Cumulative Star Formation', 'age (Gyr)', 'Fraction of Stellar Mass','solid', csf_low1, csf_up1, bound = True)

ax_3 = plot_Scatter(fig, lage_arr1, met_arr1, subx2, suby1, subsize, '(h) Metallicity', 'log(age)','Z','solid', met_low1, met_up1, bound = False)

ax_6 = plot_Scatter(fig, age_arr1, met_arr1, subx3, suby1, subsize, '(i) Metallicity', 'age (Gyr)', 'Z','solid', met_low1, met_up1, bound = True)

#plt.show()
fig.savefig(plot_name, dpi=300, bbox_inches='tight')
