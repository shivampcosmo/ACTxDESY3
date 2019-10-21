import sys
import os
import matplotlib
matplotlib.use('Agg')
sys.path.insert(0, '../../helper/')
sys.path.insert(0, '../')
import numpy as np
import copy
import pdb
import fisher_plotter as fp
import configparser
import ast
import pdb
from chainconsumer import ChainConsumer
from chainconsumer.kde import MegKDE
import matplotlib.colors
import cPickle as pickle
import scipy.interpolate as interp
import copy
import pylab as mplot
import copy
import matplotlib.pyplot as plt
import dill
import pickle as pk
from configobj import ConfigObj
from configparser import ConfigParser

latex_names = {'H0': r'$H_0$', 'Om0': r'$\Omega_{m}$', 'Ob0': r'$\Omega_{b}$', 'sigma8': r'$\sigma_8$', 'ns': r'$n_s$',
               'P0': r'$P_0$', 'alpha_p': r'$\alpha_p$', 'logMstar': r'$\log(M_{\ast})$', 'c500': r'$c_{500}$',
               'alpha': r'$\alpha$', 'beta': r'$\beta$', 'gamma': r'$\gamma$', 'P0-A_m': r'$P_0(A_m)$',
               'P0-alpha_m': r'$\alpha^{\rm high}_m$', 'P0-alpha_z': r'$P_0(\alpha_z)$', 'xc-A_m': r'$x_c(A_m)$',
               'xc-alpha_m': r'$x_c(\alpha_m)$', 'xc-alpha_z': r'$x_c(\alpha_z)$', 'beta-A_m': r'$\beta(A_m)$',
               'beta-alpha_m': r'$\beta(\alpha_m)$', 'beta-alpha_z': r'$\beta(\alpha_z)$',
               'alpha-A_m': r'$\alpha(A_m)$', 'alpha-alpha_m': r'$\alpha(\alpha_m)$',
               'alpha-alpha_z': r'$\alpha(\alpha_z)$', 'gamma-A_m': r'$\gamma(A_m)$',
               'gamma-alpha_m': r'$\gamma(\alpha_m)$', 'gamma-alpha_z': r'$\gamma(\alpha_z)$',
               'logM1': r'$\log (M_1)$', 'logM0': r'$\log (M_0)$',
               'alpha_g': r'$\alpha_g$', 'sig_logM': r'$\sigma_{\log M}$', 'eta_mb': r'$\eta_{mb}$',
               'cmis': r'$c_{mis}$', 'fmis': r'$f_{mis}$', 'hydro_mb': r'B', 'alpha_p_low': r'$\alpha_m^{\rm low}$',
               'alpha_p_high': r'$\alpha_m^{\rm mid}$'}



file_path = '/global/project/projectdirs/des/shivamp/actxdes/data_set/sz_forecasts/chain_sz_hy_1e14_1e15_allparams.txt'
# twopt_file = '/global/u1/s/spandey/cosmosis_exp/cosmosis/sz_forecasts/forecasts/output/results_7iuZ.pkl'
# d_list = (dill.load(open(twopt_file, 'rb')))
# fp_obj = fp.FisherPlotter(d_list)
# param_names =
param_vary_toplot = [r'$P_0(A_m)$',r'$\alpha^{\rm high}_m$', r'$\beta(A_m)$', r'$\alpha_m^{\rm low}$', r'$\alpha_m^{\rm mid}$', r'$\eta_{mb}$', r'$f_{mis}$']

colors = ['#0072b1', '#009d73', '#d45e00','#008856', '#E68FAC', '#0067A5', '#F99379', '#604E97', '#F6A600', '#B3446C', '#DCD300','#F2F3F4', '#882D17', '#8DB600', '#654522', '#E25822', '#2B3D26''#F3C300', '#875692', '#F38400','#222222', '#BE0032', '#A1CAF1','#C2B280', '#848482']
shade = False
tick_font_size = 18
label_font_size = 20
linewidths = 1.5
shade_alpha = 0.2
sigma2d = False
kde = False
bar_shade = True
sigmas = [0, 1, 2]
linestyles = "-"

    
burn_fac = 0.3

infile = open(file_path, 'r')
first_line = infile.readline()
data = np.loadtxt(file_path)

burned_data = data[-int((1 - burn_fac) * len(data)):, :-2]
print burned_data.shape
# pdb.set_trace()

c = ChainConsumer()
colors_toplot = []
c.add_chain(burned_data, parameters=param_vary_toplot)
colors_toplot.append('red')
c.configure(colors=colors_toplot, shade=shade,
            tick_font_size=tick_font_size, label_font_size=label_font_size,
            linewidths=linewidths,
            shade_alpha=shade_alpha,
            sigma2d=sigma2d, kde=kde, bar_shade=bar_shade, sigmas=sigmas, linestyles=linestyles,
            usetex=False,summary=False)
mplot.rc('font', weight='bold')


fig = c.plotter.plot()
# Resize fig for doco. You don't need this.
fig.set_size_inches(8.0 + fig.get_size_inches())

ax_list = fig.axes


mplot.tight_layout()

mplot.savefig('/global/u1/s/spandey/cosmosis_exp/cosmosis/sz_forecasts/forecasts/plots/test_1e13_1e14_allparams' + '.pdf', bbox_inches='tight')

