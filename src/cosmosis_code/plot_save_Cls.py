import sys, os
from cosmosis.datablock import names, option_section
from numpy import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as intspline
import scipy.interpolate as interp
import scipy as sp
import ast
import pickle as pk
import copy
import pdb

def plot_Cls(block, sec_name,save_plot_dir):
    nbins = block['nz_lens', 'nbin']
    ell = block[sec_name, 'ell']
    plotf = ell * (ell + 1) / (2 * np.pi)

    zmin_array = [0.15,0.3,0.45,0.6,0.75]
    zmax_array = [0.3, 0.45, 0.6, 0.75, 0.9]

    # plot gg
    colors = ['r', 'b', 'k', 'orange', 'magenta', 'cyan', 'r', 'b', 'k', 'orange', 'magenta', 'cyan']
    # nbar_bins = [7.5*1e5/(2*np.pi),1.7*1e6/(2*np.pi),3.2*1e6/(2*np.pi),1.5*1e6/(2*np.pi),6.5*1e5/(2*np.pi)]
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches((10, 8))
    for j in range(nbins):
        binv = j+1
        cov_d = np.diag(block[sec_name, 'cov_total_gg_gg_bin_' + str(binv) + '_' + str(binv)])
        ax.errorbar((1.015**j)*ell,plotf*block[sec_name, 'data_Clgg_bin_' + str(binv) + '_' + str(binv)],plotf*np.sqrt(cov_d),marker='*',linestyle='',color=colors[j])
        ax.plot((1.015**j)*ell,plotf*(block[sec_name, 'theory_Clgg_bin_' + str(binv) + '_' + str(binv)]),linestyle='-',color=colors[j], label=str(zmin_array[j]) + '<z<' + str(zmax_array[j]))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(fontsize=18)
    ax.set_xlabel(r'$\ell$', fontsize=20)
    ax.set_ylabel(r'$\ell (\ell + 1) C_{\ell} / 2\pi$', fontsize=20)

    plt.title('Galaxy-Galaxy correlations', size=20)
    fig.savefig(save_plot_dir + 'Cl_gg_comp.png')

    # plot gy
    # fig, ax = plt.subplots(1, 1)
    # fig.set_size_inches((10, 8))
    # for j in range((nbins)):
    #     binv = j+1
    #     cov_d = np.diag(block[sec_name, 'cov_total_gy_gy_bin_' + str(binv) + '_' + str(binv)])
    #     ax.errorbar((1.05**j)*ell,plotf*block[sec_name, 'data_Clgy_bin_' + str(binv) + '_' + str(binv)],plotf*np.sqrt(cov_d),marker='*',linestyle='',color=colors[j])
    #     ax.plot((1.05**j)*ell,plotf*block[sec_name, 'theory_Clgy_bin_' + str(binv) + '_' + str(binv)],linestyle='-',color=colors[j], label=str(zmin_array[j]) + '<z<' + str(zmax_array[j]))
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    # ax.set_xlabel(r'$\ell$', fontsize=20)
    # ax.set_ylabel(r'$\ell (\ell + 1) C_{\ell} / 2\pi$', fontsize=20)
    # ax.legend(fontsize=18)
    # plt.title('Galaxy-y correlations', size=20)
    # fig.savefig(save_plot_dir + 'Cl_gy_comp.png')

    for j in range((nbins)):
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches((10, 8))
        binv = j+1
        cov_d = np.diag(block[sec_name, 'cov_total_gy_gy_bin_' + str(binv) + '_' + str(binv)])
        ax.errorbar((1.05**j)*ell,plotf*block[sec_name, 'data_Clgy_bin_' + str(binv) + '_' + str(binv)],plotf*np.sqrt(cov_d),marker='*',linestyle='',color=colors[j])
        ax.plot((1.05**j)*ell,plotf*block[sec_name, 'theory_Clgy_bin_' + str(binv) + '_' + str(binv)],linestyle='-',color=colors[j], label=str(zmin_array[j]) + '<z<' + str(zmax_array[j]))
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel(r'$\ell$', fontsize=20)
        ax.set_ylabel(r'$\ell (\ell + 1) C_{\ell} / 2\pi$', fontsize=20)
        ax.legend(fontsize=18)
        plt.title('Galaxy-y correlations', size=20)
        fig.savefig(save_plot_dir + 'Cl_gy_comp_' + str(binv) + '.png')

    ## plot yy
    # fig, ax = plt.subplots(1, 1)
    # fig.set_size_inches((10, 8))
    # cov_d = np.diag(block[sec_name, 'cov_total_yy_yy'])
    # ax.errorbar(ell,plotf*block[sec_name, 'data_Clyy'],plotf*np.sqrt(cov_d),marker='*',linestyle='',color=colors[j])
    # ax.plot(ell,plotf*block[sec_name, 'theory_Clyy' ],linestyle='-',color=colors[j])
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    # ax.set_xlabel(r'$\ell$', fontsize=20)
    # ax.set_ylabel(r'$\ell (\ell + 1) C_{\ell} / 2\pi$', fontsize=20)
    # plt.title('y-y correlations', size=20)
    # fig.savefig(save_plot_dir + 'Cl_yy_comp.png')

def setup(options):
    do_plot = options.get_bool(option_section, "do_plot", True)
    save_plot_dir = options.get_string(option_section, "save_plot_dir",
                                       '/home/shivam/Research/cosmosis/ACTxDESY3/src/plots/')
    sec_name = options.get_string(option_section, "sec_save_name",'get_cov')
    return do_plot, save_plot_dir, sec_name


def execute(block, config):
    do_plot, save_plot_dir, sec_name = config
    plot_Cls(block, sec_name,save_plot_dir)

    return 0

