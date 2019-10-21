import sys
import sys
import os
import matplotlib

matplotlib.use('Agg')
import numpy as np
from chainconsumer import ChainConsumer
import pylab as mplot
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.gridspec as gridspec
import numpy as np
import scipy as sp
import scipy.interpolate as interpolate
import scipy.optimize as op
from numpy.linalg import inv
import general_funcs as gnf
import pdb

scale_fac = 1
def get_Ptheory_terms(bias_param, ktheory, wslin_yg_interp_array, wthetalin_gg_interp_array, stat_type, bins_array,
                      bias_fiducial_gg=None, scale_fac = scale_fac):
    # pdb.set_trace()
    Ptheory_all = []
    if stat_type == 'gg':
        nbins = len(bins_array)
        for j in range(nbins):
            b_j_gg = bias_param[j]
            bin_j = bins_array[j]
            val_bin_j = sp.interpolate.splev(ktheory[j], wthetalin_gg_interp_array[bin_j])
            # Ptheory_all.append((b_j_gg ** 2) * val_bin_j)
            if len(Ptheory_all) == 0:
                Ptheory_all = (b_j_gg ** 2) * val_bin_j
            else:
                Ptheory_all = np.hstack((Ptheory_all,(b_j_gg ** 2) * val_bin_j))

    elif stat_type == 'yg':
        nbins = len(bins_array)
        for j in range(nbins):
            b_j_yg = bias_param[j]
            bias_fiducial_j_gg = bias_fiducial_gg[j]
            bin_j = bins_array[j]
            val_bin_j = sp.interpolate.splev(ktheory[j], wslin_yg_interp_array[bin_j])
            # Ptheory_all.append((b_j_yg * bias_fiducial_j_gg) * val_bin_j)
            if len(Ptheory_all) == 0:
                Ptheory_all = (b_j_yg * bias_fiducial_j_gg) * scale_fac*val_bin_j
            else:
                Ptheory_all = np.hstack((Ptheory_all,(b_j_yg * bias_fiducial_j_gg) * scale_fac*val_bin_j))

    elif stat_type == 'gg_yg':
        nbins = len(bins_array)
        for j in range(nbins):
            b_j_gg = bias_param[j]
            bin_j = bins_array[j]
            val_bin_j = sp.interpolate.splev(ktheory[j], wthetalin_gg_interp_array[bin_j])
            # Ptheory_all.append((b_j_gg ** 2) * val_bin_j)
            if len(Ptheory_all) == 0:
                Ptheory_all = (b_j_gg ** 2) * val_bin_j
            else:
                Ptheory_all = np.hstack((Ptheory_all,(b_j_gg ** 2) * val_bin_j))

        for j in range(nbins):
            b_j_yg = bias_param[j + nbins]
            b_j_gg = bias_param[j]
            bin_j = bins_array[j]
            val_bin_j = sp.interpolate.splev(ktheory[j+nbins], wslin_yg_interp_array[bin_j])
            # Ptheory_all.append((b_j_yg * b_j_gg) * val_bin_j)
            Ptheory_all = np.hstack((Ptheory_all, (b_j_yg * b_j_gg) * scale_fac*val_bin_j))

    # # pdb.set_trace()
    # Ptheory_all = np.squeeze(np.array(Ptheory_all))
    # if len(Ptheory_all.shape) > 1:
    #     ntotal = (Ptheory_all.shape[0]) * (Ptheory_all.shape[1])
    #     Ptheory_all_f = Ptheory_all.reshape(ntotal, )
    # else:
    #     Ptheory_all_f = Ptheory_all
    return Ptheory_all


def get_is_within_prior(bias_param, bias_prior):
    if len(bias_prior) == 2 * len(bias_param):
        is_not_within_prior = 0
        for pj in xrange(len(bias_param)):
            param = bias_param[pj]
            param_prior_min, param_prior_max = bias_prior[2 * pj], bias_prior[2 * pj + 1]
            if (param > param_prior_min) & (param < param_prior_max):
                is_not_within_prior += 0
            else:
                is_not_within_prior += 1

        if is_not_within_prior == 0:
            is_within_prior = True
        else:
            is_within_prior = False

    else:
        print 'Put correct priors on all the parameters of chain'
        sys.exit(1)

    return is_within_prior


def lnprob_func(bias_param, bias_prior, Pk_obs_comp, incov_obs_comp, k_obs_comp, wslin_yg_interp_array,
                wthetalin_gg_interp_array,stat_type, bins_array, bias_fiducial_gg=None, scale_fac = scale_fac):
    is_within_prior = get_is_within_prior(bias_param, bias_prior)

    if is_within_prior:
        Pk_theory_comp = get_Ptheory_terms(bias_param, k_obs_comp, wslin_yg_interp_array, wthetalin_gg_interp_array,
                                           stat_type, bins_array, bias_fiducial_gg=bias_fiducial_gg, scale_fac = scale_fac)
        # pdb.set_trace()
        valf = -0.5 * np.dot(np.dot(np.transpose((Pk_obs_comp - Pk_theory_comp)), incov_obs_comp),
                             (Pk_obs_comp - Pk_theory_comp))
    else:
        valf = -np.inf

    # pdb.set_trace()
    return valf


def setuplnprob_func(kcomp_min, kcomp_max, k_obs_array, Pk_obs, cov_obs, stat_type, bins_array, cov_diag=False, no_cov_zbins = False):

    k_obs_comp_ll = []

    if stat_type == 'gg_yg':

        k_obs_gg_all = np.array([])
        k_obs_yg_all = np.array([])
        for j in range(len(bins_array)):
            if len(k_obs_gg_all) == 0:
                k_obs_gg_all = k_obs_array[j]
                k_obs_yg_all = k_obs_array[j+len(bins_array)]
            else:
                k_obs_gg_all = np.hstack((k_obs_gg_all,k_obs_array[j]))
                k_obs_yg_all = np.hstack((k_obs_yg_all,k_obs_array[j+len(bins_array)]))

        k_obs_all = np.hstack((k_obs_gg_all, k_obs_yg_all))


        for j in range(len(bins_array)):
            k_obs_j = k_obs_array[j]
            selection_j = np.where((k_obs_j >= kcomp_min) & (k_obs_j <= kcomp_max))[0]
            k_obs_comp_j = k_obs_j[selection_j]
            k_obs_comp_ll.append(k_obs_comp_j)


        for j in range(len(bins_array)):
            k_obs_j = k_obs_array[j+len(bins_array)]
            selection_j = np.where((k_obs_j >= kcomp_min) & (k_obs_j <= kcomp_max))[0]
            k_obs_comp_j = k_obs_j[selection_j]
            k_obs_comp_ll.append(k_obs_comp_j)



    if stat_type == 'gg':

        k_obs_gg_all = np.array([])
        for j in range(len(bins_array)):

            if len(k_obs_gg_all) == 0:
                k_obs_gg_all = k_obs_array[j]
            else:
                k_obs_gg_all = np.hstack((k_obs_gg_all, k_obs_array[j]))

        k_obs_all = k_obs_gg_all

        for j in range(len(bins_array)):
            k_obs_j = k_obs_array[j]
            selection_j = np.where((k_obs_j >= kcomp_min) & (k_obs_j <= kcomp_max))[0]
            k_obs_comp_j = k_obs_j[selection_j]
            k_obs_comp_ll.append(k_obs_comp_j)

    if stat_type == 'yg':

        k_obs_yg_all = np.array([])

        for j in range(len(bins_array)):
            if len(k_obs_yg_all) == 0:
                k_obs_yg_all = k_obs_array[j+len(bins_array)]
            else:
                k_obs_yg_all = np.hstack((k_obs_yg_all, k_obs_array[j+len(bins_array)]))

        k_obs_all = k_obs_yg_all

        for j in range(len(bins_array)):
            k_obs_j = k_obs_array[j+len(bins_array)]
            selection_j = np.where((k_obs_j >= kcomp_min) & (k_obs_j <= kcomp_max))[0]
            k_obs_comp_j = k_obs_j[selection_j]
            k_obs_comp_ll.append(k_obs_comp_j)

        # k_obs_all = k_obs_array[7]
        # k_obs_j = k_obs_array[7]
        # selection_j = np.where((k_obs_j >= kcomp_min) & (k_obs_j <= kcomp_max))[0]
        # k_obs_comp_j = k_obs_j[selection_j]
        # k_obs_comp_ll.append(k_obs_comp_j)

    selection = np.where((k_obs_all >= kcomp_min) & (k_obs_all <= kcomp_max))[0]
    # pdb.set_trace()
    cov_obs_comp = (cov_obs[:, selection])[selection, :]

    if no_cov_zbins:
        bins_n_array = np.arange(len(bins_array))
        cov_obs_comp_h = np.copy(cov_obs_comp)
        if stat_type == 'gg_yg':
            z1_0 = []
            for ji in range(len(bins_array)):
                if len(z1_0)==0:
                    z1_0 = bins_n_array[ji]*np.ones(len(k_obs_comp_ll[ji]))
                else:
                    z1_0 = np.hstack((z1_0,bins_n_array[ji]*np.ones(len(k_obs_comp_ll[ji]))))


            z1_1 = []
            for ji in range(len(bins_array)):
                if len(z1_1)==0:
                    z1_1 = bins_n_array[ji]*np.ones(len(k_obs_comp_ll[len(bins_array) + ji]))
                else:
                    z1_1 = np.hstack((z1_1,bins_n_array[ji]*np.ones(len(k_obs_comp_ll[len(bins_array) + ji]))))



            z1_mat_0 = np.tile(z1_0, (len(z1_0), 1)).transpose()
            z1_mat_1 = np.tile(z1_1, (len(z1_1), 1)).transpose()

            z1_mat_01 = np.tile(z1_0, (len(z1_1), 1)).transpose()
            z1_mat_10 = np.tile(z1_1, (len(z1_0), 1)).transpose()


            # z1_0 = np.repeat(np.arange(len(bins_array)), len(k_obs_comp_ll[0]))
            # z1_1 = np.repeat(np.arange(len(bins_array)), len(k_obs_comp_ll[len(bins_array)]))
            # z1_mat_0 = np.tile(z1_0, (len(bins_array) * len(k_obs_comp_ll[len(bins_array)]), 1)).transpose()
            # z1_mat_1 = np.tile(z1_1, (len(bins_array) * len(k_obs_comp_ll[0]), 1)).transpose()

            z1_mat_0c = -1*np.ones(z1_mat_0.shape)
            z1_mat_1c = -1*np.ones(z1_mat_1.shape)
            z1_mat2 = np.concatenate((z1_mat_0c, z1_mat_01), axis=1)
            z1_mat22 = np.concatenate((z1_mat_10,z1_mat_1c), axis=1)
            z1_matf = np.concatenate((z1_mat2, z1_mat22), axis=0)
            z2_matf = np.transpose(z1_matf)
            # pdb.set_trace()
            offdiag = np.where(z1_matf != z2_matf)
            cov_obs_comp_h[offdiag] = 0.0


        if stat_type == 'gg' or stat_type == 'yg':
            z1 = np.repeat(np.arange(len(bins_array)), len(k_obs_comp_ll[0]))
            z1_mat = np.tile(z1, (len(bins_array) * len(k_obs_comp_ll[0]), 1)).transpose()
            z2_mat = np.transpose(z1_mat)
            offdiag = np.where(z1_mat != z2_mat)
            cov_obs_comp_h[offdiag] = 0.0

        plt.figure()
        plt.imshow(np.log(np.abs(cov_obs_comp) + 10 ** -50))
        plt.savefig('/Users/shivam/Dropbox/Research/actxdes/results_dir/full_cov_yg.pdf', dpi=240)

        plt.figure()
        plt.imshow(np.log(np.abs(cov_obs_comp_h) + 10 ** -50))
        plt.savefig('/Users/shivam/Dropbox/Research/actxdes/results_dir/no_z_cov_yg.pdf', dpi=240)

        plt.figure()
        plt.imshow(z1_matf)
        plt.savefig('/Users/shivam/Dropbox/Research/actxdes/results_dir/z1matf.pdf', dpi=240)

        # pdb.set_trace()

        cov_obs_comp = cov_obs_comp_h

    if cov_diag:
        cov_obs_comp = np.diag(np.diag(cov_obs_comp))



    # fac = 1.e-7
    # fac = 2.5e-08
    fac = 0.
    cov_obs_comp = cov_obs_comp + fac*np.diag(np.diag(np.ones(cov_obs_comp.shape)))

    incov_obs_comp = inv(cov_obs_comp)

    k_obs_comp = k_obs_all[selection]

    Pk_obs_comp = Pk_obs[selection]

    pdb.set_trace()

    return Pk_obs_comp, k_obs_comp,k_obs_comp_ll, incov_obs_comp, cov_obs_comp
# 54

def get_sampler(nwalkers, ndim, bias_prior, Pk_obs_comp, incov_obs_comp, k_obs_comp, wslin_yg_interp_array,
                wthetalin_gg_interp_array,
                stat_type, bins_array, bias_fiducial_gg=None, scale_fac = scale_fac, nthreads=8):
    import emcee
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_func,
                                    args=[bias_prior, Pk_obs_comp, incov_obs_comp, k_obs_comp, wslin_yg_interp_array,
                                          wthetalin_gg_interp_array,
                                          stat_type, bins_array, bias_fiducial_gg, scale_fac], threads=nthreads)
    return sampler


def get_initial_state(nwalkers, bias_prior, Pk_obs_comp, incov_obs_comp, k_obs_comp, wslin_yg_interp_array,
                      wthetalin_gg_interp_array,stat_type, bins_array, bias_fiducial_gg=None, state_type='random', scale_fac = scale_fac):
    if state_type == 'random':
        p0_array = []
        for bj in range(0, (len(bias_prior) - 1), 2):
            rand_for_bj = np.random.rand(nwalkers)
            bj_new = bias_prior[bj] + rand_for_bj * (bias_prior[bj + 1] - bias_prior[bj])
            p0_array.append(bj_new)
        p0_array = np.vstack(p0_array)
        p0_final = p0_array.T
    else:
        nll = lambda *args: -lnprob_func(*args)
        ndim = len(bias_prior) / 2
        parambgbp = np.ones(ndim)

        # pdb.set_trace()

        result = op.minimize(nll, parambgbp, args=(
        bias_prior, Pk_obs_comp, incov_obs_comp, k_obs_comp, wslin_yg_interp_array, wthetalin_gg_interp_array,
        stat_type, bins_array, bias_fiducial_gg, scale_fac), method='Nelder-Mead')
        theta_hat = result["x"]

        print theta_hat

        p0_final = [theta_hat + 1e-6 * np.random.randn(ndim) for i in range(nwalkers)]

    return p0_final


def get_final_pos_mcmc(sampler, p0, nsteps, burn_steps, do_save_chains, filename=None, pos_type='most_like'):
    # sampler.run_mcmc(p0, burn_steps)
    # sampler.reset()
    if do_save_chains:
        f = open(filename, "w")
        f.close()

    width = 30
    most_like_pos = []
    most_like = np.infty
    for i, result in enumerate(sampler.sample(p0, iterations=nsteps)):
        n = int((width + 1) * float(i) / nsteps)
        sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
        if do_save_chains:
            position = result[0]
            neglnprob = np.array([-1 * result[1]]).T
            f = open(filename, 'ab')
            # pdb.set_trace()
            savemat = np.concatenate((position, neglnprob), axis=1)
            np.savetxt(f, savemat)
            f.close()
            ml_ind = np.argmin(neglnprob)
            if neglnprob[ml_ind] < most_like:
                most_like = neglnprob[ml_ind]
                most_like_pos = position[ml_ind, :]
                print most_like, most_like_pos

    print most_like, most_like_pos
    sys.stdout.write("\n")

    # sampler.run_mcmc(p0,nsteps)

    # pdb.set_trace()

    para_mcmc_chain = sampler.flatchain
    para_avg_mcmc = (np.sum(para_mcmc_chain, axis=0)) / (len(para_mcmc_chain))
    para_median_mcmc = np.median(para_mcmc_chain, axis=0)

    final_pos_sigma = np.std(para_mcmc_chain, axis=0)

    if pos_type == 'most_like':
        final_pos = most_like_pos
    elif pos_type == 'avg':
        final_pos = para_avg_mcmc
    elif pos_type == 'median':
        final_pos = para_median_mcmc
    else:
        print 'No predefined pos_type given'
        sys.exit(1)

    print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
    print('Mean: ', para_avg_mcmc, ' Median: ', para_median_mcmc, ' Most-Like: ', most_like_pos)

    return final_pos, final_pos_sigma, sampler


def get_chi2fit(para_avg_mcmc, bias_prior, Pk_obs_comp, incov_obs_comp, k_obs_comp, wslin_yg_interp_array,
                wthetalin_gg_interp_array,
                stat_type, bins_array, bias_fiducial_gg=None, scale_fac = scale_fac):
    chi2fit = (-2.) * lnprob_func(para_avg_mcmc, bias_prior, Pk_obs_comp, incov_obs_comp, k_obs_comp,
                                  wslin_yg_interp_array, wthetalin_gg_interp_array,
                                  stat_type, bins_array, bias_fiducial_gg=bias_fiducial_gg, scale_fac = scale_fac)
    return chi2fit


def get_final_fit_results(nwalkers, nsteps, burn_steps, bias_prior, kcomp_min, kcomp_max, k_obs_array, Pk_obs, cov_obs,
                          wslin_yg_interp_array, wthetalin_gg_interp_array, stat_type, bins_array, nthreads,
                          do_save_chains, param_name_latex, bins_to_fit,
                          chain_dir, pos_type, filename_suffix=None, bias_fiducial_gg=None, state_type='random',
                          cov_diag=False, scale_fac = None,no_cov_zbins = False):
    print('starting model: ', stat_type)
    ndim = len(bias_prior) / 2

    Pk_obs_comp, k_obs_comp,k_obs_comp_ll, incov_obs_comp,cov_obs_comp = setuplnprob_func(kcomp_min, kcomp_max, k_obs_array, Pk_obs, cov_obs, stat_type,
                                                               bins_array, cov_diag=cov_diag,no_cov_zbins = no_cov_zbins)

    # pdb.set_trace()

    p0 = get_initial_state(nwalkers, bias_prior, Pk_obs_comp, incov_obs_comp, k_obs_comp_ll, wslin_yg_interp_array,
                           wthetalin_gg_interp_array,
                           stat_type, bins_array, bias_fiducial_gg=bias_fiducial_gg, state_type=state_type, scale_fac = scale_fac)

    pdb.set_trace()
    chi2fit0 = get_chi2fit(p0[0], bias_prior, Pk_obs_comp, incov_obs_comp, k_obs_comp_ll, wslin_yg_interp_array,wthetalin_gg_interp_array,stat_type, bins_array, bias_fiducial_gg=bias_fiducial_gg, scale_fac = scale_fac)

    sampler = get_sampler(nwalkers, ndim, bias_prior, Pk_obs_comp, incov_obs_comp, k_obs_comp_ll, wslin_yg_interp_array,
                          wthetalin_gg_interp_array,
                          stat_type, bins_array, bias_fiducial_gg=bias_fiducial_gg, scale_fac = scale_fac, nthreads=nthreads)


    print chi2fit0

    if filename_suffix is not None:
        chains_filename = 'mcmc_chain_' + '_rmax_' + str(
            kcomp_max) + '_rmin_' + str(kcomp_min) + '_ns_' + str(nsteps) + '_nw_' + str(
            nwalkers) + '_model_' + stat_type + '_covdiag_' + str(cov_diag) + '_nocovz_' + str(no_cov_zbins)  + filename_suffix + '_bin_' + '_'.join(
            str(x) for x in bins_to_fit)
    else:
        chains_filename = 'mcmc_chain_' + '_rmax_' + str(
            kcomp_max) + '_rmin_' + str(kcomp_min) + '_ns_' + str(nsteps) + '_nw_' + str(
            nwalkers) + '_model_' + stat_type + '_covdiag_' + str(cov_diag) + '_nocovz_' + str(no_cov_zbins)  + '_bin_' + '_'.join(
            str(x) for x in bins_to_fit)

    chains_path = chain_dir + chains_filename + '.txt'
    f = open(chains_path, 'w')
    f.close()

    final_pos, final_pos_sigma, sampler = get_final_pos_mcmc(sampler, p0, nsteps, burn_steps, do_save_chains,
                                                             filename=chains_path,
                                                             pos_type=pos_type)
    # print final_pos

    # ind_comp = np.where((k_obs >= kcomp_min) & (k_obs <= kcomp_max))[0]
    # k_obs_comp = k_obs[ind_comp]

    burn_fac = 0.3
    data = (np.loadtxt(chains_path))
    burned_data = data[-int((1 - burn_fac) * len(data)):, :]

    if stat_type == 'gg_yg':
        data_params = burned_data[:, 0:2 * len(bins_array)]
    else:
        data_params = burned_data[:, 0:len(bins_array)]

    c = ChainConsumer()
    c.add_chain(data_params, parameters=param_name_latex)

    c.configure(shade=False, tick_font_size=18, label_font_size=20, linewidths=3, shade_alpha=[0.1, 0.25, 0.5, 0.65],
                sigma2d=False, kde=False, bar_shade=True, sigmas=[0, 1, 2], linestyles=["-", "--", ":", "-."])
    mplot.rc('font', weight='bold')

    summary_dict = c.get_summary()

    best_fit_params = []
    best_fit_params_sighigh = []
    best_fit_params_siglow = []

    for param_name in param_name_latex:
        # pdb.set_trace()
        best_fit_params.append(summary_dict[param_name][1])
        best_fit_params_sighigh.append(summary_dict[param_name][2] - summary_dict[param_name][1])
        best_fit_params_siglow.append(summary_dict[param_name][1] - summary_dict[param_name][0])

    # best_fit_params,best_fit_params_sighigh,best_fit_params_siglow = final_pos, final_pos_sigma, final_pos_sigma

    # pdb.set_trace()
    chi2fit = get_chi2fit(best_fit_params, bias_prior, Pk_obs_comp, incov_obs_comp, k_obs_comp_ll, wslin_yg_interp_array,
                          wthetalin_gg_interp_array, stat_type, bins_array, bias_fiducial_gg=bias_fiducial_gg, scale_fac = scale_fac)
    Pk_th = get_Ptheory_terms(best_fit_params, k_obs_comp_ll, wslin_yg_interp_array, wthetalin_gg_interp_array,
                              stat_type, bins_array, bias_fiducial_gg=bias_fiducial_gg, scale_fac = scale_fac)

    # if stat_type == 'gg_yg':
    #     k_obs_all_gg = np.tile(k_obs_gg, len(bins_array))
    #     k_obs_all_yg = np.tile(k_obs_yg, len(bins_array))
    #     k_obs_all = np.hstack((k_obs_all_gg,k_obs_all_yg))
    #
    # if stat_type == 'gg':
    #     k_obs_all = np.tile(k_obs_gg, len(bins_array))
    #
    # if stat_type == 'yg':
    #     k_obs_all = np.tile(k_obs_yg, len(bins_array))
    #
    #
    # selection = np.where((k_obs_all >= kcomp_min) & (k_obs_all <= kcomp_max))[0]

    dof = len(k_obs_comp) - ndim

    #     pdb.set_trace()

    chi2fit_per_dof = chi2fit * 1.0 / dof
    # Pk_kobs = gnf.interp_array(k_obs, karr, Pk)

    print 'done with model ', stat_type, ' ;chi2fit:', chi2fit, ' ;chi2fit per dof:', chi2fit_per_dof, ' ;best fit params:', best_fit_params, ' ;sigma ', best_fit_params_siglow
    # print sampler.acor.max()
    sampler.reset()

    return best_fit_params, best_fit_params_siglow,best_fit_params_sighigh, chi2fit, chi2fit_per_dof, Pk_obs_comp, Pk_th, k_obs_comp, k_obs_comp_ll, cov_obs_comp
