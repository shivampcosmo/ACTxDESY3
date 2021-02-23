import numpy as np
from scipy.stats import norm
from scipy.linalg import sqrtm
from cosmosis.runtime.config import Inifile
from cosmosis.runtime.pipeline import LikelihoodPipeline
import dill
import bayesfast as bf
import time
import sys
import os
from distributed import Client, LocalCluster
os.chdir('/global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/ini')
os.environ['OMP_NUM_THREADS'] = '1'
n_core=64


def parse_param_files(ini,init_v):

    values_file = ini.get('pipeline', 'values')
    try:
        prior_file = ini.get('pipeline', 'priors')
    except:
        pass

    values_ini = Inifile(values_file)
    try:
        prior_ini = Inifile(prior_file)
    except:
        prior_ini = {}

    pkeys, vkeys = [], []
    init_values, para_range = [], []
    _prior_mu, _prior_sig = [], []
    print(values_ini.keys())
    for k in values_ini.keys():

        param = '{}--{}'.format(k[0][0], k[0][1])
        line = values_ini.get(k[0][0], k[0][1])
        values = line.split()
        if len(values) == 1:
            continue
        elif len(values) == 3:
            vkeys.append(param)
            if init_v is None:
                init_values.append(float(values[1]))
            else:
                init_values.append(init_v[param])
            para_range.append([float(values[0]), float(values[2])])
        else:
            raise("Cannot parse values file, too many quanties specified for {}".format(
                k[0][0], k[0][1]))

    for k in prior_ini.keys():

        param = '{}--{}'.format(k[0][0], k[0][1])
        line = prior_ini.get(k[0][0], k[0][1])
        values = line.split()

        if (len(values) == 1):
            continue
        elif len(values) == 3:
            if values[0] != 'gaussian':
                raise(
                    'Non-gaussian/tophat prior specified. This is not currently supported by BayesFast')

            pkeys.append(param)
            _prior_mu.append(float(values[1]))
            _prior_sig.append(float(values[2]))
        else:
            raise("Cannot parse priors file, too many quanties specified for {}".format(
                k[0][0], k[0][1]))

    init_mu = np.array(init_values)
    print('init_mu is ' , init_mu)
    para_range = np.array(para_range)

    _prior_mu = np.asarray(_prior_mu)
    _prior_sig = np.asarray(_prior_sig)

    pkeys = np.array(pkeys)
    vkeys = np.array(vkeys)

    # only keep parameters in priors that are actually specified in the values file
    pidx = np.in1d(pkeys, vkeys)
    pkeys = pkeys[pidx]
    _prior_mu = _prior_mu[pidx]
    _prior_sig = _prior_sig[pidx]

    init_sig = (para_range[:, 1] - para_range[:, 0]) / 1000
    print('parameters are', vkeys)
    idx_dict = dict(zip(vkeys, np.arange(len(vkeys))))
    import pdb; pdb.set_trace()
    print(idx_dict)

    nl_params = [p.split(',')[0] for p in ini.get(
        'bayesfast', 'nonlinear-params').split()]
    _nonlinear_indices = np.array([idx_dict[k] for k in nl_params])
    print('Non-linear Params are: ')
    print(nl_params)
    useIS = eval(ini.get('bayesfast', 'useIS'))
    n_IS = np.int(ini.get('bayesfast', 'n_IS'))

    return init_mu, para_range, _prior_mu, _prior_sig, pkeys, vkeys, init_sig,\
        _nonlinear_indices, idx_dict, useIS, n_IS, nl_params


def main(ini_string, fname,fnamer0,init_v=None,n_chain=6, n_iter=2500, n_warmup=1000):

    ini = Inifile(ini_string)

    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    init_mu, para_range, _prior_mu, _prior_sig, pkeys, vkeys, init_sig,\
        _nonlinear_indices, idx_dict, useIS, n_IS, nl_params = parse_param_files(ini,init_v)
    
    print('NL params')
    pipeline = LikelihoodPipeline(ini)
    sys.stdout = old_stdout

    start = pipeline.start_vector()
    print(dict(zip(vkeys,start)))
    results = pipeline.run_results(start)

    pkeys = pkeys[np.in1d(np.array(pkeys), vkeys)]
    _prior_indices = np.array([idx_dict[p] for p in pkeys])
    _flat_indices = np.setdiff1d(np.fromiter(
        idx_dict.values(), dtype=int), _prior_indices)

    if len(_prior_indices)>0:
        _prior_norm = (
            -0.5 * np.sum(np.log(2 * np.pi * _prior_sig**2)) - np.sum(np.log(
                norm.cdf(para_range[_prior_indices, 1], _prior_mu, _prior_sig) -
                norm.cdf(para_range[_prior_indices, 0], _prior_mu, _prior_sig))) -
            np.sum(np.log(para_range[_flat_indices, 1] -
                          para_range[_flat_indices, 0]))
        )
    else:
        print('Only flat priors')
        _prior_norm = (
            -0.5 * np.sum(np.log(2 * np.pi * _prior_sig**2))
            - np.sum(np.log(para_range[_flat_indices, 1] -
                          para_range[_flat_indices, 0])))

    _d = results.block['data_vector', '2pt_data']
    nData = _d.shape[0]

    _invC = results.block['data_vector', '2pt_inverse_covariance']
    _invC_r = np.real(sqrtm(_invC))

    _d_diag = _d @ _invC_r
    _norm = results.block['data_vector', '2pt_norm']

    try:
        extra_params = ini.get('pipeline', 'extra_output')
        if (not hasattr(extra_params, '__iter__')) | (type(extra_params) == str):
            extra_params = extra_params.split()
            ep = np.copy(extra_params)
            extra_params = [e.split('/') for e in extra_params]
        else:
            ep = np.copy(extra_params)
            extra_params = [e.split('/') for e in extra_params]            
    except:
            extra_params = []

    nExtraParams = len(extra_params)
    nData += nExtraParams
    print('Also building surrogate for extra parameters: {}'.format(extra_params))
    
    # datafile = os.environ['DATAFILE']
    # demodel = os.environ['DEMODEL']
    # scale_cuts = os.environ['SCALE_CUTS']
    # dataset = os.environ['DATASET']
    # run_name_str = os.environ['RUN_NAME_STR']
    # scale_cut_dir = os.environ['SCALE_CUT_DIR']

    nParams = len(vkeys)

    if len(_prior_indices)>0:

        def des_prior_f(x): # prior chisq + log prior 
            chi2 = -0.5 * np.sum(((x[_prior_indices] - _prior_mu) / _prior_sig)**2)
            return chi2 + _prior_norm

        def des_prior_j(x):  # prior gradient
            foo = np.zeros((1, nParams))
            foo[0, _prior_indices] = - \
                    (x[_prior_indices] - _prior_mu) / _prior_sig**2
            return foo
        
    else:
        def des_prior_f(x): # prior chisq + log prior chi2 = -0.5 *
            return _prior_norm

        def des_prior_j(x):  # prior gradient
            foo = np.zeros((1, nParams))
            return foo

    # def des_2pt_theory(x, _invC_r=_invC_r, datafile=datafile, demodel=demodel,
    #                    scale_cuts=scale_cuts, dataset=dataset, run_name_str=run_name_str,
    #                    ini_string=ini_string, scale_cut_dir=scale_cut_dir):
    def des_2pt_theory(x, _invC_r=_invC_r,ini_string=ini_string):

        # run DES pipeline to get data*invCov , theory *invCov
        try:
            import os
            import sys
            os.environ['OMP_NUM_THREADS'] = '1'
            from cosmosis.runtime.config import Inifile
            from cosmosis.runtime.pipeline import LikelihoodPipeline
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            ini = Inifile(ini_string)
            pipeline = LikelihoodPipeline(ini)
            sys.stdout = old_stdout
            res = pipeline.run_results(x)
            try:
                rres = res.block['data_vector', '2pt_theory'] @ _invC_r
                try:
                    extra_params = ini.get('pipeline', 'extra_output')
                    if (not hasattr(extra_params, '__iter__')) | (type(extra_params) == str):
                        extra_params = extra_params.split()
    #                    print(extra_params)
                        extra_params = [e.split('/') for e in extra_params]
                except:
                    extra_params = None
                if extra_params is not None:
                    try:
                        extra = [res.block[e[0], e[1]] for e in extra_params]
                    except:
                        extra = [1]
                        print(x)
                        pass
                else:
                    extra = [1]
                return np.hstack([rres, extra])
            except:
                print(x)
                return np.nan * np.ones(nData)
                pass

        except Exception as e:
            raise(e)
            print('Failed to run pipeline! Returning nans')
            return np.nan * np.ones(nData)

    def select_2pt(allout, nData=(nData-nExtraParams)):

        return allout[:nData]

    def select_extra(allout, nExtraParams=nExtraParams):

        return allout[-nExtraParams:]

    def chi2_f(m):  # lhood chisq, uses covariance now
        return np.atleast_1d(-0.5 * np.sum((m - _d_diag)**2) + _norm)

    def chi2_fj(m):  # lood chisq gradient
        return (np.atleast_1d(-0.5 * np.sum((m - _d_diag)**2) + _norm),
                -(m - _d_diag)[np.newaxis])

    def select_2pt_jac(allout, nData=(nData - nExtraParams), nExtraParams=nExtraParams):

        jac = np.diag(np.ones(nData + nExtraParams))
        jac[:,-nExtraParams:] = 0
        
        return allout[:nData], jac[:-nExtraParams,:]

    def des_post_f(like, x):  # like+prior
        return like + des_prior_f(x)

    def des_post_fj(like, x):  # like + prior and prior gradients
        return like + des_prior_f(x), np.concatenate(
            (np.ones((1, 1)), des_prior_j(x)), axis=-1)

    # parameters-> theory model
    m_0 = bf.Module(fun=des_2pt_theory, input_vars='x',
                    output_vars=['allout'])

    m_1 = bf.Module(fun=select_2pt, input_vars='allout',
                    fun_and_jac=select_2pt_jac,
                    output_vars=['m'])

    # theory model -> likelihood
    m_2 = bf.Module(fun=chi2_f, fun_and_jac=chi2_fj,
                    input_vars=['m'], output_vars='like')

    # likelihood and parameters -> log posterior
    m_3 = bf.Module(fun=des_post_f, fun_and_jac=des_post_fj,
                    input_vars=['like', 'x'], output_vars='logp')

    # stack modules to go from params -> log posterior
    d_0 = bf.Density(density_name='logp', module_list=[m_0, m_1, m_2, m_3],
                     input_vars='x', input_dims=nParams, input_scales=para_range,
                     hard_bounds=True)

    print('BF posterior density, cosmosis posterior: {}, {}'.format(
        d_0(start), results.post))
    print('These should match closely!!')

    print('Checking size of polynomial inputs. There are {0} model paramters and {1} data points'.format(
        nParams, nData))

    # approximate theory model with linear model
    s_0 = bf.modules.PolyModel('linear', input_size=nParams, output_size=nData, input_vars='x',
                               output_vars=['allout'], input_scales=para_range)

    # another linear model
    pc_0 = bf.modules.PolyConfig('linear')

    # nonlinear model - quadratic
    pc_1 = bf.modules.PolyConfig('quadratic', input_mask=_nonlinear_indices)
    pc_2 = bf.modules.PolyConfig('cubic-2', input_mask=_nonlinear_indices)

    # approximate theory model as linear + quadratic
    s_1 = bf.modules.PolyModel([pc_0, pc_1, pc_2],  input_size=nParams, output_size=nData, input_vars='x',
                               output_vars=['allout'], input_scales=para_range)


    # # nonlinear model - quadratic
    # pc_1 = bf.modules.PolyConfig('quadratic', input_mask=_nonlinear_indices)
    # pc_2 = bf.modules.PolyConfig('cubic-2', input_mask=_nonlinear_indices)

    # approximate theory model as linear + quadratic
    # s_1 = bf.modules.PolyModel([pc_0, pc_1],  input_size=nParams, output_size=nData, input_vars='x',
    #                            output_vars=['allout'], input_scales=para_range)

    # check if in bounds provided by ranges in values.ini
    def _in_bound(xx, bound):
        xxt = np.atleast_2d(xx).T
        return np.product([np.where(xi > bound[i, 0], True, False) *
                           np.where(xi < bound[i, 1], True, False) for i, xi in
                           enumerate(xxt)], axis=0).astype(bool)

    # surrogate model training points
#    x_0 = bf.utils.random.multivariate_normal(
#        init_mu, np.diag(init_sig**2), 200, random_state=100)
    x_0 = bf.utils.sobol.multivariate_normal(init_mu, np.diag(init_sig**2), 100)
    bf.utils.parallel.set_backend(n_core)
    # checking that points are inside bounds
    x_0 = x_0[_in_bound(x_0, para_range)]

    # setting up steps in recipe
    # linear model for optimizer

    sample_trace_0={"n_chain": n_chain,
                "n_iter": n_iter,
                "n_warmup": n_warmup}
    sample_trace_1={"n_chain": n_chain,
                "n_iter": n_iter,
                "n_warmup": n_warmup}
    opt_0 = bf.recipe.OptimizeStep(s_0, alpha_n=2,x_0=x_0,
                                   sample_trace=sample_trace_0)

    # linear+quadratic sample step
#    sam_0 = bf.recipe.SampleStep(s_1, alpha_n=2, reuse_samples=1,
#                                                sample_trace=sample_trace_0)
#    # linear+quadratic sample step
#    sam_1 = bf.recipe.SampleStep(s_1, alpha_n=2, reuse_samples=1,
#                                                sample_trace=sample_trace_1)

    # linear+quadratic sample step
    sam_0 = bf.recipe.SampleStep(s_1, alpha_n=2,alpha_min=0.75, reuse_samples=1,
                                                sample_trace=sample_trace_0,logp_cutoff=False)
    # linear+quadratic sample step
    sam_1 = bf.recipe.SampleStep(s_1, alpha_n=2,alpha_min=0.75, reuse_samples=1,
                                                sample_trace=sample_trace_1,logp_cutoff=False)

    pos_0 = bf.recipe.PostStep(n_is=n_IS, k_trunc=0.25, evidence_method='GBS')

    r_0 = bf.recipe.Recipe(density=d_0, optimize=opt_0,
                            sample=[sam_0,sam_1], post=pos_0)

    time.strftime('%H:%M%p %Z on %b %d, %Y')


    r_0.run()
    
    results = r_0.get()
    samples = results.samples
    weights = r_0.get().weights_trunc
    weights_ut = r_0.get().weights
    logp = np.atleast_2d(results.logp).T
    surrogate = r_0._density._surrogate_list[0]
    sigma8 = np.array([surrogate(samples[i,:])[0][-1:] for i in range(len(samples))])
    vkeys = list(vkeys)
    vkeys.extend(['COSMOLOGICAL_PARAMETERS--SIGMA_8','logp', 'weight'])
    chain = np.hstack([samples, sigma8, logp,  np.array([weights]).T])
    chain_ut = np.hstack([samples, sigma8, logp,  np.array([weights_ut]).T])
    np.savetxt(fname, chain, header=' '.join(vkeys))    
    dill.dump(r_0.get(),open(fnamer0,'wb'))
    return r_0, vkeys, chain, chain_ut

#     time.strftime('%H:%M%p %Z on %b %d, %Y')
#     results = r_0.get()
#     samples = results.samples

#     if useIS:
#         weights = r_0.get().weights_trunc
#         logp = np.atleast_2d(results.logp).T
#     else:
#         weights = np.ones((len(samples),1))
#         logp = np.ones((len(samples),1))

#     data = r_0.result.data
#     sampleresult = data.sample[-1]
#     surrogate = sampleresult.surrogate_list[0]
#     extras = np.array([surrogate(samples[i,:])[0][-nExtraParams:] for i in range(len(samples))])

#     vkeys = list(vkeys)
#     vkeys.extend(ep)

#     return samples, extras, np.atleast_2d(weights).T, vkeys,\
#         r_0, logp, np.atleast_2d(results.logq).T


if __name__ == '__main__':

    import os
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import scipy as sp
    import dill
    import sys, os
    from astropy.io import fits
    import scipy.interpolate as interpolate
    import pickle as pk
    import time
    import sys, os
    import os.path
    from os import path


    ini_file = '/global/cfs/cdirs/des/shivamp/nl_cosmosis/y3-3x2pt-methods/cosmosis/fiducial/params_bf.ini'  
    outdir =  'bf_new'
    demodel = 'lcdm'
    scale_cuts = 'scales_3x2pt_0.5_8_6_v0.4.ini'
    os.environ['OUTDIR'] = outdir


    ti = time.time()
    t0 = ti

    run_name = 'v0'
    # dataf = 'buzzard_mean_dv_18_sompz_bin_zs_true_true_zl_fixed_sn_no_jk_20xrand_NGcov.fits'
    # dataf = 'sim_buzzardlike_nlbias_dv.fits'
    dataf = 'v0.40_fiducial.fits'
    print(dataf)

    fname = '/global/cfs/cdirs/des/shivamp/nl_cosmosis/y3-3x2pt-methods/cosmosis/' +  outdir + '/3x2pt_linbias_' + scale_cuts + '_' + dataf + '_' + demodel + '_' + run_name +  '_wit8_cub2_nis4k.txt'  
    fnamer0 = '/global/cfs/cdirs/des/shivamp/nl_cosmosis/y3-3x2pt-methods/cosmosis/' +  outdir + '/3x2pt_linbias_' + scale_cuts + '_' + dataf + '_' + demodel + '_' + run_name +  '_r0get_wit8_cub2_nis4k.pkl'  

    os.environ['RUN_NAME_STR'] = run_name
    # os.environ['DATAFILE'] = 'buzzard_v2.0/' + dataf
    os.environ['DATAFILE'] = dataf
    os.environ['DEMODEL'] = demodel
    os.environ['SCALE_CUT_DIR'] = '3x2pt_cuts'
    os.environ['DATASET'] = '3x2pt'
    os.environ['SCALE_CUTS'] = scale_cuts
    r_0, vkeys, chain, chain2 = main(ini_file, fname, fnamer0)



