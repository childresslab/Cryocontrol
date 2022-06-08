import numpy as np
import lmfit as lm
from warnings import warn
import logging as log
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy.special import erfc
import corner
from pathlib import Path
from metrolopy import gummy
gummy.style = 'pm'
gummy.nsig = 1
## DEBUG
#log.basicConfig(format='%(levelname)s:%(message)s ', level=log.DEBUG)
## WARNING
log.basicConfig(format='%(levelname)s:%(message)s ', level=log.WARNING)

def subarray_idxs(values,threshold=5):
    idxs = [0]
    subsum=0
    for i, val in enumerate(values):
        if val < threshold:
            subsum += val
            if subsum >= threshold:
                idxs.append(i+1)
                subsum=0
            # Handle edge case of the final element not being part of a sum.
            if i == len(values)-1:
                idxs.pop(-1)
                idxs.append(i+1)
        else:
            subsum = 0
            idxs.append(i+1)
    return idxs

def rebin_counts(edges,counts,indices):
    rbd_counts = [sum(counts[idx:indices[i+1]]) for i,idx in enumerate(indices[:-1])]
    rbd_edges = [edges[idx] for idx in indices]

    return np.array(rbd_edges),np.array(rbd_counts)

def edges_to_centers(edges):
    mids = [(edges[i+1]+edge)/2 for i,edge in enumerate(edges[:-1])]
    return np.array(mids)

def import_pico(filename):
    with open(filename,'r') as f:
        channels = None
        ns_per_channel = None
        reached_counts = False  
        header_lines = 0
        while not reached_counts:
            line = f.readline()
            header_lines += 1
            if "#channels per curve" == line.strip():
                channels = int(f.readline().strip())
                header_lines += 1
            elif "#ns/channel" == line.strip():
                ns_per_channel = float(f.readline().strip())
                header_lines += 1
            elif "#counts" == line.strip():
                reached_counts = True
    log.debug(f"{channels = }, {ns_per_channel = }")
    counts = np.genfromtxt(filename,skip_header=header_lines,dtype=int)
    if len(counts) != channels:
        log.warning("Number of counts read doesn't match given number of channels.")
    edges = np.arange(0,channels+1,1) * ns_per_channel
    return np.array(edges),np.array(counts)

def split_rise(times,counts,lower_cut_ns=1):
    max_idx = np.argmax(counts)
    min_idx = np.argmin(np.abs(times[:max_idx]-(times[max_idx]-lower_cut_ns)))
    return min_idx,max_idx

def strip_zeros(times,counts):
    idx = np.argmax(np.flip(counts>0))+1
    return times[:-idx],counts[:-idx]

def fit_irf(times,counts,plot=False):
    # Dumb Fit
    model = lm.Model(single_gexp)
    max_idx = np.argmax(counts)
    max_val = counts[np.argmax(counts)]
    const_val = np.mean(counts[:1000])
    center_guess = times[max_idx]
    fwhm_guess = (center_guess-times[np.argmin(np.abs(counts-max_val/2))])*2
    sigma_guess = fwhm_guess/(2*np.sqrt(2*np.log(2)))
    amp_guess = (max_val-const_val) * sigma_guess * np.sqrt(2*np.pi)
    params = model.make_params(amp = np.abs(amp_guess),
                               center = np.abs(center_guess),
                               sigma = np.abs(sigma_guess),
                               lifetime = 0.5,
                               offset = const_val)
    uncert = np.sqrt(counts)
    uncert[np.where(uncert==0)] = 1
    results = model.fit(counts,params,x=times,weights=1/uncert)
    if plot:
        results.plot(show_init=True,data_kws={'zorder':3},fit_kws={'zorder':5},init_kws={'zorder':4})
        plt.show(block=False)
    cut_times,cut_counts = cut_before_after(times,counts,results.params['sigma'].value,
                                                         results.params['lifetime'].value)
    # Smart Fit
    if plot:
        plt.scatter(cut_times,cut_counts,alpha=0.5,s=2,color='red',zorder=7)
    poisson_fitter = lm.minimize(single_gexp_poisson_residual,
                                 results.params,method='leastsq',
                                 args=(cut_times,), kws={'data': cut_counts})
    return cut_times,cut_counts,poisson_fitter

def fit_decay_double(times,counts,lt_guess,irf_fit,plot=False):
    # Dumb Fit
    model = lm.Model(double_gexp,independent_vars='x')
    params = model.make_params()
    irf_params = irf_fit.params.valuesdict()
    max_val = np.max(counts)
    params['offset'].set(value=np.mean(counts[:1000]))
    params['sigma'].set(value=irf_params['sigma'],vary=True, min=0)
    params['center'].set(value=irf_params['center'])

    params['amp1'].set(value=max_val/2*lt_guess, min=0)
    params['lifetime1'].set(value = lt_guess, min=0.15,max=30)
    params['amp2'].set(value=max_val/2*irf_params['lifetime'], min=0)
    params['lifetime2'].set(value = irf_params['lifetime'],min=0.15,max=30)
    uncert = np.sqrt(counts)
    uncert[np.where(uncert==0)] = 1
    results = model.fit(counts,params,x=times,weights=1/uncert)
    if plot:
        results.plot(show_init=True,data_kws={'zorder':3},fit_kws={'zorder':5},init_kws={'zorder':4})
        plt.show(block=False)
    # Smart Fit
    cut_times,cut_counts = cut_before_after(times,counts,results.params['sigma'].value,
                                                         results.params['lifetime1'].value)
    if plot:
        plt.scatter(cut_times,cut_counts,alpha=0.5,s=2,color='red',zorder=7)
    poisson_fitter = lm.minimize(double_gexp_poisson_residual,
                                results.params,method='leastsq',
                                args=(cut_times,), kws={'data': cut_counts})
    return cut_times,cut_counts,poisson_fitter

def fit_decay_single(times,counts,lt_guess,irf_fit,plot=False):
    # Dumb Fit
    model = lm.Model(single_gexp,independent_vars='x')
    params = model.make_params()
    irf_params = irf_fit.params.valuesdict()
    max_val = np.max(counts)
    params['offset'].set(value=np.mean(counts[:1000]))
    params['sigma'].set(value=irf_params['sigma'],vary=True)
    params['center'].set(value=irf_params['center'])

    params['amp'].set(value=max_val/2*lt_guess)
    params['lifetime'].set(value = lt_guess, min=0.15,max=30)
    uncert = np.sqrt(counts)
    uncert[np.where(uncert==0)] = 1
    results = model.fit(counts,params,x=times,weights=1/uncert)
    if plot:
        results.plot(show_init=True,data_kws={'zorder':3},fit_kws={'zorder':5},init_kws={'zorder':4})
        plt.show(block=False)
    # Smart Fit
    cut_times,cut_counts = cut_before_after(times,counts,results.params['sigma'].value,
                                                         results.params['lifetime'].value)
    if plot:
        plt.scatter(cut_times,cut_counts,alpha=0.5,s=2,color='red',zorder=7)
    poisson_fitter = lm.minimize(single_gexp_poisson_residual,
                                results.params,method='leastsq',
                                args=(cut_times,), kws={'data': cut_counts})
    return cut_times,cut_counts,poisson_fitter

def fit_data(times,counts,irf_times=None,irf_counts=None,lt_guess=4.0,plot=False,double=True):
    # Get initial Gaussian Guesses
    if irf_counts is None:
        cutlow,cuthigh = split_rise(times,counts)
        irf_x = times[cutlow:cuthigh]
        irf_counts = counts[cutlow:cuthigh]
    else:
        irf_x = irf_times
    irf_cut_times,irf_cut_counts,irf_fit = fit_irf(irf_x,irf_counts,plot=plot)

    if double:
        cut_times,cut_counts,decay_fit = fit_decay_double(times,counts,lt_guess,irf_fit,plot=plot)
    else:
        cut_times,cut_counts,decay_fit = fit_decay_single(times,counts,lt_guess,irf_fit,plot=plot)

    return cut_times,cut_counts,irf_cut_times,irf_cut_counts, decay_fit, irf_fit

def cut_threshold(times,fit,threshold=5):
    new_times = np.linspace(times[0],times[-1],2*len(times))
    expected = fit.eval(fit.params,x=new_times)
    lower_cut = np.argmax(expected>threshold)
    upper_cut = np.argmin(expected[lower_cut:]>threshold) + lower_cut
    if upper_cut==0:
        upper_cut=-1
    lower_time = new_times[lower_cut]
    upper_time = new_times[upper_cut]

    lower_cut = np.argmin(np.abs(times-lower_time))
    upper_cut = np.argmin(np.abs(times-upper_time))
    return lower_cut,upper_cut

def cut_before_after(times,counts,sigma,decay,mult=5):
    peak_idx = np.argmax(counts)
    dt = np.mean(np.diff(times))
    pre_cut = max(int(round(peak_idx - 5*np.sqrt(2*np.log(2))*sigma*mult/dt)),0)
    post_cut = min(int(round(peak_idx + decay*mult/dt)),len(times))
    return times[pre_cut:post_cut], counts[pre_cut:post_cut]

def single_gexp(x,sigma,center,amp,lifetime,offset):
    gamma = 1/lifetime
    erf_comp = erfc((center + gamma*sigma**2 - x) / (np.sqrt(2)*sigma))
    exp_comp = np.exp(gamma*(center-x+gamma*sigma**2/2))
    return amp*gamma/2 * exp_comp*erf_comp + offset

def double_gexp(x,sigma,center,amp1,lifetime1,amp2,lifetime2,offset):
    return single_gexp(x,sigma,center,amp1,lifetime1,offset) + single_gexp(x,sigma,center,amp2,lifetime2,0)

def single_gexp_poisson_residual(params,x,data=None):
    pars = params.valuesdict()
    sigma = pars['sigma']
    center = pars['center']
    amp1 = pars['amp']
    lifetime1 = pars['lifetime']
    offset = pars['offset']

    model = single_gexp(x,sigma,center,amp1,lifetime1,offset)
    if data is None:
        return model
    model[np.where(model<=0)] = 1E-9
    return (model-data) / np.sqrt(model)

def double_gexp_poisson_residual(params,x,data=None):
    pars = params.valuesdict()
    sigma = pars['sigma']
    center = pars['center']
    amp1 = pars['amp1']
    lifetime1 = pars['lifetime1']
    amp2 = pars['amp2']
    lifetime2 = pars['lifetime2']
    offset = pars['offset']

    model = double_gexp(x,sigma,center,amp1,lifetime1,amp2,lifetime2,offset)
    if data is None:
        return model
    model[np.where(model<=0)] = 1E-9
    return (model-data) / np.sqrt(model)

def mcmc_fit(times,counts,decay_fit):
    params = decay_fit.params
    params['sigma'].set(vary=True)
    emcee_res = lm.minimize(double_gexp_poisson_residual,
                            decay_fit.params,method='emcee',
                            args=(times,), kws={'data': counts},
                            burn=300,steps=8000,thin=20)
    return emcee_res

if __name__ == "__main__":
    DO_EMCEE=False
    PLOT_INTERMEDIATE=False
    FIT_SINGLE = True
    FIT_DOUBLE = True
    filenames = []
    single_lifetimes = []
    #edges,counts = import_pico(r"X:\DiamondCloud\Cryostat setup\Data\2022-04-07_sample_fluro_scans\lifetime\big_bright_blob.dat")
    #edges,counts = import_pico(r"X:\DiamondCloud\Cryostat setup\Data\2022-04-07_sample_fluro_scans\lifetime\small_bright_blob2.dat")
    #edges,counts = import_pico(r"X:\DiamondCloud\Cryostat setup\Data\2022-04-07_sample_fluro_scans\lifetime\small_bright_blob.dat")
    #edges,counts = import_pico(r"X:\DiamondCloud\Cryostat setup\Data\2022-04-07_sample_fluro_scans\lifetime\faint_blob_long.dat")
    # filenames.append(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-02_cooldown\lifetime\bright_blob_2_post_powerout.dat").
    #edges,counts = import_pico(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-02_cooldown\lifetime\first_bright_blob.dat")
    #filename = r"X:\DiamondCloud\Cryostat setup\Data\2022-05-02_cooldown\lifetime\second_bright_blob.dat"
    # edges,counts = import_pico(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-02_cooldown\lifetime\first_bright_blob.dat")
    #edges,counts = import_pico(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-02_cooldown\lifetime\second_bright_blob_unsynced.dat")

    #Lateral Scan Data
    #filename = r"X:\DiamondCloud\Cryostat setup\Data\2022-05-02_cooldown\lifetime\Lateral Scan\-17p153_-14p874.dat"
    # filenames.append(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-02_cooldown\lifetime\Lateral Scan\-17p153_-14p524.dat")
    # filename = r"X:\DiamondCloud\Cryostat setup\Data\2022-05-02_cooldown\lifetime\Lateral Scan\-17p153_-14p167.dat"
    #filename = r"X:\DiamondCloud\Cryostat setup\Data\2022-05-02_cooldown\lifetime\Lateral Scan\-17p153_-13p810.dat"
    #filename = r"X:\DiamondCloud\Cryostat setup\Data\2022-05-02_cooldown\lifetime\Lateral Scan\-17p153_-13p453.dat"

    # Length Scan Data
    #filename = r"X:\DiamondCloud\Cryostat setup\Data\2022-05-02_cooldown\lifetime\Length Scan\111p1V.dat"
    #filename = r"X:\DiamondCloud\Cryostat setup\Data\2022-05-02_cooldown\lifetime\Length Scan\112p1V.dat"
    #filename = r"X:\DiamondCloud\Cryostat setup\Data\2022-05-02_cooldown\lifetime\Length Scan\113p1V.dat"
    #filename = r"X:\DiamondCloud\Cryostat setup\Data\2022-05-02_cooldown\lifetime\length_scan_blob_long.dat"
    # filename = r"X:\DiamondCloud\Cryostat setup\Data\2022-05-02_cooldown\lifetime\bright_blob_2_post_powerout.dat"
    # filename = r"X:\DiamondCloud\Cryostat setup\Data\2022-05-02_cooldown\lifetime\bright_blob_post_powerout_2.dat"
    #filenames.append(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-02_cooldown\lifetime\bb2_post\Vertical\87p3.dat")
    #filenames.append(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-02_cooldown\lifetime\bb2_post\Vertical\87p6.dat")
    #filenames.append(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-02_cooldown\lifetime\bb2_post\Vertical\87p9.dat")
    #filenames.append(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-02_cooldown\lifetime\bb2_post\Vertical\88p2.dat")
    #filenames.append(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-02_cooldown\lifetime\bb2_post\Vertical\88p5.dat")
    """
    filenames.append(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-02_cooldown\lifetime\b3 Lateral\-6p6227.dat")
    filenames.append(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-02_cooldown\lifetime\b3 Lateral\-6p3723.dat")
    filenames.append(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-02_cooldown\lifetime\b3 Lateral\-6p1223.dat")
    filenames.append(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-02_cooldown\lifetime\b3 Lateral\-5p8723.dat")
    filenames.append(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-02_cooldown\lifetime\b3 Lateral\-5p6223.dat")
    filenames.append(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-02_cooldown\lifetime\b3 Lateral\-4p0.dat")
    """
    #filenames.append(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-30_cooldown_cont\lifetime\lateral b1\initial_blob.dat")
    #filenames.append(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-30_cooldown_cont\lifetime\lateral b2\initial_blob_no_sync.dat")
    #filenames.append(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-30_cooldown_cont\lifetime\lateral b2\initial_blob_no_sync_faster.dat")
    #filenames.append(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-30_cooldown_cont\lifetime\lateral b2\initial_blob_sync_faster.dat")
    #filenames.append(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-30_cooldown_cont\lifetime\lateral b2\initial_blob_resync_faster.dat")
    """
    filenames.append(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-30_cooldown_cont\lifetime\lateral b3\centered.dat")
    filenames.append(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-30_cooldown_cont\lifetime\lateral b3\centered_galvo.dat")
    filenames.append(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-30_cooldown_cont\lifetime\lateral b3\-1Vy.dat")
    filenames.append(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-30_cooldown_cont\lifetime\lateral b3\-2Vy.dat")
    filenames.append(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-30_cooldown_cont\lifetime\lateral b3\-3Vy.dat")
    filenames.append(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-30_cooldown_cont\lifetime\lateral b3\-4Vy.dat")
    filenames.append(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-30_cooldown_cont\lifetime\lateral b3\-5Vy.dat")
    """
    filenames.append(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-30_cooldown_cont\lifetime\b4\peak_synced.dat")
    filenames.append(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-30_cooldown_cont\lifetime\b4\(-0p4,-0p4)_synced.dat")
    filenames.append(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-30_cooldown_cont\lifetime\b4\(-0p8,-0p8)_synced.dat")
    filenames.append(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-30_cooldown_cont\lifetime\b4\(-1p2,-1p2)_synced.dat")
    filenames.append(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-30_cooldown_cont\lifetime\b4\(-1p6,-1p6)_synced.dat")
    filenames.append(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-30_cooldown_cont\lifetime\b4\(-2p0,-2p0)_synced.dat")
    filenames.append(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-30_cooldown_cont\lifetime\b4\(-2p4,-2p4)_synced.dat")
    filenames.append(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-30_cooldown_cont\lifetime\b4\(-2p8,-2p8)_synced.dat")
    filenames.append(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-30_cooldown_cont\lifetime\b4\(-3p2,-3p2)_synced.dat")
    filenames.append(r"X:\DiamondCloud\Cryostat setup\Data\2022-05-30_cooldown_cont\lifetime\b4\(-3p6,-3p6)_synced.dat")

    

    for filename in filenames:
        print(filename)
        path = Path(filename)
        edges,counts = import_pico(path)
        irf_edges,irf_counts = import_pico(r"X:\DiamondCloud\Cryostat setup\Data\2022-04-07_sample_fluro_scans\lifetime\IRF_offset.dat")
        #edges,counts = import_pico(r"X:\DiamondCloud\Cryostat setup\Data\2021_09_03_pos2_short\bloblifetime2.dat")
        #edges,counts = import_pico(r"X:\DiamondCloud\Cryostat setup\Data\2021_09_03_pos2_short\bloblifetime3.dat")
        times = edges_to_centers(edges)
        # After changing offset
        irf_times = edges_to_centers(irf_edges) - 22

        times,counts = strip_zeros(times,counts)
        irf_times,irf_counts = strip_zeros(irf_times,irf_counts)

        if FIT_SINGLE:
            s_cut_times,s_cut_counts,irf_cut_times,irf_cut_counts,s_decay,irf = fit_data(times,counts,irf_times,irf_counts,plot=PLOT_INTERMEDIATE,double=False)
        if FIT_DOUBLE:
            if FIT_SINGLE:
                lt_guess = s_decay.params['lifetime'].value
            else:
                lt_guess = 4
            d_cut_times,d_cut_counts,irf_cut_times,irf_cut_counts,d_decay,irf = fit_data(times,counts,irf_times,irf_counts,lt_guess,plot=PLOT_INTERMEDIATE,double=True)
        print("IRF Fit Results:")
        irf.params.pretty_print()
        fig,axes = plt.subplots(2,1,sharex=True,gridspec_kw={"height_ratios" : [0.3,0.7]})
        legend_title = ""
        if FIT_SINGLE:
            print("Single Exp. Decay Fit Results:")
            s_decay.params.pretty_print()
            axes[1].errorbar(s_cut_times,s_cut_counts,np.sqrt(s_cut_counts),fmt='o',linestyle=None,ms=2,alpha=0.5,zorder=6,color="C0")
            pparams = s_decay.params.valuesdict()

            new_cut_times = np.linspace(s_cut_times[0],s_cut_times[-1],4*len(s_cut_times))
            axes[0].scatter(s_cut_times,s_decay.residual,s=2,zorder=3,color="C4",alpha=0.5)
            axes[0].fill_between(s_cut_times,s_decay.residual-1,s_decay.residual+1,alpha=0.5,color="C3")
            axes[0].axhline(0,0,1,linestyle='--',color='gray',alpha=0.5,zorder=4)
            func_eval = single_gexp(new_cut_times,
                                    pparams['sigma'],
                                    pparams['center'],
                                    pparams['amp'],
                                    pparams['lifetime'],
                                    pparams['offset'])
            axes[1].plot(new_cut_times,func_eval,zorder=7,label="Single Fit",color="C4")
            axes[1].set_yscale('log')
            tau = gummy(pparams['lifetime'],s_decay.params['lifetime'].stderr)
            chi2 = gummy(s_decay.redchi,np.sqrt(2/s_decay.nfree))
            legend_title += f"Single: tau = {tau} ns\n    chi2red = {chi2}"
            single_lifetimes.append(tau)
            if FIT_DOUBLE:
                legend_title += "\n"
        if FIT_DOUBLE:
            print("Double Exp. Decay Fit Results:")
            d_decay.params.pretty_print()
            axes[1].errorbar(d_cut_times,d_cut_counts,np.sqrt(d_cut_counts),fmt='o',linestyle=None,ms=2,alpha=0.5,zorder=5,color="C0")
            pparams = d_decay.params.valuesdict()

            new_cut_times = np.linspace(d_cut_times[0],d_cut_times[-1],4*len(d_cut_times))
            axes[0].scatter(d_cut_times,d_decay.residual,s=2,zorder=3,color="C2",alpha=0.5)
            axes[0].fill_between(d_cut_times,d_decay.residual-1,d_decay.residual+1,alpha=0.5,color="C1")
            axes[0].axhline(0,0,1,linestyle='--',color='gray',alpha=0.5,zorder=4)
            func_eval = double_gexp(new_cut_times,
                                    pparams['sigma'],
                                    pparams['center'],
                                    pparams['amp1'],
                                    pparams['lifetime1'],
                                    pparams['amp2'],
                                    pparams['lifetime2'],
                                    pparams['offset'])
            axes[1].plot(new_cut_times,func_eval,zorder=7,label="Double Fit",color="C2")
            axes[1].set_yscale('log')
            tau = gummy(pparams['lifetime1'],s_decay.params['lifetime'].stderr)
            chi2 = gummy(s_decay.redchi,np.sqrt(2/s_decay.nfree))
            legend_title += f"Double: tau = {tau} ns\n    chi2red = {chi2}"
        plt.tight_layout()
        axes[0].set_title(path.name)
        axes[1].legend(loc="upper right",
                    title=legend_title)
        axes[0].set_ylabel("Stud. Resid.")
        axes[1].set_xlabel("Time (ns)")
        axes[1].set_ylabel("Counts")
        fig,ax=plt.subplots()

        ax.plot(irf_cut_times,irf_cut_counts/np.max(irf_counts),alpha=0.5,label="IRF")
        if FIT_DOUBLE:
            ax.plot(d_cut_times,d_cut_counts/np.max(counts),alpha=0.5,label="Decay")
        elif FIT_SINGLE:
            ax.plot(s_cut_times,s_cut_counts/np.max(counts),alpha=0.5,label="Decay")
        ax.legend(loc='upper right', frameon=False).set_zorder(1)
        
        plt.show(block=False)


        #mcmc
        if DO_EMCEE:
            emcee_res = mcmc_fit(d_cut_times,d_cut_counts,d_decay)
            plt.figure()
            plt.plot(emcee_res.acceptance_fraction, 'o')
            plt.xlabel('walker')
            plt.ylabel('acceptance fraction')
            try:
                corner.corner(emcee_res.flatchain,labels=emcee_res.var_names,
                            truths=list(emcee_res.params.valuesdict().values()))
            except ValueError:
                pass
            print("Emcee Results:")
            emcee_res.params.pretty_print()
            print(f"chi2red = {emcee_res.redchi:.3f}±{np.sqrt(2/emcee_res.nfree):.1e}")
        plt.show(block=False)
    for f,t in zip(filenames,single_lifetimes):
        print(f"{f}: {t} ns")