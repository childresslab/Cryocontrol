#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants as const
from scipy import interpolate as intp
from pathlib import Path 
import lmfit as lm
import os

plt.style.use(r"X:\DiamondCloud\Personal\Rigel\style_pub_new.mplstyle")

def nm_to_THz(wl):
    freqs = np.array([const.c/(w*1E-9)/1E12 if w != 0 else 0 for w in wl])
    return freqs

def THz_to_nm(freq):
    return const.c/(freq*1E12) / 1E-9

def fit_n_gaussians(xdata,ydata,error=None,
                    nmin=2,nmax=5,
                    center=None,amp=None,sigma=None,
                    lbound = 0, ubound = np.inf,
                    tol = 1,
                    fix_c1 = True):
    n = nmin
    chisquare = np.inf
    results = None
    if error is None:
        error = np.ones_like(xdata)
    weights = 1/error
    while n <= nmax:
        if center is None:
            acenter = np.zeros(n)
        elif isinstance(center,int) or isinstance(center,float):
            acenter = np.ones(n) * center
        else:
            acenter = np.array(center)[:n] 

        if amp is None:
            aamp = np.ones(n)
        elif isinstance(amp,int) or isinstance(amp,float):
            aamp = np.ones(n) * amp
        else:
            aamp = np.array(amp)[:n]

        if sigma is None:
            asigma = np.ones(n)
        elif isinstance(sigma,int) or isinstance(sigma,float):
            asigma = np.ones(n) * sigma
        else:
            asigma = np.array(sigma)[:n]


        model = sum([lm.models.GaussianModel(prefix=f'g{i}_') for i in range(2,n+1)],
                     lm.models.GaussianModel(prefix='g1_'))
        params = model.make_params()
        for i in range(1,n+1):
            if fix_c1 and i == 1:
                params[f"g{i}_center"].set(value = acenter[i-1],min=lbound,max=ubound,vary=False)
            else:
                params[f"g{i}_center"].set(value = acenter[i-1],min=lbound,max=ubound)
            params[f"g{i}_amplitude"].set(value = aamp[i-1]*asigma[i-1]*np.sqrt(2*np.pi)/np.pi,min=lbound,max=ubound)
            params[f"g{i}_sigma"].set(value = asigma[i-1],min=lbound,max=5)
        new_results = model.fit(ydata,params=params,weights=weights,x=xdata)
        new_chisquare = new_results.chisqr
        if chisquare - new_chisquare > tol:
            chisquare = new_chisquare
            results = new_results
            n += 1
        else:
            break
    #print(f"Returning fit with {n-1} gaussians, {chisquare = }.")
    return results, n

def fit_exactly_n_gaussians(xdata,ydata,error=None,n=1,
                            center=None,amp=None,sigma=None,
                            lbound = 0, ubound = np.inf,
                            fix_c1 = True):
    if error is None:
        error = np.ones_like(xdata)
    weights = 1/error
    if center is None:
        acenter = np.zeros(n)
    elif isinstance(center,int) or isinstance(center,float):
        acenter = np.ones(n) * center
    else:
        acenter = np.array(center)[:n] 

    if amp is None:
        aamp = np.ones(n)
    elif isinstance(amp,int) or isinstance(amp,float):
        aamp = np.ones(n) * amp
    else:
        aamp = np.array(amp)[:n]

    if sigma is None:
        asigma = np.ones(n)
    elif isinstance(sigma,int) or isinstance(sigma,float):
        asigma = np.ones(n) * sigma
    else:
        asigma = np.array(sigma)[:n]

        
    model = sum([lm.models.GaussianModel(prefix=f'g{i}_') for i in range(2,n+1)],
                    lm.models.GaussianModel(prefix='g1_'))
    params = model.make_params()
    for i in range(2,n+1):
        if fix_c1 and i == 1:
            params[f"g{i}_center"].set(value = acenter[i-1],min=lbound,max=ubound,vary=False)
        else:
            params[f"g{i}_center"].set(value = acenter[i-1],min=lbound,max=ubound)
        params[f"g{i}_amplitude"].set(value = aamp[i-1]*asigma[i-1]*np.sqrt(2*np.pi)/np.pi,min=lbound,max=ubound)
        params[f"g{i}_sigma"].set(value = asigma[i-1],min=lbound,max=5)
    return model.fit(ydata,params=params,weights=weights,x=xdata),n

class WLFitter():
    def __init__(self):
        self.ref = None
        self.settings = {'auto_gaussian' : True,
                         'scale' : 1,
                         'shift' : 0,
                         'ref_freq_cutoff' : 100,
                         'n_max' : 3,
                         'n_min' : 2,
                         'chi2_tol' : 0.01,
                         'init_centers' :[0,1,2,3,4,5],
                         'init_sigmas': [4,4,4,4,4,4],
                         'init_amps': [4,1,1/2,1/3,1/4,1/5],
                         'offset_cut': 20.0}
        self.reference = None
        self.wl = None
    
    def set_reference(self, wl, sig):
        signal = (sig-min(sig))/np.max(sig)
        fft = np.fft.ifftshift(np.fft.ifft(signal))
        ts = np.linspace(-len(fft)//2,len(fft)//2,len(fft))
        fft[np.where(np.abs(ts) > self.settings['ref_freq_cutoff'])] = 0
        
        self.wavelength = wl
        self.reference_spectrum = np.abs(np.fft.fft(np.fft.fftshift(fft)))
        self.nu = np.flip(const.c/(self.wavelength*1E-9))
        self.uniform_nu = np.linspace(min(self.nu),max(self.nu),len(self.nu))
        self.timeu = np.arange(len(self.uniform_nu)) / (len(self.uniform_nu) * np.mean(np.diff(self.uniform_nu)))
        self.lengthu =  (const.c * self.timeu[:len(self.timeu)//2])/2 * 1E6
        self.length_fine = np.linspace(min(self.lengthu),max(self.lengthu),len(self.nu)*2)
        
        self.calc_reference_gaussian()

    def calc_reference_gaussian(self):
        wl = self.wavelength
        signal = self.reference_spectrum
        model = lm.models.GaussianModel()
        params = model.make_params(center=wl[np.argmax(signal)],
                                    amplitude = 1,
                                    sigma = 50) 
        fit = model.fit(self.reference_spectrum,params,x=wl)
        params = fit.params
        if self.settings['auto_gaussian']:
            params['sigma'].value = params['sigma'].value * self.settings['scale']
            params['center'].value = params['center'].value + self.settings['shift']
            params['amplitude'].value = params['amplitude'].value * self.settings['scale']
        else:
            params['amplitude'].value = params['amplitude'].value/params['sigma'].value * self.settings['scale']
            params['sigma'].value = self.settings['scale']
            params['center'].value = self.settings['shift']
        self.reference_gaussian = fit.eval(x=wl,params=params)
        self.reference_gaussian /= np.max(self.reference_gaussian)
        self.ref_mul = self.reference_gaussian/self.reference_spectrum

    def fit_spectra(self,spectrum):
        data = np.array(spectrum)
        signal = (data-min(data))/np.max(data)
        despectrumed_signal_wl = signal * self.ref_mul
        despectrumed_signal_wl /= np.max(despectrumed_signal_wl)
        self.wind = despectrumed_signal_wl
        despectrumed_signal = np.flip(despectrumed_signal_wl)

        signal_intp = intp.interp1d(self.nu,despectrumed_signal,'linear')
        uniform_signal = signal_intp(self.uniform_nu)

        fftc = np.fft.ifft(uniform_signal,norm='ortho')
        fftu = np.abs(fftc)
        fftu = fftu[:len(fftu)//2]

        c_guess = self.lengthu[np.argmax(fftu[18:])+18]
        a_guess = fftu[np.argmax(fftu[18:])+18] * 4

        # Cut up to peak + offset
        max_length = c_guess + self.settings["offset_cut"]
        indices = np.where(self.lengthu <= max_length)
        fit_lengths = self.lengthu[indices]
        fit_fft = fftu[indices]
        if self.settings['n_max'] == self.settings['n_min']:
            fitu, n = fit_exactly_n_gaussians(fit_lengths,fit_fft, n = self.settings['n_min'],
                                              center= [c_guess + shift for shift in self.settings['init_centers']],
                                              amp =  [self.settings['init_amps'][0]] + [a_guess * scale for scale in self.settings['init_amps'][1:]],
                                              sigma = self.settings['init_sigmas'])
        else:
            fitu, n = fit_n_gaussians(fit_lengths,fit_fft,
                                      center= [c_guess + shift for shift in self.settings['init_centers']],
                                      amp =  [self.settings['init_amps'][0]] + [a_guess * scale for scale in self.settings['init_amps'][1:]],
                                      sigma = self.settings['init_sigmas'],
                                      nmax=self.settings['n_max'],
                                      nmin=self.settings['n_min'],
                                      tol = self.settings['chi2_tol'])
        self.fft = fftu
        self.fit = fitu
        self.fit_lengths = fit_lengths 

        zero_peak = fitu.eval_components(x=self.lengthu,prefix='g1_center')['g1_']
        self.components = fitu.eval_components(x=self.length_fine)
        self.zero_subtracted = self.fft-zero_peak

        #center = fitu.params['g2_center'].value
        #error = fitu.params['g2_center'].stderr
        #return center,error

if __name__ == "__main__":
    fitter = WLFitter()
    fitter.settings['auto_gaussian'] = False
    fitter.settings['scale'] = 15
    fitter.settings['shift'] = 823
    
    reference_file = r"X:\DiamondCloud\Cryostat setup\Data\2022-01-18-WL_scan_test\new_ref.csv"
    reference_data = np.genfromtxt(reference_file,delimiter=',',names=True)
    fitter.set_reference(reference_data['Wavelength'],reference_data['Count'])
    data_file = r"X:\DiamondCloud\Cryostat setup\Data\2022-01-18-WL_scan_test\post_tighten_back_10um.asc"
    data = np.genfromtxt(data_file,skip_header=28,delimiter=',').T
    fitter.fit_spectra(data[1])
    print(fitter.fit.best_values['g2_center'])
    

# %%
