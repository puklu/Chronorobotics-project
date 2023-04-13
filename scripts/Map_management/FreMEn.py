import math
import matplotlib.pyplot as plt

import numpy as np
import finufft


class FreMEn:
    """
    fiNUFFT based frequency map enhancement
    the calculation of complex numbers is done by the functions from library finufft
    the rest is very similar to FreMen (the mean is subtracted from values at the start)
    we use type 3 non-uniform fast fourier transformation
    _build_frequencies() method is expanded compared to FreMEn, do not use alternative ways if you do not know, what you are doing :)
    """

    def __init__(self):
        self.gamma_0 = None
        self.phis = None
        self.alphas = None
        self.omegas = None
        self.freqs_step_type = None

    def fit(self, times, values, params={'no_freqs': 5, 'longest': 604800., 'shortest': 3600., 'freqs_step_type': 'base'}):
        """
        input: times ... numpy array of floats, vector of times of measurings expects large amount of data
               values ... numpy array of floats, vector of measured values, len(values) = len(times)
               no_freqs ... integer, number of periods that define the model longest float, length of the longest wanted
                            period in default units
               shortest ... float, length of the shortest wanted period in default units
               freqs_step_type ... string, define the way of frequency series generation possible: 'base', 'uniform', 'random'

        output: self, define all attributes
        objective: to build a model of the time series (times, values)
        """
        times = times.astype(float)
        values = values.astype(float)

        no_freqs = int(params['no_freqs'])
        longest = float(params['longest'])
        shortest = float(params['shortest'])
        self.freqs_step_type = params['freqs_step_type']

        # print('values: ' + str(values))
        self.gamma_0 = float(np.mean(values))
        self.phis, self.alphas, self.omegas = self._get_model(times, values, no_freqs, longest, shortest)
        return self

    def predict(self, pred_times, saturation=0.0, probability=True):
        """
        input: pred_times ... numpy array (or list) of numbers or number, times to predict value
               saturation ... float, small value limitting prediction into boundary of "probability" in FreMEn 2017: saturation=0.01
        output: prediction ... numpy array of floats, len(pred_times)=len(predictions)
        objective: to estimate value in specified time(s)
        """
        # print('pred_times: ' + str(pred_times))
        # print('self.gamma_0:' + str(self.gamma_0))
        if isinstance(pred_times, (float, int)):
            pred_times = np.array([float(pred_times)])
        # print('pred_times podruhe: ' + str(pred_times))
        if len(self.omegas):
            prediction = self.fremen_predict_numpy(pred_times, self.gamma_0, self.phis, self.alphas, self.omegas)
        else:
            prediction = np.ones_like(pred_times) * self.gamma_0
        if probability:
            prediction[prediction > 1. - saturation] = 1. - saturation
            prediction[prediction < 0. + saturation] = 0. + saturation
        return prediction

    def _get_model(self, times, values, no_freqs, longest, shortest, do_not_use=False):
        """
        input: times ... numpy array of floats, vector of times of measurings expects large amount of data
               values ... numpy array of floats, vector of measured values, len(values) = len(times)
               no_freqs ... integer, number of periods that define the model longest float, length of the longest wanted period in default units
               shortest float, legth of the shortest wanted period in default units
        output: phis ... numpy array, shifts of chosen periods
                alphas ... numpy array, amplitudes of chosen periods
                omegas ... numpy array, angular velocity of chosen periods
        objective: to estimate the most influencing parameters of time series
        """
        if no_freqs == 0:
            return np.array([]), np.array([]), np.array([])
        tested_omegas = self._build_frequencies(longest, shortest)
        if no_freqs == -1:
            no_freqs = len(tested_omegas)
        else:
            no_freqs = min(no_freqs, len(tested_omegas))

        phis, alphas, omegas = self._fit_nufft_3(x=times, y=values - self.gamma_0, c=tested_omegas, nFreqs=no_freqs)
        return phis, alphas, omegas

    def _build_frequencies(self, longest, shortest):
        """
        input: longest float, legth of the longest wanted period in default units
               shortest float, legth of the shortest wanted period in default units
        output: tested_omegas numpy array, sequence of expected angular velocities uses: np.arange()
        objective: to create a sequence of expected angular velocities
        """
        k = int(longest / shortest)
        if self.freqs_step_type == 'random':
            tested_omegas = 2. * np.pi / (np.float64(np.random.rand(k)) * (longest - shortest) + shortest)
        elif self.freqs_step_type == 'uniform_periods':
            tested_omegas = 2. * np.pi / (longest - (np.float64(np.arange(k)) * shortest))
        else:  # unknown or orthogonal (FreMEn default)
            tested_omegas = (2.0 * np.pi * np.float64(np.arange(k) + 1)) / float(longest)
        return tested_omegas

    def _fit_nufft_3(self, x, y, c, nFreqs):
        """    
        uses library: https://finufft.readthedocs.io/en/latest/python.html
        x ... numpy array, positions of measurements (eg time)
        y ... numpy array, values of wave (eg amplitudes)
        c ... numpy array, chosen frequencies 
        nFreqs ... int, number of dominant frequencies defining the model
        """

        y = np.array([complex(v, 0.) for v in y])
        N = x.shape[0]

        f = finufft.nufft1d3(x, y, c)  # , isign=1)

        # commented version should be similar to fremen_numpy.py
        # gammas = 2. * f / N
        # amplitudes = np.absolute(gammas)
        # phase = np.angle(gammas)

        # following version should be similar to c++ version (both versions are mathematically equal)
        amplitudes = 2. * np.absolute(f) / N
        phase = np.angle(f)

        dominantIdxs = amplitudes.argsort()[-nFreqs:][::-1]

        return phase[dominantIdxs], amplitudes[dominantIdxs], c[dominantIdxs]

    def fremen_predict_numpy(self, pred_times, average, phis, alphas, omegas):
        """
        """
        return average + np.sum(alphas * np.cos(omegas * pred_times.reshape(-1, 1) - phis), axis=1)


if __name__ == "__main__":
    times = np.arange(0,20, 1/100)

    times1 = times[14::-1]
    times2 = times[30:14:-1]
    new_times = np.concatenate((times1, times2), axis=0)

    values = np.sin(2*np.pi*1*times) + np.sin(2*np.pi*2*times)  # + 0.5*np.sin(2*np.pi*4*times)

    values1 = values[14::-1]
    values2 = values[30:14:-1]
    new_values = np.concatenate((values1, values2), axis=0)

    fre = FreMEn()
    fre.fit(times, values, params={'no_freqs': 5, 'longest' : 4, 'shortest' :0.25, 'freqs_step_type':'base'})

    amplitudes = fre.alphas
    omegas = fre.omegas
    phis = fre.phis

    print(amplitudes)
    print(omegas)
    print(phis)
    print(2*np.pi/omegas)

    y = fre.predict(times)

    plt.plot(times, values)
    plt.plot(times, y)
    plt.show()
