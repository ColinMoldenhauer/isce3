import math

from dataclasses import dataclass
import journal
import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import resample

import isce3
from nisar.workflows.focus import cosine_window


@dataclass(frozen=True)
class bandpass_meta_data:
    # slant range spacing
    rg_pxl_spacing: float
    # wavelength
    wavelength: float
    # sampling frequency
    rg_sample_freq: float
    # bandwidth
    rg_bandwidth: float
    # center frequency
    center_freq: float
    # slant range
    slant_range: 'method'
    # az sampling frequency/pulse repetition frequency
    prf: float
    # azimuth bandwidth
    az_bandwidth: float

    @classmethod
    def load_from_slc(cls, slc_product, freq):
        """Get meta data from SLC object.
        Parameters
        ----------
        slc_product : nisar.products.readers.SLC
            slc object
        freq : {'A', 'B'}
            frequency band
        Returns
        -------
        meta_data : bandpass_meta_data
            bandpass meta data object
        """
        rdr_grid = slc_product.getRadarGrid(freq)
        rg_sample_freq = \
            isce3.core.speed_of_light * 0.5 /\
            rdr_grid.range_pixel_spacing
        is_close = math.isclose(rg_sample_freq,
                                np.round(rg_sample_freq),
                                rel_tol=1e-8)
        if is_close:
            rg_sample_freq = np.round(rg_sample_freq)

        rg_bandwidth = \
            slc_product.getSwathMetadata(freq).processed_range_bandwidth
        center_frequency = \
            isce3.core.speed_of_light / rdr_grid.wavelength

        az_bandwidth = \
            slc_product.getSwathMetadata(freq).processed_azimuth_bandwidth

        return cls(rdr_grid.range_pixel_spacing, rdr_grid.wavelength,
                   rg_sample_freq, rg_bandwidth, center_frequency,
                   rdr_grid.slant_range, rdr_grid.prf, az_bandwidth)


def check_range_bandwidth_overlap(ref_slc, sec_slc, pols):
    """Check if bandpass is needed.

    If the two SLCs differ in center frequency or bandwidth, then
    one SLC shall be bandpassed to a common frequency band. If
    necessary, determine which SLC will be bandpassed

    Parameters
    ----------
    ref_slc : nisar.products.readers.SLC
        Reference SLC object
    sec_slc : nisar.products.readers.SLC
        Secondary SLC object
    pols : dict
        Dict keying frequency ('A' or 'B') to list of polarization values.

    Returns
    -------
    mode : dict
        Dict mapping frequency band (e.g. "A" or "B") to
        SLC to be bandpassed ("ref" or "sec").
    """
    mode = dict()

    for freq, pol_list in pols.items():
        ref_meta_data = bandpass_meta_data.load_from_slc(ref_slc, freq)
        sec_meta_data = bandpass_meta_data.load_from_slc(sec_slc, freq)

        ref_wvl = ref_meta_data.wavelength
        sec_wvl = sec_meta_data.wavelength
        ref_bw = ref_meta_data.rg_bandwidth
        sec_bw = sec_meta_data.rg_bandwidth

        # check if two SLCs have same bandwidth and center frequency
        if (ref_wvl != sec_wvl) or (ref_bw != sec_bw):
            if ref_bw > sec_bw:
                mode[freq] = 'ref'
            else:
                mode[freq] = 'sec'
    return mode


class SplitSpectrum:
    '''
    Split the spectrum in slant range or azimuth direction.
    '''

    def __init__(self,
                 sample_freq,
                 bandwidth,
                 center_frequency,
                 slant_range,
                 freq,
                 axis,
                 sampling_bandwidth_ratio=None):
        """Initialized Bandpass Class with SLC meta data

        Parameters
        ----------
        sample_freq : float
            range sampling frequency
        bandwidth : float
            range bandwidth [Hz]
        center_frequency : float
            center frequency of SLC [Hz]
        slant_range : new center frequency for bandpass [Hz]    # TODO: is this docstring correct?
        freq : {'A', 'B'}
            frequency band
        axis : {'rg', 'az'}
            Whether to split the spectrum in range or azimuth direction
        sampling_bandwidth_ratio: float
            The ratio of range sampling frequency to bandwidth.
            If not provided, sampling frequency will be same as
            input.
        """
        self.freq = freq
        self.sample_freq = sample_freq
        self.pxl_spacing =
	     isce3.core.speed_of_light * 0.5 / self.sample_freq
        self.bandwidth = bandwidth
        self.center_frequency = center_frequency

        self.axis = axis

        self.slant_range = slant_range
        if sampling_bandwidth_ratio is None:
            sampling_bandwidth_ratio = sample_freq / bandwidth
        self.sampling_bandwidth_ratio = sampling_bandwidth_ratio

    @property
    def _fft_axis(self):
        """Axis index along which to perform fft."""
        return 1 if self.axis == "rg" else 0

    def _broadcast(self, array):
        """
        Expand arrays, such like bandpass windows,
        according to the splitting axis, such that
        numpy broadcasting rules are obeyed.

        Parameters
        ----------
        array : np.ndarray
            Array to be reshaped to proper dimensions

        Returns
        -------
        np.ndarray
            Reshaped array
        """
        return np.expand_dims(array, axis=0 if self.axis=="rg" else 1)      # opposite axis for expansion than for fft


    def bandpass_shift_spectrum(self,
                                slc_raster,
                                low_frequency,
                                high_frequency,
                                new_center_frequency,
                                window_function,
                                window_shape=0.25,
                                fft_size=None,
                                resampling=True,
                                doppler_centroid=None,
                                compensate_az_antenna_pattern=True,
                                ):

        """Bandpass SLC for a given bandwidth and shift the bandpassed
        spectrum to a new center frequency

        Parameters
        ----------
        slc_raster : numpy.ndarray
            numpy array of slc raster,
        low_frequency : float
            low  frequency of band to be passed [Hz]
        high_frequency : float
            high frequency band to be passed [Hz]
        new_center_frequency : float
            new center frequency for new bandpassed slc [Hz]
        window_function : str
            window type {tukey, kaiser, cosine}
        window_shape : float
            parameter for the raised cosine filter (e.g. 0 ~ 1)
        fft_size : int
            fft size.
        resampling : bool
            if True, then resample SLC and meta data with new range spacing
            If False, return SLC and meta with original range spacing

        Returns
        -------
        resampled_slc or slc_demodulate: numpy.ndarray
            numpy array of bandpassed slc
            if resampling is True,
            return resampled slc with bandpass and demodulation
            if resampling is False,
            return slc with bandpass and demodulation without resampling
        meta : dict
            dict containing meta data of bandpassed slc
            center_frequency, bandwidth, range_spacing, slant_range
        """

        sample_freq = self.sample_freq
        diff_frequency = self.center_frequency - new_center_frequency
        height, width = slc_raster.shape
        slc_raster = np.asanyarray(slc_raster, dtype='complex')

        slc_bp = self.bandpass_spectrum(
                          slc_raster=slc_raster,
                          low_frequency=low_frequency,
                          high_frequency=high_frequency,
                          window_function=window_function,
                          window_shape=window_shape,
                          remove_window=self.axis == "rg",      # TODO: make parameter for bandpass_shift_spectrum/SplitSpectrum or leave hardcoded?
                          fft_size=fft_size,
                          doppler_centroid=doppler_centroid,
                          compensate_az_antenna_pattern=compensate_az_antenna_pattern
                          )

        # demodulate the SLC to be baseband to new center frequency
        # if fft_size > width, then crop the spectrum from 0 to width
        if self.axis == "rg":
            slc_demodulate = self.demodulate_slc(slc_bp[:, :width],
                                                diff_frequency,
                                                sample_freq)
        else:
            slc_demodulate = self.demodulate_slc(slc_bp[:height, :],
                                                diff_frequency,
                                                sample_freq)

        # update metadata with new parameters
        meta = dict()
        new_bandwidth = high_frequency - low_frequency
        new_sample_freq = np.abs(new_bandwidth) * \
            self.sampling_bandwidth_ratio

        if self.axis == "rg":
            meta['center_frequency'] = new_center_frequency
            meta['bandwidth'] = new_bandwidth
            meta['sample_freq'] = new_sample_freq
        else:
            meta['az_bandwidth'] = new_bandwidth
            # TODO

        # Resampling changes the spacing and slant range
        spacing_key = "range_spacing" if self.axis == "rg" else "azimuth_spacing"
        if resampling:
            # due to the precision of the floating point, the resampling
            # scaling factor may be not integer.
            resampling_scale_factor = sample_freq / new_sample_freq

            # convert to integer
            if sample_freq % new_sample_freq == 0:
                resampling_scale_factor = np.round(resampling_scale_factor)
            else:
                err_msg = 'Resampling scaling factor ' \
                          f'{resampling_scale_factor} must be an integer.'
                raise ValueError(err_msg)

            if self.axis == "rg":
                sub_dim = int(width / resampling_scale_factor)
                x_cand = np.arange(1, width + 1)
            else:
                sub_dim = int(height / resampling_scale_factor)
                x_cand = np.arange(1, height + 1)

            # find the maximum of the multiple of resampling_scale_factor
            resample_end = np.max(x_cand[x_cand %
                                               resampling_scale_factor == 0])

            # resample SLC
            if self.axis == "rg":
                resampled_slc = resample(
                    slc_demodulate[:, :resample_end], sub_dim, axis=1)
            else:
                resampled_slc = resample(
                    slc_demodulate[:resample_end, :], sub_dim, axis=0)

            meta[spacing_key] = \
                self.pxl_spacing * resampling_scale_factor

            # TODO: whats the function of the slant_range method? is there an azimuth equivalent?
            if self.axis == "rg":
                meta['slant_range'] = \
                    self.slant_range(0) + \
                    np.arange(sub_dim) * meta['range_spacing']
            else:
                pass	# TODO

            return resampled_slc, meta

        else:
            meta[spacing_key] = self.pxl_spacing
            # TODO: whats the function of the slant_range method? is there an azimuth equivalent?
            if self.axis == "rg":
                meta['slant_range'] = \
                    self.slant_range(0) + \
                    np.arange(width) * meta['range_spacing']
            else:
                pass

            return slc_demodulate, meta

    def bandpass_spectrum(self,
                          slc_raster,
                          low_frequency,
                          high_frequency,
                          window_function,
                          window_shape=0.25,
                          remove_window=True,
                          fft_size=None,
                          doppler_centroid=None,
                          compensate_az_antenna_pattern=True
                          ):
        """Bandpass SLC for given center frequency and bandwidth

        Parameters
        ----------
        slc_raster : numpy.ndarray
            numpy array of slc raster,
        low_frequency : float
            low  frequency of band to be passed [Hz]
        high_frequency : float
            high frequency band to be passed [Hz]
        window_function : str
            window type {'tukey', 'kaiser', 'cosine'}
        window_shape : float
            parameter for the window shape
            kaiser 0<= window_shape < inf
            tukey and cosine 0 <= window_shape <= 1
        remove_window : bool
            whether to remove a windowing effect in the input spectrum
        fft_size : int
            fft size.
        doppler_centroid : None | numpy.ndarray
            optional array of doppler centroid values per range bin. If None, the spectrum is used as is, else
             will be used to center the respective azimuth spectrum around zero doppler
        compensate_az_antenna_pattern : bool
            whether to account for the azimuth spectrum asymmetry caused by the antenna pattern.

        Returns
        -------
        slc_bandpassed : numpy.ndarray
            numpy array of bandpassed slc
        """
        error_channel = journal.error('splitspectrum.bandpass_spectrum')

        sample_freq = self.sample_freq
        bandwidth = self.bandwidth
        center_frequency = self.center_frequency
        height, width = slc_raster.shape
        slc_raster = np.asanyarray(slc_raster, dtype='complex')
        new_bandwidth = high_frequency - low_frequency
        new_sample_freq = self.sampling_bandwidth_ratio * new_bandwidth
        resampling_scale_factor = sample_freq / new_sample_freq

        if new_bandwidth < 0:
            err_str = "Low frequency is higher than high frequency"
            error_channel.log(err_str)
            raise ValueError(err_str)

        dim = width if self.axis == "rg" else height
        if fft_size is None:
            fft_size = dim

        if fft_size < dim:
            err_str = f"FFT size is smaller than number of {self.axis} bins"
            error_channel.log(err_str)
            raise ValueError(err_str)

        # construct window to be deconvolved
        # from the original SLC in freq domain
        window_target = self.get_bandpass_window(
            center_frequency=0,
            freq_low=-bandwidth/2,
            freq_high=bandwidth/2,
            sampling_frequency=sample_freq,
            fft_size=fft_size,
            window_function=window_function,
            window_shape=window_shape
            )
        # construct window to bandpass spectrum
        # for given low and high frequencies
        window_bandpass = self.get_bandpass_window(
            center_frequency=0,
            freq_low=low_frequency - center_frequency,
            freq_high=high_frequency - center_frequency,
            sampling_frequency=sample_freq,
            fft_size=fft_size,
            window_function=window_function,
            window_shape=window_shape
            )

        # optionally shift azimuth spectrum to zero doppler
        if self.axis == "az" and doppler_centroid is not None:
            slc_raster = self.shift_doppler_centroid(
                slc_raster,
                doppler_centroid,
                sample_freq
            )

        # get spectrum
        spectrum_target = fft(slc_raster, n=fft_size, workers=-1, axis=self._fft_axis)

        # optionally correct for azimuth antenna pattern
        if self.axis == "az" and compensate_az_antenna_pattern:
            spectrum_target = self.compensate_azimuth_antenna_pattern(
                spectrum_target,
                center_frequency,
                sample_freq,
                fft_size)  # TODO: implement

        # remove the windowing effect from the spectrum
        if remove_window:
            window_target_bc = self._broadcast(window_target)
            spectrum_target = np.divide(spectrum_target,
                                        window_target_bc,
                                        out=np.zeros_like(spectrum_target),
                                        where=window_target_bc != 0)

        # apply new bandpass window to spectrum
        slc_bandpassed = ifft(spectrum_target
                              * self._broadcast(window_bandpass)
                              * np.sqrt(resampling_scale_factor),
                              n=fft_size,
                              workers=-1, axis=self._fft_axis)

        return slc_bandpassed

    def demodulate_slc(self, slc_array, diff_frequency, sample_freq):
        """ Demodulate SLC

        If diff_frequency is not zero, then the spectrum of SLC is shifted
        so that the sub-band slc is demodulated to center the sub band spectrum

        Parameters
        ----------
        slc_array : numpy.ndarray
            SLC raster or block of SLC raster
        diff_frequency : float
            difference between original and new center frequency [Hz]
        sample_freq : float
            range sampling frequency [Hz]

        Returns
        -------
        slc_baseband  : numpy.ndarray
            demodulated SLC
        """
        height, width = slc_array.shape

        if self.axis == "rg":
            time = self._broadcast(np.arange(width) / sample_freq)
        else:
            time = self._broadcast(np.arange(height) / sample_freq)

        slc_shifted = slc_array * np.exp(1j * 2.0 * np.pi
					  * diff_frequency * time)
        return slc_shifted

    def freq_spectrum(self, cfrequency, dt, fft_size):
        ''' Return Discrete Fourier Transform sample frequencies
        with center frequency bias.

        Parameters:
        ----------
        cfrequency : float
            Center frequency (Hz)
        dt : float
            Sample spacing.
        fft_size : int
            Window length.
        Returns:
        -------
        freq : ndarray
            Array of length fft_size containing sample frequencies.
        '''
        freq = cfrequency + fftfreq(fft_size, dt)
        return freq

    def get_bandpass_window(self,
                            center_frequency,
                            sampling_frequency,
                            fft_size,
                            freq_low,
                            freq_high,
                            window_function='tukey',
                            window_shape=0.25):
        '''Get bandpass window such as Tukey, Kaiser, cosine.
        Window is constructed in frequency domain from low to high frequencies
        with given window_function and shape.

        Parameters
        ----------
        center_frequency : float
            Center frequency of frequency bin [Hz]
            If slc is basebanded, center_frequency can be 0.
        sampling_frequency : float
            sampling frequency [Hz]
        fft_size : int
            fft size
        freq_low : float
            low frequency to be passed [Hz]
        freq_high: float
            high frequency to be passed [Hz]
        window_function : str
            window type {tukey, kaiser, cosine}
        window_shape : float
            parameter for the window shape
            kaiser 0<= window_shape < inf
            tukey and cosine 0 <= window_shape <= 1

        Returns
        -------
        filter_1d : np.ndarray
            one dimensional bandpass filter in frequency domain
        '''
        error_channel = journal.error('splitspectrum.get_bandpass_window')
        # construct the frequency bin [Hz]
        frequency = self.freq_spectrum(
                    cfrequency=center_frequency,
                    dt=1.0/sampling_frequency,
                    fft_size=fft_size
                    )

        window_kind = window_function.lower()

        # Windowing effect will appear from freq_low to freq_high 
        # for given frequency bin
        if window_kind == 'tukey':
            if not (0 <= window_shape <= 1):
                err_str = f"Expected window_shape between 0 and 1, got {window_shape}."
                error_channel.log(err_str)
                raise ValueError(err_str)

            filter_1d = self.construct_bandpass_tukey(
                frequency_range=frequency,
                freq_low=freq_low,
                freq_high=freq_high,
                window_shape=window_shape
            )

        elif window_kind == 'kaiser':
            if not (window_shape > 0):
                err_str = f"Expected pedestal bigger than 0, got {window_shape}."
                error_channel.log(err_str)
                raise ValueError(err_str)

            filter_1d = self.construct_bandpass_kaiser(
                frequency_range=frequency,
                freq_low=freq_low,
                freq_high=freq_high,
                window_shape=window_shape
            )

        elif window_kind == 'cosine':
            if not (0 <= window_shape <= 1):
                err_str = f"Expected window_shape between 0 and 1, got {window_shape}."
                error_channel.log(err_str)
                raise ValueError(err_str)
            filter_1d = self.construct_bandpass_cosine(
                frequency_range=frequency,
                freq_low=freq_low,
                freq_high=freq_high,
                window_shape=window_shape
            )

        else:
            err_str = f"window {window_kind} not in (Kaiser, Cosine, Tukey)."
            error_channel.log(err_str)
            raise ValueError(err_str)

        return filter_1d

    def construct_bandpass_cosine(self,
                                  frequency_range,
                                  freq_low,
                                  freq_high,
                                  window_shape):
        '''Generate a Cosine bandpass window

        Parameters
        ----------
        frequency_range : np.ndarray
            Discrete Fourier Transform sample frequency range bins[Hz]
        freq_low : float
            low frequency to be passed [Hz]
        freq_high: float
            high frequency to be passed [Hz]
        window_shape : float
            parameter for the cosine window

        Returns
        -------
        filter_1d : np.ndarray
            one dimensional Cosine bandpass filter in frequency domain
        '''
        filter_1d = self._construct_bandpass_kaiser_cosine(
            frequency_range,
            freq_low,
            freq_high,
            cosine_window,
            window_shape)
        return filter_1d

    def construct_bandpass_kaiser(self,
                                  frequency_range,
                                  freq_low,
                                  freq_high,
                                  window_shape):
        '''Generate a Kaiser bandpass window

        Parameters
        ----------
        frequency_range : np.ndarray
            Discrete Fourier Transform sample frequency range bins[Hz]
        freq_low : float
            low frequency to be passed [Hz]
        freq_high: float
            high frequency to be passed [Hz]
        window_shape : float
            parameter for the kaiser window

        Returns
        -------
        filter_1d : np.ndarray
            one dimensional kaiser bandpass filter in frequency domain
        '''
        filter_1d = self._construct_bandpass_kaiser_cosine(
            frequency_range,
            freq_low,
            freq_high,
            np.kaiser,
            window_shape)
        return filter_1d

    def _construct_bandpass_kaiser_cosine(
            self,
            frequency_range,
            freq_low,
            freq_high,
            window_function,
            window_shape):
        '''Generate a Kaiser or cosine bandpass window

        Parameters
        ----------
        frequency_range : np.ndarray
            Discrete Fourier Transform sample frequency range bins[Hz]
        freq_low : float
            low frequency to be passed [Hz]
        freq_high: float
            high frequency to be passed [Hz]
        window_function : class function
            window type {np.kaiser, cosine_window}
        window_shape : float
            parameter for the kaiser window

        Returns
        -------
        filter_1d : np.ndarray
            one dimensional kaiser bandpass filter in frequency domain
        '''
        error_channel = journal.error(
            'splitspectrum._construct_bandpass_kaiser_cosine')

        subbandwidth = np.abs(freq_high - freq_low)
        fft_size = len(frequency_range)

        if freq_high > np.max(frequency_range):
            err_str = "High frequency is out of frequency bins."
            error_channel.log(err_str)
            raise ValueError(err_str)

        if freq_low < np.min(frequency_range):
            err_str = "Low frequency is out of frequency bins."
            error_channel.log(err_str)
            raise ValueError(err_str)

        # sampling frequency is 1.2 times wider than bandwidth for NISAR
        sampling_bandwidth_ratio = self.sampling_bandwidth_ratio

        sampling_low_frequency = \
            freq_low - (sampling_bandwidth_ratio - 1) * subbandwidth * 0.5
        sampling_high_frequency = \
            freq_high + (sampling_bandwidth_ratio - 1) * subbandwidth * 0.5

        # index for low and high sampling frequency in frequency_range
        idx_freq_low = np.abs(
            frequency_range - sampling_low_frequency).argmin()
        idx_freq_high = np.abs(
            frequency_range - sampling_high_frequency).argmin()

        if idx_freq_low >= idx_freq_high:
            subband_length = idx_freq_high + fft_size - idx_freq_low + 1
        else:
            subband_length = idx_freq_high - idx_freq_low + 1

        filter_1d = np.zeros([fft_size], dtype='complex')

        # window_function is function class {np.kaiser or consine}
        subwindow = window_function(subband_length, window_shape)

        if idx_freq_low >= idx_freq_high:
            filter_1d[idx_freq_low:] = subwindow[0:fft_size - idx_freq_low]
            filter_1d[: idx_freq_high + 1] = subwindow[fft_size - idx_freq_low:]
        else:
            filter_1d[idx_freq_low:idx_freq_high+1] = subwindow

        return filter_1d

    def construct_bandpass_tukey(self,
                                 frequency_range,
                                 freq_low,
                                 freq_high,
                                 window_shape):
        '''Generate a Tukey (raised-cosine) window

        Parameters
        ----------
        frequency_range : np.ndarray
            Discrete Fourier Transform sample frequency range [Hz]
        freq_low : float
            low frequency to be passed [Hz]
        freq_high, : list of float
            high frequency to be passed [Hz]
        window_shape : float
            parameter for the Tukey (raised cosine) filter

        Returns
        -------
        filter_1d : np.ndarray
            one dimensional Tukey bandpass filter in frequency domain
        '''

        fft_size = len(frequency_range)
        freq_mid = 0.5 * (freq_low + freq_high)
        subbandwidth = np.abs(freq_high - freq_low)
        df = 0.5 * subbandwidth * window_shape

        filter_1d = np.zeros(fft_size, dtype='complex')
        for i in range(0, fft_size):
            # Get the absolute value of shifted frequency
            freq = frequency_range[i]
            freqabs = np.abs(freq - freq_mid)
            # Passband. i.e. range of frequencies that can pass
            # through a filter
            if (freq <= (freq_high - df)) and (freq >= (freq_low + df)):
                filter_1d[i] = 1
            # Transition region
            elif ((freq < freq_low + df) and (freq >= freq_low - df)) \
                    or ((freq <= freq_high + df) and (freq > freq_high - df)):
                filter_1d[i] = 0.5 * (
                    1.0 + np.cos(np.pi / (subbandwidth * window_shape)
                    * (freqabs - 0.5 * (1.0 - window_shape) * subbandwidth)))
        return filter_1d


    def shift_doppler_centroid(self, slc_raster, doppler_centroid, sampling_frequency):
        n_az, n_rg = slc_raster.shape

        f = np.arange(0, n_az) / sampling_frequency
        for col_idx_ in range(n_rg):
            slc_col = slc_raster[:, col_idx_]
            dc = doppler_centroid[col_idx_]

            # shift spectrum to zero doppler
            slc_col_shifted = slc_col * np.exp(-1j * 2 * np.pi * dc * f)

            slc_raster[:, col_idx_] = slc_col_shifted

        return slc_raster


    def compensate_azimuth_antenna_pattern(self,
                                           spectrum,
                                           center_frequency,
                                           sampling_frequency,
                                           fft_size):

	    # TODO: use proper compensation algorithm using antenna pattern metadata
        from scipy.ndimage import maximum_filter1d


        def fit_antenna_pattern(f, X, thresh=None, quantile=None):
            """
            Fit a polynomial (degree 2) to the amplitude of the spectrum `X`, considering only values larger than `thresh`.
            If `quantile` is provided, it's used to compute the threshold from the amplitude data.
            """
            X_abs = np.abs(X)

            # optionally determine threshods
            if quantile is not None:
                thresh = np.quantile(X_abs, .3)
            elif thresh is None:
                thresh = X_abs.min()

            # fit polynomial to thresholded amplitude
            coeff = np.polyfit(f[X_abs > thresh], X_abs[X_abs > thresh], 2)

            # evaluate polynomial on frequency axis and return
            fit = np.polyval(coeff, f)
            return fit

        def compensate_az_spectrum(f, X, window_size=None):
            """
            Compensate the azimuth antenna pattern. The amplitude decay from low to high frequencies is determined and corrected.
            The amplitude data is first filtered by taking the max in a moving window in order to filter out low amplitude bins, while preserving the overall decaying shape.
            Then, a polynomial (degree 2) is fit to the filtered amplitude data and its decay (aka `d = max/min`) determined. The polynomial is then transformed to have a min of 1 in the low frequencies
            and a max of `d` towards the high frequencies. The transformed polynomial is multiplied with the raw data and returned as the compensated data.
            """
            X_abs = np.abs(X)
            window_size = window_size or len(X)//10

            # compute max in moving window
            X_max = maximum_filter1d(X_abs, size=window_size)

            # fit 2nd degree polyonomial to filtered spectrum, using only values of the top 70% quantile to avoid zero-amplitude bins outside of bandwidth
            poly_fit_max = fit_antenna_pattern(f, X_max, quantile=.3)

            # compute compensation polynomial
            min = poly_fit_max.min(); max = poly_fit_max.max()
            poly_compensate = -(poly_fit_max-max)/(max-min) * (max/min-1) + 1           # move to origin, flip, rescale to max/min for multiplication and finally add 1

            return X * poly_compensate


        frequency = self.freq_spectrum(
                    cfrequency=center_frequency,
                    dt=1.0/sampling_frequency,
                    fft_size=fft_size
                    )

        for col_idx_ in range(spectrum.shape[1]):
            spectrum_col = spectrum[:, col_idx_]
            spectrum_col_corrected = compensate_az_spectrum(frequency, spectrum_col)

            # assign corrected spectrum
            spectrum[:, col_idx_] = spectrum_col_corrected

        return spectrum
