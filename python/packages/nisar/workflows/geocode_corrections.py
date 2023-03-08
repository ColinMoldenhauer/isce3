'''
Compute azimuth and slant range geocoding corrections as LUT2d
'''
import isce3
from isce3.atmosphere.tec_product import tec_lut2d_from_json

from nisar.products.readers import SLC


def _get_accumulated_azimuth_corrections():
    '''
    Compute, accumulate, and return azimuth time corrections as LUT2d. Default
    to default LUT2d if provided parameters do not require corrections to be
    computed.

    Yields:
    -------
    _: isce3.core.LUT2d
        Azimuth correction for geocoding. Currently only no corrections are
        computed and a default isce3.core.LUT2d is be passed back.
    '''
    # empty until actual correction computations are ready for integration
    return isce3.core.LUT2d()


def _get_accumulated_srange_corrections(cfg, slc, frequency, orbit):
    '''
    Compute, accumulate, and return slant range corrections as LUT2d. Default
    to default LUT2d if provided parameters do not require corrections to be
    computed.

    Currently on TEC corrections available. Others will be added as they
    become available.

    Parameters:
    -----------
    cfg: dict
        Dict containing the runconfiguration parameters
    slc: nisar.products.readers.SLC
        NISAR single look complex (SLC) object containing swath and radar grid
        parameters
    frequency: ['A', 'B']
        Str identifcation for NISAR SLC frequencies
    orbit: isce3.core.Orbit
        Object containing orbit associated with SLC

    Yields:
    -------
    tec_correction: isce3.core.LUT2d
        Slant range correction for geocoding. Currently only TEC corrections
        are considered. If no TEC JSON file is provided in the cfg parameter,
        a default isce3.core.LUT2d will be passed back.
    '''

    # Default TEC slant range correction to default LUT2d
    tec_correction = isce3.core.LUT2d()

    # Compute TEC slant range correction if TEC file is provided
    tec_file = cfg["dynamic_ancillary_file_group"]['tec_file']
    if tec_file is not None:
        # Get SLC object for parameters inside necessary for TEC computations
        input_hdf5 = cfg['input_file_group']['input_file_path']
        slc = SLC(hdf5file=input_hdf5)
        center_freq = slc.getSwathMetadata(frequency).processed_center_frequency
        doppler = isce3.core.LUT2d()
        radar_grid = slc.getRadarGrid(frequency)

        # DEM file for DEM interpolator and ESPF for ellipsoid
        dem_file = cfg['dynamic_ancillary_file_group']['dem_file']

        tec_correction = tec_lut2d_from_json(tec_file, center_freq, orbit,
                                             radar_grid, doppler, dem_file)

    return tec_correction


def get_az_srg_corrections(cfg, slc, frequency, orbit):
    '''
    Compute azimuth and slant range geocoding corrections and return as LUT2d.
    Default to default LUT2d for either if provided parameters do not require
    corrections to be computed.

    Parameters:
    -----------
    cfg: dict
        Dict containing the runconfiguration parameters
    slc: nisar.products.readers.SLC
        NISAR single look complex (SLC) object containing swath and radar grid
        parameters
    frequency: ['A', 'B']
        Str identifcation for NISAR SLC frequencies
    orbit: isce3.core.Orbit
        Object containing orbit associated with SLC

    Yields:
    -------
    az_corrections: isce3.core.LUT2d
        Azimuth correction for geocoding. Currently only no corrections are
        computed and a default isce3.core.LUT2d is be passed back.
    srange_corrections: isce3.core.LUT2d
        Slant range correction for geocoding. Currently only TEC corrections
        are considered. If no TEC JSON file is provided in the cfg parameter,
        a default isce3.core.LUT2d will be passed back.
    '''
    az_corrections = _get_accumulated_azimuth_corrections()
    srange_corrections = _get_accumulated_srange_corrections(cfg, slc,
                                                             frequency, orbit)

    return az_corrections, srange_corrections
