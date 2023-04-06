import csv
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

import isce3


@dataclass(frozen=True)
class TriangularTrihedralCornerReflector:
    """
    A triangular trihedral corner reflector (CR).

    Parameters
    ----------
    id : str
        The unique identifier of the corner reflector.
    llh : isce3.core.LLH
        The geodetic coordinates of the corner reflector: the geodetic longitude and
        latitude in radians and the height above the WGS 84 ellipsoid in meters.
    elevation : float
        The elevation angle, in radians. This is the tilt angle of the vertical axis of
        the corner reflector with respect to the ellipsoid normal vector.
    azimuth : float
        The azimuth angle, in radians. This is the heading angle of the corner
        reflector, or the angle that the corner reflector boresight makes w.r.t
        geographic East, measured clockwise positive in the E-N plane.
    side_length : float
        The length of each leg of the trihedral, in meters.
    """

    id: str
    llh: isce3.core.LLH
    elevation: float
    azimuth: float
    side_length: float


def parse_triangular_trihedral_cr_csv(
    csvfile: os.PathLike,
) -> Iterator[TriangularTrihedralCornerReflector]:
    """
    Parse a CSV file containing triangular trihedral corner reflector (CR) data.

    Returns an iterator over corner reflectors within the file.

    The CSV file is assumed to be in the format used by the `UAVSAR Rosamond Corner
    Reflector Array (RCRA) <https://uavsar.jpl.nasa.gov/cgi-bin/calibration.pl>`_, which
    contains the following columns:

    1. Corner reflector ID
    2. Latitude (deg)
    3. Longitude (deg)
    4. Height above ellipsoid (m)
    5. Azimuth (deg)
    6. Tilt / Elevation angle (deg)
    7. Side length (m)

    Parameters
    ----------
    csvfile : path_like
        The CSV file path.

    Yields
    ------
    cr : TriangularTrihedralCornerReflector
        A corner reflector.
    """
    dtype = np.dtype(
        [
            ("id", np.str_),
            ("lat", np.float64),
            ("lon", np.float64),
            ("height", np.float64),
            ("az", np.float64),
            ("el", np.float64),
            ("side_length", np.float64),
        ]
    )

    # Parse CSV data. Skip initial (header) row.
    # Any additional columns beyond those mentioned above will be ignored so that new
    # additions to the file spec don't break compatibility.
    cols = range(len(dtype))
    crs = np.loadtxt(csvfile, dtype=dtype, delimiter=",", skiprows=1, usecols=cols)

    # Convert lat, lon, az, & el angles to radians.
    for attr in ["lat", "lon", "az", "el"]:
        crs[attr] = np.deg2rad(crs[attr])

    for cr in crs:
        id, lat, lon, height, az, el, side_length = cr
        llh = isce3.core.LLH(lon, lat, height)
        yield TriangularTrihedralCornerReflector(id, llh, el, az, side_length)


def cr_to_enu_rotation(el: float, az: float) -> isce3.core.Quaternion:
    """
    Get a quaternion to rotate from a corner reflector (CR) intrinsic coordinate system
    to East, North, Up (ENU) coordinates.

    The CR coordinate system has three orthogonal axes aligned with the three legs of
    the trihedral. The coordinate system is defined such that, when the elevation and
    azimuth angles are each 0 degrees:

    * the x-axis points 45 degrees South of the East-axis
    * the y-axis points 45 degrees North of the East-axis
    * the z-axis is aligned with the Up-axis

    Parameters
    ----------
    el : float
        The elevation angle, in radians. This is the tilt angle of the vertical axis of
        the corner reflector with respect to the ellipsoid normal vector.
    az : float
        The azimuth angle, in radians. This is the heading angle of the corner
        reflector, or the angle that the corner reflector boresight makes w.r.t
        geographic East, measured clockwise positive in the E-N plane.

    Returns
    -------
    q : isce3.core.Quaternion
        A unit quaternion representing the rotation from CR to ENU coordinates.
    """
    q1 = isce3.core.Quaternion(angle=el, axis=[1.0, -1.0, 0.0])
    q2 = isce3.core.Quaternion(angle=0.25 * np.pi + az, axis=[0.0, 0.0, -1.0])
    return q2 * q1


def enu_to_cr_rotation(el: float, az: float) -> isce3.core.Quaternion:
    """
    Get a quaternion to rotate from East, North, Up (ENU) coordinates to a corner
    reflector (CR) intrinsic coordinate system.

    The CR coordinate system has three orthogonal axes aligned with the three legs of
    the trihedral. The coordinate system is defined such that, when the elevation and
    azimuth angles are each 0 degrees:

    * the x-axis points 45 degrees South of the East-axis
    * the y-axis points 45 degrees North of the East-axis
    * the z-axis is aligned with the Up-axis

    Parameters
    ----------
    el : float
        The elevation angle, in radians. This is the tilt angle of the vertical axis of
        the corner reflector with respect to the ellipsoid normal vector.
    az : float
        The azimuth angle, in radians. This is the heading angle of the corner
        reflector, or the angle that the corner reflector boresight makes w.r.t
        geographic East, measured clockwise positive in the E-N plane.

    Returns
    -------
    q : isce3.core.Quaternion
        A unit quaternion representing the rotation from ENU to CR coordinates.
    """
    return cr_to_enu_rotation(el, az).conjugate()


def normalize_vector(v: ArrayLike) -> np.ndarray:
    """
    Normalize a vector.

    Compute the unit vector pointing in the direction of the input vector.

    Parameters
    ----------
    v : array_like
        The input vector. Must be nonzero.

    Returns
    -------
    u : numpy.ndarray
        The normalized vector.
    """
    return np.asanyarray(v) / np.linalg.norm(v)


def target2platform_unit_vector(
    target_llh: isce3.core.LLH,
    orbit: isce3.core.Orbit,
    doppler: isce3.core.LUT2d,
    wavelength: float,
    look_side: str,
    ellipsoid: isce3.core.Ellipsoid = isce3.core.WGS84_ELLIPSOID,
    *,
    geo2rdr_params: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    """
    Compute the target-to-platform line-of-sight (LOS) unit vector.

    Parameters
    ----------
    target_llh : isce3.core.LLH
        The target position expressed as longitude, latitude, and height above the
        reference ellipsoid in radians, radians, and meters respectively.
    orbit : isce3.core.Orbit
        The trajectory of the radar antenna phase center.
    doppler : isce3.core.LUT2d
        The Doppler centroid of the data, in hertz, expressed as a function of azimuth
        time, in seconds relative to the orbit epoch, and slant range, in meters. Note
        that this should be the native Doppler of the data acquisition, which may in
        general be different than the Doppler associated with the radar grid that the
        focused image was projected onto.
    wavelength : float
        The radar wavelength, in meters.
    look_side : {"Left", "Right"}
        The radar look direction.
    ellipsoid : isce3.core.Ellipsoid, optional
        The geodetic reference ellipsoid, with dimensions in meters. Defaults to the
        WGS 84 ellipsoid.
    geo2rdr_params : dict or None, optional
        An optional dict of parameters configuring the behavior of the root-finding
        routine used in geo2rdr. The following keys are supported:

        'threshold':
          The absolute azimuth time convergence tolerance, in seconds.

        'maxiter':
          The maximum number of Newton-Raphson iterations.

        'delta_range':
          The step size for computing numerical gradient of Doppler, in meters.

    Returns
    -------
    u : (3,) numpy.ndarray
        A unit vector pointing from the target position to the platform position in
        Earth-Centered, Earth-Fixed (ECEF) coordinates.
    """
    # Convert LLH object to an array containing [lon, lat, height].
    target_llh = target_llh.to_vec3()

    if geo2rdr_params is None:
        geo2rdr_params = {}

    # Run geo2rdr to get the target azimuth time coordinate in seconds since the orbit
    # epoch.
    aztime, _ = isce3.geometry.geo2rdr(
        target_llh, ellipsoid, orbit, doppler, wavelength, look_side, **geo2rdr_params,
    )

    # Get platform (x,y,z) position in ECEF coordinates.
    platform_xyz, _ = orbit.interpolate(aztime)

    # Get target (x,y,z) position in ECEF coordinates.
    target_xyz = ellipsoid.lon_lat_to_xyz(target_llh)

    return normalize_vector(platform_xyz - target_xyz)


def predict_triangular_trihedral_cr_rcs(
    cr: TriangularTrihedralCornerReflector,
    orbit: isce3.core.Orbit,
    doppler: isce3.core.LUT2d,
    wavelength: float,
    look_side: str,
    *,
    geo2rdr_params: Optional[Mapping[str, float]] = None,
) -> float:
    r"""
    Predict the radar cross-section (RCS) of a triangular trihedral corner reflector.

    Calculate the predicted monostatic RCS of a triangular trihedral corner reflector,
    given the corner reflector dimensions and imaging geometry\ [1]_.

    Parameters
    ----------
    cr : TriangularTrihedralCornerReflector
        The corner reflector position, orientation, and size.
    orbit : isce3.core.Orbit
        The trajectory of the radar antenna phase center.
    doppler : isce3.core.LUT2d
        The Doppler centroid of the data, expressed as a function of azimuth time, in
        seconds relative to the orbit epoch, and slant range, in meters.
    wavelength : float
        The radar wavelength, in meters.
    look_side : {"Left", "Right"}
        The radar look direction.
    geo2rdr_params : dict or None, optional
        An optional dict of parameters configuring the behavior of the root-finding
        routine used in geo2rdr. The following keys are supported:

        'threshold':
          The absolute azimuth time convergence tolerance, in seconds.

        'maxiter':
          The maximum number of Newton-Raphson iterations.

        'delta_range':
          The step size for computing numerical gradient of Doppler, in meters.

    Returns
    -------
    sigma : float
        The predicted radar cross-section of the corner reflector, in meters squared
        (linear scale -- not dB).

    References
    ----------
    .. [1] R. R. Bonkowski, C. R. Lubitz, and C. E. Schensted, “Studies in Radar
       Cross-Sections - VI. Cross-sections of corner reflectors and other multiple
       scatterers at microwave frequencies,” University of Michigan Radiation
       Laboratory, Tech. Rep., October 1953.
    """
    # Get the target-to-platform line-of-sight vector in ECEF coordinates.
    los_vec_ecef = target2platform_unit_vector(
        target_llh=cr.llh,
        orbit=orbit,
        doppler=doppler,
        wavelength=wavelength,
        look_side=look_side,
        ellipsoid=isce3.core.WGS84_ELLIPSOID,
        geo2rdr_params=geo2rdr_params,
    )

    # Convert to ENU coordinates and then to CR-intrinsic coordinates.
    los_vec_enu = isce3.geometry.enu_vector(
        cr.llh.longitude, cr.llh.latitude, los_vec_ecef
    )
    los_vec_cr = enu_to_cr_rotation(cr.elevation, cr.azimuth).rotate(los_vec_enu)

    # Get the CR boresight unit vector in the same coordinates.
    boresight_vec = normalize_vector([1.0, 1.0, 1.0])

    # Get the direction cosines between the two vectors, sorted in ascending order.
    p1, p2, p3 = np.sort(los_vec_cr * boresight_vec)

    # Compute expected RCS.
    a = p1 + p2 + p3
    if (p1 + p2) > p3:
        b = np.sqrt(3.0) * a - 2.0 / (np.sqrt(3.0) * a)
    else:
        b = 4.0 * p1 * p2 / a

    return 4.0 * np.pi * cr.side_length ** 4 * b ** 2 / wavelength ** 2
