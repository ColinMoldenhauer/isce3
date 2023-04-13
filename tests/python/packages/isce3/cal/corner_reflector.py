import tempfile
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
from numpy.random import default_rng
from numpy.typing import ArrayLike

import isce3
import iscetest
from isce3.cal.corner_reflector import (
    cr_to_enu_rotation,
    enu_to_cr_rotation,
    normalize_vector,
)


def wrap(phi: ArrayLike) -> np.ndarray:
    """Wrap the input angle (in radians) to the interval [-pi, pi)."""
    phi = np.asarray(phi)
    return np.mod(phi + np.pi, 2.0 * np.pi) - np.pi


def angle_between(u: ArrayLike, v: ArrayLike, *, degrees: bool = False) -> float:
    """Measure the angle between two vectors."""
    u, v = map(normalize_vector, [u, v])
    theta = np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))
    if degrees:
        theta = np.rad2deg(theta)
    return theta


class TestParseTriangularTrihedralCornerReflectorCSV:
    def test_parse_csv(self):
        # Parse CSV file containing 3 Northeast-looking corner reflectors.
        csv = Path(iscetest.data) / "abscal/REE_CORNER_REFLECTORS_INFO.csv"
        crs = list(isce3.cal.parse_triangular_trihedral_cr_csv(csv))

        # Check the number of corner reflectors.
        assert len(crs) == 3

        # Check that CR latitudes & longitudes are within ~1 degree of their approximate
        # expected location.
        atol = np.sin(np.deg2rad(1.0))
        lats = [cr.llh.latitude for cr in crs]
        approx_lat = np.deg2rad(69.5)
        npt.assert_allclose(np.abs(wrap(lats - approx_lat)), 0.0, atol=atol)
        lons = [cr.llh.longitude for cr in crs]
        approx_lon = np.deg2rad(-128.5)
        npt.assert_allclose(np.abs(wrap(lons - approx_lon)), 0.0, atol=atol)

        # Check that CR heights are within 1 meter of their approximate expected location.
        heights = [cr.llh.height for cr in crs]
        approx_height = 490.0
        npt.assert_allclose(heights, approx_height, atol=1.0)

        # Check that CR azimuth & elevation angles are each within ~1 degree of their
        # approximate expected orientation.
        azs = [cr.azimuth for cr in crs]
        approx_az = np.deg2rad(317.0)
        npt.assert_allclose(np.abs(wrap(azs - approx_az)), 0.0, atol=atol)
        els = [cr.elevation for cr in crs]
        approx_el = np.deg2rad(12.5)
        npt.assert_allclose(np.abs(wrap(els - approx_el)), 0.0, atol=atol)

        # Check that CR side lengths all match their expected value.
        side_lengths = [cr.side_length for cr in crs]
        expected_side_length = 3.4629120649497214
        npt.assert_array_equal(side_lengths, expected_side_length)

    def test_empty_csv(self):
        with tempfile.NamedTemporaryFile() as f:
            csv = f.name
            crs = isce3.cal.parse_triangular_trihedral_cr_csv(csv)
            assert len(list(crs)) == 0


class TestCRToENURotation:

    sin = lambda x: np.sin(np.deg2rad(x))
    cos = lambda x: np.cos(np.deg2rad(x))

    # The test is parameterized as follows:
    # The first two parameters (az,el) are the orientation angles of the corner
    # reflector, in degrees.
    # The remaining three parameters (x,y,z) are ENU unit vectors that are expected to
    # be aligned with the three legs of the corner reflector.
    @pytest.mark.parametrize(
        "az,el,x,y,z",
        [
            (0, 0, [cos(-45), sin(-45), 0], [cos(45), sin(45), 0], [0, 0, 1]),
            (45, 0, [0, -1, 0], [1, 0, 0], [0, 0, 1]),
            (0, 90, [0, -sin(45), cos(45)], [0, -sin(-45), cos(-45)], [-1, 0, 0]),
            (
                270,
                45,
                [cos(45), sin(45) * cos(45), sin(45) * sin(45)],
                [-sin(45), cos(45) * cos(45), cos(45) * sin(45)],
                [0, -sin(45), cos(45)],
            ),
            (
                -45,
                -90,
                [cos(-45) ** 2, sin(-45) * cos(-45), sin(-45)],
                [cos(135) * cos(-45), sin(135) * cos(-45), sin(-45)],
                [cos(45), sin(45), 0],
            ),
        ],
    )
    def test_cr_to_enu(self, az, el, x, y, z):
        # # Convert degrees to radians.
        az, el = np.deg2rad([az, el])

        # Get rotation (quaternion) from CR-intrinsic coordinates to ENU coordinates.
        q = cr_to_enu_rotation(az=az, el=el)

        # Compare the rotated basis vectors to the expected ENU vectors.
        atol = 1e-6
        assert angle_between(q.rotate([1.0, 0.0, 0.0]), x) < atol
        assert angle_between(q.rotate([0.0, 1.0, 0.0]), y) < atol
        assert angle_between(q.rotate([0.0, 0.0, 1.0]), z) < atol

    # The test is parameterized as follows:
    # The first two parameters (az,el) are defined as in the previous test.
    # The remaining three parameters (e,n,u) are unit vectors in the corner
    # reflector-intrinsic coordinate system that are expected to align with the E,N,U
    # basis vectors.
    @pytest.mark.parametrize(
        "az,el,e,n,u",
        [
            (0, 0, [cos(45), sin(45), 0], [cos(135), sin(135), 0], [0, 0, 1]),
            (45, 0, [0, 1, 0], [-1, 0, 0], [0, 0, 1]),
            (0, 90, [0, 0, -1], [-sin(45), cos(45), 0], [-sin(-45), cos(-45), 0]),
            (
                270,
                45,
                [cos(-45), sin(-45), 0],
                [cos(45) * sin(45), sin(45) * cos(-45), -sin(45)],
                [cos(45) * cos(-45), sin(45) * cos(45), -sin(-45)],
            ),
            (
                -45,
                -90,
                [cos(-45) ** 2, sin(-45) * cos(-45), sin(45)],
                [cos(135) * cos(45), sin(135) * cos(45), sin(45)],
                [cos(-135), sin(-135), 0],
            ),
        ],
    )
    def test_enu_to_cr(self, az, el, e, n, u):
        # Convert degrees to radians.
        az, el = np.deg2rad([az, el])

        # Get rotation (quaternion) from CR-intrinsic coordinates to ENU coordinates.
        q = enu_to_cr_rotation(az=az, el=el)

        # Compare the rotated basis vectors to the expected ENU vectors.
        atol = 1e-6
        assert angle_between(q.rotate([1.0, 0.0, 0.0]), e) < atol
        assert angle_between(q.rotate([0.0, 1.0, 0.0]), n) < atol
        assert angle_between(q.rotate([0.0, 0.0, 1.0]), u) < atol

    def test_az_and_el(self):
        # Randomly sample azimuth angles in [-pi, pi) and elevation angles in [0, pi/2).
        rng = default_rng(seed=1234)
        n = 100
        azimuths = rng.uniform(-np.pi, np.pi, size=n)
        elevations = rng.uniform(0.0, np.pi / 2, size=n)

        for az, el in zip(azimuths, elevations):
            q = cr_to_enu_rotation(az=az, el=el)

            # Rotate the corner reflector boresight vector to ENU coordinates. Compute
            # the angle in the E-N plane of the vector from the E-axis, measured
            # clockwise. This should match the azimuth angle of the corner reflector.
            boresight = np.array([1, 1, 0]) / np.sqrt(2)
            x = q.rotate(boresight)
            theta = np.arctan2(-x[1], x[0])
            assert np.abs(wrap(theta - az)) < 1e-6

            # Rotate the corner reflector Z-axis to ENU coordinates. Compute the angle
            # between the rotated vector and the U-axis. This should match the elevation
            # angle of the corner reflector.
            z = q.rotate([0.0, 0.0, 1.0])
            phi = np.arccos(z[2])
            assert np.abs(wrap(phi - el)) < 1e-6

    def test_roundtrip(self):
        # Randomly sample azimuth angles in [-pi, pi) and elevation angles in [0, pi/2).
        rng = default_rng(seed=1234)
        n = 100
        azimuths = rng.uniform(-np.pi, np.pi, size=n)
        elevations = rng.uniform(0.0, np.pi / 2, size=n)

        # Test that a roundtrip rotation (CR -> ENU -> CR) is identity.
        identity = isce3.core.Quaternion([1.0, 0.0, 0.0, 0.0])
        for az, el in zip(azimuths, elevations):
            q_cr2enu = cr_to_enu_rotation(az=az, el=el)
            q_enu2cr = enu_to_cr_rotation(az=az, el=el)
            assert (q_enu2cr * q_cr2enu).is_approx(identity)
