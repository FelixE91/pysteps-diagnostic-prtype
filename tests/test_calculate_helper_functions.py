# python
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

def test_calculate_members_mean():
    data = np.array([[[1, 2]], [[3, 4]]])  # Shape (2,1,2)
    mean_result = calculate_members_mean(data)
    expected = np.array([[2, 3]])
    assert mean_result.shape == (1, 2)
    assert_array_almost_equal(mean_result, expected)

def test_calculate_members_mean_234():
    data = np.arange(24).reshape(2, 3, 4)
    mean_result = calculate_members_mean(data)
    expected = np.array([
        [ 6,  7,  8,  9],
        [10, 11, 12, 13],
        [14, 15, 16, 17]
    ])
    assert mean_result.shape == (3, 4)
    assert_array_almost_equal(mean_result, expected)


#     # Result grid
#     result = np.zeros((precipGrid.shape[0], precipGrid.shape[1]))
#     topoZSDiffGrid = (Znow - topographyGrid)  # dzs
#     precipMask = (precipGrid > RRMIN)
#
#     # SNOW ((dzs<-1.5*DZML) || ( (ZH[i][j] <= 1.5*DZML) && (dzs<=0)))
#     snowMask = (topoZSDiffGrid < (-1.5 * DZML)) | ((topographyGrid <= (1.5 * DZML)) & (topoZSDiffGrid <= 0))
#     result[snowMask & precipMask] = 3
#
#     # RAIN+SNOW DIAGNOSIS (dzs < 0.5 * DZML) = 2
#     rainSnowMask = ~snowMask & (topoZSDiffGrid < (0.5 * DZML))
#     result[rainSnowMask & precipMask] = 2
#
#     # RAIN
#     rainMask = ~snowMask & ~rainSnowMask
#     result[rainMask & precipMask] = 1
#
#     # FREEZING RAIN DIAGNOSIS 4
#     # if ((PT[i][j]==1) && ( (tg_<TG0 && TT[i][j]<TT0) || TT[i][j]<TG0))
#     freezingMask = (result == 1) & (((GroundTemp < TG0) & (Temp < TT0)) | (Temp < TG0))
#     result[freezingMask] = 4

# Test the calculate_precip_type function with different scenarios:
    # PT=0  no precip
    # PT=1  rain
    # PT=2  rain/snow mix
    # PT=3  snow
    # PT=4  freezing rain
def test_calculate_precip_type_dry():
    # Test with no precipitation
    precip = np.zeros((2, 3, 4))
    snow_level = np.zeros((2, 3, 4))
    topography = np.zeros((3, 4))
    ground_temp = np.zeros((2, 3, 4))
    temp = np.ones((2, 3, 4))
    result = calculate_precip_type(precip, snow_level, topography, ground_temp, temp)
    expected_result = np.zeros((2, 3, 4))
    assert_array_almost_equal(result, expected_result)

def test_calculate_precip_type_rain():
    # Test with rain
    precip = np.ones((2, 3, 4))
    snow_level = np.ones((2, 3, 4))* 200
    topography = np.zeros((3, 4))
    ground_temp = np.ones((2, 3, 4))*5
    temp = np.ones((2, 3, 4)) * 10
    result = calculate_precip_type(precip, snow_level, topography, ground_temp, temp)
    expected_result = np.ones((2, 3, 4))
    assert_array_almost_equal(result, expected_result)

def test_calculate_precip_type_snow():
    # Test with snow
    precip = np.ones((2, 3, 4))
    snow_level = np.ones((2, 3, 4)) * 200
    topography = np.ones((3, 4)) * 100
    ground_temp = np.ones((2, 3, 4)) * -10
    temp = np.ones((2, 3, 4)) * -5
    result = calculate_precip_type(precip, snow_level, topography, ground_temp, temp)
    expected_result = np.ones((2, 3, 4)) * 3
    assert_array_almost_equal(result, expected_result)

def test_calculate_precip_type_freezing_rain():
    # Test with freezing rain
    # T2m < TTO and TG < TG0
    precip = np.ones((2, 3, 4))
    snow_level = np.ones((2, 3, 4)) * 200
    topography = np.ones((3, 4)) * 100
    ground_temp = np.ones((2, 3, 4)) * -10
    temp = np.ones((2, 3, 4)) * 1
    result = calculate_precip_type(precip, snow_level, topography, ground_temp, temp)
    expected_result = np.ones((2, 3, 4)) * 4
    assert_array_almost_equal(result, expected_result)

    # T2m < TGO
    precip = np.ones((2, 3, 4))
    snow_level = np.ones((2, 3, 4)) * 200
    topography = np.ones((3, 4)) * 100
    ground_temp = np.ones((2, 3, 4)) * 1
    temp = np.ones((2, 3, 4)) * -5
    result = calculate_precip_type(precip, snow_level, topography, ground_temp, temp)
    expected_result = np.ones((2, 3, 4)) * 4
    assert_array_almost_equal(result, expected_result)

def test_calculate_precip_type_rain_snow_mix():

