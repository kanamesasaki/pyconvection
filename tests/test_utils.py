import numpy as np
import pytest
from pyconvection.utils import atmosphere_temperature

# pytest tests/test_utils.py

def test_atmosphere_temperature():
    input = np.array([-100, 0, 11000, 20000, 32000, 47000, 51000, 71000, 84852, 90000])
    with pytest.warns(UserWarning):
        result = atmosphere_temperature(input)
    assert np.isnan(result[0])
    assert result[1] == 15.0
    assert result[2] == -56.5
    assert result[3] == -56.5
    assert result[4] == -44.5
    assert result[5] == -2.5
    assert result[6] == -2.5
    assert result[7] == -58.5
    assert result[8] == -58.5
    assert np.isnan(result[9])

