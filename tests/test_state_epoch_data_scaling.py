import numpy as np

from utils.state_epoch_data import scale_data


def test_scale_data_normalize_handles_constant_columns():
    data = np.array([[5.0, 1.0], [5.0, 2.0]])

    scaled = scale_data(data.copy(), method="normalize")

    assert np.all(np.isfinite(scaled))
    assert np.allclose(scaled[:, 0], 0.0)


def test_scale_data_standardize_handles_zero_variance():
    data = np.array([[3.0], [3.0], [4.0]])

    scaled = scale_data(data.copy(), method="standardize")

    assert np.all(np.isfinite(scaled))


def test_fractional_change_epoch_respects_baseline_epoch():
    data = np.array([[1.0], [2.0], [5.0], [6.0]])
    epochs = [(0, 2), (2, 4)]
    epoch_names = ["baseline", "stim"]
    period = 1

    scaled = scale_data(
        data.copy(),
        method="fractional_change",
        epochs=epochs,
        period=period,
        baseline_epoch="stim",
        epoch_names=epoch_names,
    )

    shifted = data + np.abs(np.nanmin(data))
    stim_mean = np.nanmean(shifted[2:], axis=0)

    assert np.allclose(scaled, shifted / stim_mean)
