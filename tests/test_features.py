import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from features import clean, make_features, make_labels
from simulator import PARAMS, simulate_pond


def _sample_sensor_data(rows=100):
    hours = np.arange(rows)
    return pd.DataFrame(
        {
            "pond_id": np.repeat([1, 2], rows // 2),
            "datetime": pd.date_range("2024-01-01", periods=rows, freq="h"),
            "aerator_on": (hours % 24 < 8).astype(np.int8),
            "sensor_temp_c": 27 + np.sin(hours / 24),
            "sensor_do_mgl": 5 + np.cos(hours / 12),
            "sensor_ph": 7.5 + np.sin(hours / 36) * 0.1,
            "sensor_turb_ntu": 3 + np.cos(hours / 18),
        }
    )


def test_make_features_returns_46_non_leaking_features():
    features = make_features(clean(_sample_sensor_data()))

    assert features.shape[1] == 46
    forbidden = {
        "alive",
        "deaths",
        "hazard_per_h",
        "stress_index",
        "chl_index",
        "rain_day",
        "stock_density_kg_m3",
        "has_aerator",
    }
    assert not any(
        name.startswith("true_") or name in forbidden or name.startswith("h_")
        for name in features.columns
    )


def test_labels_contain_only_binary_values_or_nan():
    rows = 40
    data = pd.DataFrame(
        {
            "pond_id": np.repeat([1, 2], rows // 2),
            "alive": 1000,
            "deaths": np.zeros(rows, dtype=np.int64),
        }
    )
    labels = make_labels(data)

    for column in labels:
        assert set(labels[column].dropna().unique()).issubset({0.0, 1.0})
        assert labels[column].isna().any()


def test_two_ponds_simulate_deterministically_with_seed_one():
    def simulate_two_ponds():
        rng = np.random.default_rng(1)
        return pd.concat(
            [
                simulate_pond(
                    pond_id,
                    rng,
                    PARAMS,
                    start="2024-01-01",
                    end="2024-01-04",
                )
                for pond_id in (1, 2)
            ],
            ignore_index=True,
        )

    assert_frame_equal(simulate_two_ponds(), simulate_two_ponds())
