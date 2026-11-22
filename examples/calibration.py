"""Exact finite-sample calibration on an independent synthetic population."""
import json
import math
import numpy as np
from grail.conformal import split_conformal


def main():
    rng = np.random.default_rng(42)
    calibration_errors = rng.normal(size=(100, 3))
    test_predictions = np.zeros((128, 3))
    test_targets = rng.normal(size=(128, 3))
    band = split_conformal(calibration_errors, test_predictions, alpha=.1)
    k = math.ceil((100+1)*.9)
    radius = np.sort(np.max(np.abs(calibration_errors), axis=1))[k-1]
    np.testing.assert_allclose(np.asarray(band.high)-test_predictions, radius)
    hits = np.all((test_targets >= np.asarray(band.low)) &
                  (test_targets <= np.asarray(band.high)), axis=1)
    tiny = split_conformal(calibration_errors[:2], np.zeros(3), alpha=.1)
    assert np.isneginf(np.asarray(tiny.low)).all() and np.isposinf(np.asarray(tiny.high)).all()
    print(json.dumps({"calibration_count": 100, "order_statistic": k,
        "radius": float(radius), "test_hits": int(hits.sum()), "test_count": 128,
        "tiny_two_sample_90_percent_band": "infinite", "population": "independent synthetic normal errors"}, indent=2))


if __name__ == "__main__":
    main()
