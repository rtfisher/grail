"""Finite-sample counterexamples for the calibrated numerical threshold."""
import numpy as np
import pytest

from grail.conformal import conformal_quantile, split_conformal


@pytest.mark.unit
def test_exact_order_statistic_instead_of_interpolation():
    # n=4, alpha=.4 gives k=3: interpolating at k/n would return 27.5.
    assert conformal_quantile([0., 10., 20., 50.], alpha=.4) == 20.


@pytest.mark.unit
def test_unavailable_ninetieth_percentile_is_infinite():
    # Six calibration samples cannot support a finite standard 90% threshold.
    band = split_conformal(np.arange(6.), np.zeros(1), alpha=.1)
    assert np.isneginf(np.asarray(band.low)).all()
    assert np.isposinf(np.asarray(band.high)).all()


@pytest.mark.unit
@pytest.mark.parametrize('scores,alpha,minimum_hits', [
    ([0., 1., 2., 3.], .25, 3),
    ([0., 1., 2., 3.], .1, 4),
    ([0., 0., 1., 1.], .5, 2),
])
def test_all_held_out_labels_satisfy_finite_sample_count(scores, alpha, minimum_hits):
    # Every labeling of one finite exchangeable orbit must satisfy the rank
    # count, including ties and the infinite-threshold case. No Monte Carlo.
    hits = 0
    for i, held_out in enumerate(scores):
        calibration = scores[:i] + scores[i + 1:]
        hits += held_out <= conformal_quantile(calibration, alpha=alpha)
    assert hits >= minimum_hits
    assert hits / len(scores) >= 1 - alpha


@pytest.mark.unit
def test_custom_score_must_support_the_box_geometry():
    with pytest.raises(ValueError, match='sup-norm'):
        split_conformal(np.array([[0., 10.], [0., 20.]]), np.zeros(2),
                        alpha=.5, nonconformity=lambda row: 0.)


@pytest.mark.unit
@pytest.mark.parametrize('scores,alpha', [([], .1), ([np.nan], .1), ([1.], 0.), ([1.], 1.)])
def test_threshold_requires_the_stated_real_score_domain(scores, alpha):
    with pytest.raises(ValueError):
        conformal_quantile(scores, alpha=alpha)
