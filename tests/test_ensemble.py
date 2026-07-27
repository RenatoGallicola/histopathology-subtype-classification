"""Blending the two branches and searching the mixing weight."""
import numpy as np
import pytest

from src.ensemble import alpha_curve, blend, score, tune_alpha


class TestBlend:
    def test_endpoints_select_one_branch(self):
        a, b = np.array([[1.0, 0, 0, 0]]), np.array([[0, 1.0, 0, 0]])
        assert np.allclose(blend(a, b, 1.0), a)
        assert np.allclose(blend(a, b, 0.0), b)

    def test_midpoint_is_the_mean(self):
        a, b = np.zeros((3, 4)), np.ones((3, 4))
        assert np.allclose(blend(a, b, 0.5), 0.5)

    def test_convex_combination_preserves_normalisation(self):
        rng = np.random.default_rng(0)
        a = rng.dirichlet(np.ones(4), size=20)
        b = rng.dirichlet(np.ones(4), size=20)
        assert np.allclose(blend(a, b, 0.3).sum(axis=1), 1.0)

    @pytest.mark.parametrize("alpha", [-0.1, 1.1, 2.0])
    def test_weight_outside_unit_interval_raises(self, alpha):
        with pytest.raises(ValueError, match="alpha"):
            blend(np.zeros((2, 4)), np.zeros((2, 4)), alpha)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="align"):
            blend(np.zeros((5, 4)), np.zeros((4, 4)), 0.5)


class TestScore:
    def test_perfect_prediction(self):
        y = np.array([0, 1, 2, 3])
        assert score(np.eye(4), y) == pytest.approx(1.0)

    def test_macro_and_weighted_differ_under_imbalance(self):
        y = np.array([0] * 10 + [1])
        probs = np.zeros((11, 4))
        probs[:, 0] = 1.0            # always predict the majority class
        assert score(probs, y, "weighted") > score(probs, y, "macro")


class TestTuneAlpha:
    def test_picks_the_informative_branch(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 4, size=200)
        good = np.eye(4)[y] * 0.9 + 0.025      # nearly perfect
        noise = rng.dirichlet(np.ones(4), size=200)
        alpha, f1 = tune_alpha(good, noise, y)
        # `>=` rather than `>`: when one branch dominates, every weight above some
        # point scores identically and the search returns the first of them.
        assert alpha >= 0.5, "the informative branch should get the larger weight"
        assert f1 > 0.9
        assert f1 > score(noise, y), "ensembling must beat the noise branch alone"

    def test_returns_a_weight_in_range(self, oof):
        alpha, f1 = tune_alpha(oof["resnet"], oof["uni"], oof["y"])
        assert 0.0 <= alpha <= 1.0
        assert 0.0 <= f1 <= 1.0

    def test_never_scores_below_either_endpoint(self, oof):
        """A search that returns something worse than not ensembling is broken."""
        _, best = tune_alpha(oof["resnet"], oof["uni"], oof["y"])
        assert best >= score(oof["resnet"], oof["y"]) - 1e-9
        assert best >= score(oof["uni"], oof["y"]) - 1e-9

    def test_refinement_never_hurts(self, oof):
        _, coarse = tune_alpha(oof["resnet"], oof["uni"], oof["y"], refine=False)
        _, refined = tune_alpha(oof["resnet"], oof["uni"], oof["y"], refine=True)
        assert refined >= coarse


class TestAlphaCurve:
    def test_shape_and_endpoints(self, oof):
        grid, scores = alpha_curve(oof["resnet"], oof["uni"], oof["y"])
        assert len(grid) == len(scores) == 101
        assert grid[0] == 0.0 and grid[-1] == 1.0
        assert scores[0] == pytest.approx(score(oof["uni"], oof["y"]))
        assert scores[-1] == pytest.approx(score(oof["resnet"], oof["y"]))
