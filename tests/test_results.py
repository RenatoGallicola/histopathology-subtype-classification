"""The committed artifacts, the submission, and the numbers quoted in the docs.

These pin the claims the documentation makes: if an artifact or a config value
changes so that a documented figure moves, this fails.
"""
import importlib.util
import sys

import numpy as np
import pandas as pd
import pytest

from src.config import CLASS_NAMES, ENSEMBLE, IDX_TO_LABEL, LABEL_TO_IDX
from src.ensemble import blend, score, tune_alpha
from src.submission import build_submission, check_submission

N_TRAIN_SLIDES = 627
N_TEST_SLIDES = 477


class TestLabelMap:
    def test_round_trips(self):
        for i, name in IDX_TO_LABEL.items():
            assert LABEL_TO_IDX[name] == i

    def test_matches_the_submission_vocabulary(self):
        assert set(CLASS_NAMES) == {"Luminal A", "Luminal B", "HER2(+)",
                                    "Triple negative"}


class TestArtifacts:
    def test_shapes(self, oof):
        assert oof["resnet"].shape == (N_TRAIN_SLIDES, 4)
        assert oof["uni"].shape == (N_TRAIN_SLIDES, 4)
        assert oof["y"].shape == (N_TRAIN_SLIDES,)
        assert oof["slide_ids"].shape == (N_TRAIN_SLIDES,)

    def test_probabilities_are_finite_and_normalised(self, oof):
        for name in ("resnet", "uni"):
            probs = oof[name]
            assert np.isfinite(probs).all(), name
            assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5), name
            assert (probs >= 0).all(), name

    def test_slide_ids_unique_and_well_formed(self, oof):
        ids = oof["slide_ids"]
        assert len(set(ids.tolist())) == N_TRAIN_SLIDES
        assert all(str(s).startswith("img_") and str(s).endswith(".png") for s in ids)

    def test_class_distribution(self, oof):
        assert list(np.bincount(oof["y"], minlength=4)) == [174, 219, 158, 76]


class TestDocumentedResults:
    """Values quoted in README.md and docs/RESULTS.md."""

    @pytest.mark.parametrize("branch,expected", [("resnet", 0.4248), ("uni", 0.4723)])
    def test_single_branch_macro_f1(self, oof, branch, expected):
        assert score(oof[branch], oof["y"]) == pytest.approx(expected, abs=5e-5)

    @pytest.mark.parametrize("alpha,expected", [(0.04, 0.4731), (0.20, 0.4637)])
    def test_ensemble_macro_f1(self, oof, alpha, expected):
        probs = blend(oof["resnet"], oof["uni"], alpha)
        assert score(probs, oof["y"]) == pytest.approx(expected, abs=5e-5)

    def test_cross_validated_optimum(self, oof):
        alpha, f1 = tune_alpha(oof["resnet"], oof["uni"], oof["y"])
        assert alpha == pytest.approx(0.05, abs=1e-9)
        assert f1 == pytest.approx(0.4747, abs=5e-5)

    def test_uni_branch_is_the_stronger_one(self, oof):
        """The central claim of the write-up."""
        assert score(oof["uni"], oof["y"]) > score(oof["resnet"], oof["y"])

    def test_optimum_favours_uni(self, oof):
        alpha, _ = tune_alpha(oof["resnet"], oof["uni"], oof["y"])
        assert alpha < 0.5, "the search should put most of the weight on UNI"

    def test_majority_baseline_quoted_in_data_readme(self, oof):
        majority = np.full(len(oof["y"]), np.bincount(oof["y"]).argmax())
        from sklearn.metrics import accuracy_score, f1_score
        assert accuracy_score(oof["y"], majority) == pytest.approx(0.349, abs=5e-4)
        assert f1_score(oof["y"], majority, average="macro",
                        zero_division=0) == pytest.approx(0.129, abs=5e-4)


class TestSubmissions:
    def test_final_submission_is_valid(self, repo_root):
        frame = pd.read_csv(repo_root / "submissions" / "submission_final.csv")
        check_submission(frame)

    def test_final_submission_distribution(self, repo_root):
        frame = pd.read_csv(repo_root / "submissions" / "submission_final.csv")
        assert frame["label"].value_counts().to_dict() == {
            "Luminal B": 192, "HER2(+)": 135, "Luminal A": 108,
            "Triple negative": 42}

    def test_every_submission_is_valid(self, repo_root):
        for path in sorted((repo_root / "submissions").glob("*.csv")):
            check_submission(pd.read_csv(path))

    def test_final_and_five_fold_mostly_agree(self, repo_root):
        """Documented in docs/RESULTS.md as 95.8%."""
        a = pd.read_csv(repo_root / "submissions" / "submission_final.csv")
        b = pd.read_csv(repo_root / "submissions" / "submission_5fold_alpha004.csv")
        merged = a.merge(b, on="sample_index", suffixes=("_a", "_b"))
        agreement = (merged["label_a"] == merged["label_b"]).mean()
        assert agreement == pytest.approx(0.958, abs=0.002)


class TestBuildSubmission:
    def test_maps_argmax_to_label_names(self):
        frame = build_submission(["img_0001.png", "img_0000.png"], np.eye(4)[[2, 0]])
        assert list(frame["sample_index"]) == ["img_0000.png", "img_0001.png"]
        assert list(frame["label"]) == ["Luminal A", "HER2(+)"]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="slide ids"):
            build_submission(["img_0000.png"], np.eye(4)[[0, 1]])

    def test_rejects_wrong_row_count(self):
        frame = build_submission(["img_0000.png"], np.eye(4)[[0]])
        with pytest.raises(ValueError, match="rows"):
            check_submission(frame)

    def test_rejects_malformed_ids(self):
        ids = [f"img_{i:04d}.png" for i in range(N_TEST_SLIDES)]
        ids[0] = "slide-1.png"
        frame = build_submission(ids, np.eye(4)[[0] * N_TEST_SLIDES])
        with pytest.raises(ValueError, match="malformed"):
            check_submission(frame)

    def test_rejects_a_gap_in_the_index_range(self):
        ids = [f"img_{i:04d}.png" for i in range(N_TEST_SLIDES)]
        ids[5] = "img_9999.png"
        frame = build_submission(ids, np.eye(4)[[0] * N_TEST_SLIDES])
        with pytest.raises(ValueError, match="index range"):
            check_submission(frame)


def load_script(repo_root, name):
    """Import a script from `scripts/` and return the module.

    Imported and called in-process rather than spawned: a child process racing
    the parent over the same working directory makes these tests flaky on
    Windows. CI still runs both scripts as real commands.
    """
    path = repo_root / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestScripts:
    def test_ablation_reports_the_documented_scores(self, repo_root, capsys,
                                                    monkeypatch):
        monkeypatch.chdir(repo_root)
        exit_code = load_script(repo_root, "ablation.py").main()
        out = capsys.readouterr().out
        assert exit_code == 0, out
        assert "0.4723" in out and "0.4248" in out, out

    def test_tune_alpha_reports_the_optimum(self, repo_root, capsys, monkeypatch):
        monkeypatch.chdir(repo_root)
        exit_code = load_script(repo_root, "tune_alpha.py").main()
        out = capsys.readouterr().out
        assert exit_code == 0, out
        assert "0.4747" in out, out


class TestConfig:
    def test_shipped_and_cross_validated_alphas_are_both_recorded(self):
        assert 0.0 <= ENSEMBLE.alpha_cv <= 1.0
        assert 0.0 <= ENSEMBLE.alpha_shipped <= 1.0
        assert ENSEMBLE.alpha_shipped > ENSEMBLE.alpha_cv

    def test_dihedral_group_has_eight_views(self):
        n = len(ENSEMBLE.tta_rotations) * (2 if ENSEMBLE.tta_flip else 1)
        assert n == 8
