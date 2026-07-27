"""Slide indexing, class weights, tile pooling and dihedral TTA."""
import numpy as np
import pytest

from src.data import build_slide_index, class_weights, encode_labels, slide_labels
from src.inference import pool_slide_probabilities


class TestSlideIndex:
    def test_groups_tiles_by_slide(self):
        ids = np.array(["a.png", "b.png", "a.png", "c.png", "a.png"])
        index = build_slide_index(ids)
        assert set(index) == {"a.png", "b.png", "c.png"}
        assert list(index["a.png"]) == [0, 2, 4]

    def test_every_tile_is_indexed_exactly_once(self):
        rng = np.random.default_rng(0)
        ids = rng.choice(["s1", "s2", "s3"], size=50)
        index = build_slide_index(ids)
        assert sorted(np.concatenate(list(index.values())).tolist()) == list(range(50))

    def test_slide_label_comes_from_its_tiles(self):
        ids = np.array(["a", "a", "b"])
        index = build_slide_index(ids)
        assert slide_labels(index, np.array([2, 2, 3])) == {"a": 2, "b": 3}


class TestEncodeLabels:
    def test_maps_the_four_subtypes(self):
        got = encode_labels(["Luminal A", "Luminal B", "HER2(+)", "Triple negative"])
        assert list(got) == [0, 1, 2, 3]

    def test_unknown_label_raises(self):
        with pytest.raises(KeyError):
            encode_labels(["Luminal C"])


class TestClassWeights:
    def test_rare_classes_weigh_more(self):
        w = class_weights(np.array([0] * 174 + [1] * 219 + [2] * 158 + [3] * 76))
        assert w[3] > w[0] > w[1]      # 76 < 174 < 219

    def test_balanced_data_gives_equal_weights(self):
        assert np.allclose(class_weights(np.array([0, 1, 2, 3] * 10)), 1.0)

    def test_normalised_to_mean_one(self):
        w = class_weights(np.array([0] * 10 + [1] * 5 + [2] * 2 + [3]))
        assert w.mean() == pytest.approx(1.0)


class TestPooling:
    def test_averages_tiles_per_slide(self):
        probs = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]])
        ids = np.array(["a", "a", "b"])
        slides, pooled = pool_slide_probabilities(probs, ids)
        assert list(slides) == ["a", "b"]
        assert np.allclose(pooled[0], [0.5, 0.5, 0, 0])
        assert np.allclose(pooled[1], [0, 0, 1.0, 0])

    def test_output_is_sorted_by_slide_id(self):
        probs = np.ones((3, 4)) / 4
        slides, _ = pool_slide_probabilities(probs, np.array(["c", "a", "b"]))
        assert list(slides) == ["a", "b", "c"]

    def test_pooled_probabilities_still_sum_to_one(self):
        rng = np.random.default_rng(1)
        probs = rng.dirichlet(np.ones(4), size=30)
        ids = rng.choice(["s1", "s2"], size=30)
        _, pooled = pool_slide_probabilities(probs, ids)
        assert np.allclose(pooled.sum(axis=1), 1.0)

    def test_mean_not_max(self):
        """One confident tile must not decide the slide."""
        probs = np.array([[0.99, 0.01, 0, 0]] + [[0, 1.0, 0, 0]] * 5)
        ids = np.array(["s"] * 6)
        _, pooled = pool_slide_probabilities(probs, ids)
        assert pooled[0].argmax() == 1, "the majority of tiles should win"


torch = pytest.importorskip("torch")


class TestDihedralTTA:
    """The augmentation is only valid because the group is label-preserving on
    tissue, so the natural test is that an invariant model is unmoved by it."""

    def test_invariant_model_gives_identical_probabilities(self):
        from torch import nn

        from src.data import make_tile_dataset
        from src.inference import predict_tiles, predict_tiles_dihedral_tta

        class MeanColour(nn.Module):
            """Rotation- and flip-invariant by construction: mean over space."""
            def forward(self, x):
                return x.mean(dim=(2, 3))[:, :4] if x.shape[1] >= 4 else \
                    torch.cat([x.mean(dim=(2, 3)), x.mean(dim=(2, 3))[:, :1]], dim=1)

        rng = np.random.default_rng(0)
        tiles = rng.integers(0, 255, size=(4, 16, 16, 3), dtype=np.uint8)
        ds = make_tile_dataset(tiles, None, None, train_aug=False)
        model, device = MeanColour(), torch.device("cpu")

        plain = predict_tiles(model, ds, device, batch_size=2)
        tta = predict_tiles_dihedral_tta(model, ds, device, batch_size=2)
        assert np.allclose(plain, tta, atol=1e-5)

    def test_tta_returns_probabilities(self):
        from torch import nn

        from src.data import make_tile_dataset
        from src.inference import predict_tiles_dihedral_tta

        rng = np.random.default_rng(0)
        tiles = rng.integers(0, 255, size=(3, 16, 16, 3), dtype=np.uint8)
        ds = make_tile_dataset(tiles, None, None, train_aug=False)
        model = nn.Sequential(nn.Flatten(), nn.Linear(16 * 16 * 3, 4))
        out = predict_tiles_dihedral_tta(model, ds, torch.device("cpu"), batch_size=2)
        assert out.shape == (3, 4)
        assert np.allclose(out.sum(axis=1), 1.0)


class TestTileDataset:
    def test_normalises_and_reorders_to_chw(self):
        from src.data import make_tile_dataset

        tiles = np.full((2, 8, 8, 3), 255, dtype=np.uint8)
        ds = make_tile_dataset(tiles, None, None, train_aug=False)
        x = ds[0]
        assert x.shape == (3, 8, 8)
        # a saturated white tile maps to (1 - mean) / std per channel
        expected = (1.0 - 0.485) / 0.229
        assert x[0].mean().item() == pytest.approx(expected, abs=1e-4)

    def test_returns_label_when_given_one(self):
        from src.data import make_tile_dataset

        tiles = np.zeros((2, 8, 8, 3), dtype=np.uint8)
        ds = make_tile_dataset(tiles, np.array([1, 3]), None, train_aug=False)
        x, y = ds[1]
        assert int(y) == 3 and x.shape == (3, 8, 8)

    def test_subset_selects_the_requested_tiles(self):
        from src.data import make_tile_dataset

        tiles = np.zeros((5, 8, 8, 3), dtype=np.uint8)
        ds = make_tile_dataset(tiles, np.arange(5), idxs=[1, 3], train_aug=False)
        assert len(ds) == 2
        assert int(ds[0][1]) == 1 and int(ds[1][1]) == 3
