"""Tests for pyzm.ml.backends.birdnet -- BirdNET audio bird recognition backend.

Updated for birdnet_analyzer v2.4 API:
- audio module (not audio_processing)
- model.predict() returns numpy arrays of raw logits
- model.flat_sigmoid() converts logits to probabilities
- model.predict_filter() for species list (not return_species_list)
- utils.ensure_model_exists() (not model.ensure_model_exists)
- cfg.LABELS loaded from file in load()
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

from pyzm.models.config import ModelConfig, ModelFramework, ModelType
from pyzm.models.detection import BBox, Detection


# ===================================================================
# Helpers
# ===================================================================

def _make_birdnet_config(**overrides) -> ModelConfig:
    """Create a ModelConfig for BirdNET with sensible test defaults."""
    defaults = dict(
        name="BirdNET",
        type=ModelType.AUDIO,
        framework=ModelFramework.BIRDNET,
        birdnet_lat=43.65,
        birdnet_lon=-79.38,
        birdnet_min_conf=0.5,
        birdnet_sensitivity=1.0,
        birdnet_overlap=0.0,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _make_birdnet_modules():
    """Create properly wired mock birdnet_analyzer modules.

    ``from birdnet_analyzer import config`` resolves via
    ``sys.modules["birdnet_analyzer"].config``, so the child mocks
    must be attributes on the parent module.

    Returns (parent_module, cfg_mock, audio_mock, model_mock, utils_mock, modules_dict).
    """
    mock_cfg = MagicMock()
    mock_audio = MagicMock()
    mock_model = MagicMock()
    mock_utils = MagicMock()

    parent = types.ModuleType("birdnet_analyzer")
    parent.config = mock_cfg
    parent.audio = mock_audio
    parent.model = mock_model
    parent.utils = mock_utils

    modules = {
        "birdnet_analyzer": parent,
        "birdnet_analyzer.config": mock_cfg,
        "birdnet_analyzer.audio": mock_audio,
        "birdnet_analyzer.model": mock_model,
        "birdnet_analyzer.utils": mock_utils,
    }
    return parent, mock_cfg, mock_audio, mock_model, mock_utils, modules


# ===================================================================
# TestBirdnetBackendInit
# ===================================================================

class TestBirdnetBackendInit:
    """Tests for BirdnetBackend construction and properties."""

    def test_name_from_config(self):
        from pyzm.ml.backends.birdnet import BirdnetBackend

        config = _make_birdnet_config(name="MyBirdNET")
        backend = BirdnetBackend(config)
        assert backend.name == "MyBirdNET"

    def test_name_defaults_to_birdnet(self):
        from pyzm.ml.backends.birdnet import BirdnetBackend

        config = _make_birdnet_config(name=None)
        backend = BirdnetBackend(config)
        assert backend.name == "BirdNET"

    def test_is_loaded_initially_false(self):
        from pyzm.ml.backends.birdnet import BirdnetBackend

        backend = BirdnetBackend(_make_birdnet_config())
        assert backend.is_loaded is False

    def test_detect_image_returns_empty(self):
        """Audio backend's detect(image) always returns empty list."""
        from pyzm.ml.backends.birdnet import BirdnetBackend

        backend = BirdnetBackend(_make_birdnet_config())
        result = backend.detect(MagicMock())
        assert result == []


# ===================================================================
# TestBirdnetBackendLoad
# ===================================================================

class TestBirdnetBackendLoad:
    """Tests for BirdnetBackend.load()."""

    def test_load_marks_loaded(self):
        _, mock_cfg, _, _, _, modules = _make_birdnet_modules()
        mock_cfg.LABELS_FILE = "/fake/labels.txt"
        with patch.dict("sys.modules", modules):
            with patch("builtins.open", mock_open(read_data="Sci_Robin\nSci_Sparrow\n")):
                from pyzm.ml.backends.birdnet import BirdnetBackend

                backend = BirdnetBackend(_make_birdnet_config())
                backend.load()
                assert backend.is_loaded is True

    def test_load_calls_ensure_model_exists(self):
        _, mock_cfg, _, _, mock_utils, modules = _make_birdnet_modules()
        mock_cfg.LABELS_FILE = "/fake/labels.txt"
        with patch.dict("sys.modules", modules):
            with patch("builtins.open", mock_open(read_data="Sci_Robin\n")):
                from pyzm.ml.backends.birdnet import BirdnetBackend

                backend = BirdnetBackend(_make_birdnet_config())
                backend.load()
                mock_utils.ensure_model_exists.assert_called_once()

    def test_load_sets_config_paths(self):
        """load() copies BIRDNET_* config paths to runtime config attributes."""
        _, mock_cfg, _, _, _, modules = _make_birdnet_modules()
        mock_cfg.BIRDNET_MODEL_PATH = "/models/BirdNET.tflite"
        mock_cfg.BIRDNET_LABELS_FILE = "/models/labels.txt"
        mock_cfg.BIRDNET_SAMPLE_RATE = 48000
        mock_cfg.BIRDNET_SIG_LENGTH = 3.0
        with patch.dict("sys.modules", modules):
            with patch("builtins.open", mock_open(read_data="Sci_Robin\n")):
                from pyzm.ml.backends.birdnet import BirdnetBackend

                backend = BirdnetBackend(_make_birdnet_config())
                backend.load()
                assert mock_cfg.MODEL_PATH == "/models/BirdNET.tflite"
                assert mock_cfg.LABELS_FILE == "/models/labels.txt"
                assert mock_cfg.SAMPLE_RATE == 48000
                assert mock_cfg.SIG_LENGTH == 3.0

    def test_load_reads_labels_from_file(self):
        """load() reads labels from cfg.LABELS_FILE into cfg.LABELS."""
        _, mock_cfg, _, _, _, modules = _make_birdnet_modules()
        mock_cfg.LABELS_FILE = "/fake/labels.txt"
        label_data = "Turdus migratorius_American Robin\nPasser domesticus_House Sparrow\n\n"
        with patch.dict("sys.modules", modules):
            with patch("builtins.open", mock_open(read_data=label_data)):
                from pyzm.ml.backends.birdnet import BirdnetBackend

                backend = BirdnetBackend(_make_birdnet_config())
                backend.load()
                assert mock_cfg.LABELS == [
                    "Turdus migratorius_American Robin",
                    "Passer domesticus_House Sparrow",
                ]

    def test_load_raises_if_birdnet_not_installed(self):
        """load() should raise ImportError with helpful message when birdnet_analyzer is missing."""
        from pyzm.ml.backends.birdnet import BirdnetBackend

        backend = BirdnetBackend(_make_birdnet_config())

        with patch.dict("sys.modules", {"birdnet_analyzer": None}):
            with pytest.raises(ImportError, match="birdnet_analyzer is not installed"):
                backend.load()


# ===================================================================
# TestBirdnetDetectAudio
# ===================================================================

class TestBirdnetDetectAudio:
    """Tests for BirdnetBackend.detect_audio()."""

    @patch("pyzm.ml.backends.birdnet.BirdnetBackend.load")
    def test_detect_audio_auto_loads(self, mock_load):
        """detect_audio() calls load() if not yet loaded."""
        from pyzm.ml.backends.birdnet import BirdnetBackend

        _, mock_cfg, mock_audio, mock_model, _, modules = _make_birdnet_modules()
        mock_audio.open_audio_file.return_value = (np.zeros(48000), 48000)
        mock_audio.split_signal.return_value = []
        mock_model.predict_filter.return_value = []

        backend = BirdnetBackend(_make_birdnet_config())
        assert backend.is_loaded is False

        with patch.dict("sys.modules", modules):
            backend.detect_audio("/tmp/test.wav")

        mock_load.assert_called_once()

    def test_detect_audio_returns_detections_for_species(self):
        """detect_audio returns Detection objects for each detected species."""
        from pyzm.ml.backends.birdnet import BirdnetBackend

        _, mock_cfg, mock_audio, mock_model, _, modules = _make_birdnet_modules()

        mock_cfg.LABELS = [
            "Turdus migratorius_American Robin",
            "Passer domesticus_House Sparrow",
            "Cardinalis cardinalis_Northern Cardinal",
        ]

        mock_audio.open_audio_file.return_value = (np.zeros(48000), 48000)
        mock_audio.split_signal.return_value = [np.zeros(48000), np.zeros(48000)]

        # model.predict returns arrays of raw logits, flat_sigmoid converts to scores
        mock_model.predict.side_effect = [
            np.array([np.zeros(3)]),
            np.array([np.zeros(3)]),
        ]
        mock_model.flat_sigmoid.side_effect = [
            np.array([0.7, 0.6, 0.0]),   # chunk 1: Robin=0.7, Sparrow=0.6
            np.array([0.9, 0.0, 0.8]),   # chunk 2: Robin=0.9, Cardinal=0.8
        ]
        mock_model.predict_filter.return_value = ["species1", "species2"]

        config = _make_birdnet_config(birdnet_min_conf=0.5)
        backend = BirdnetBackend(config)
        backend._loaded = True

        with patch.dict("sys.modules", modules):
            detections = backend.detect_audio("/tmp/test.wav", event_week=20, monitor_lat=43.0, monitor_lon=-79.0)

        assert len(detections) == 3
        labels = {d.label for d in detections}
        assert labels == {"American Robin", "House Sparrow", "Northern Cardinal"}

        for d in detections:
            assert d.detection_type == "audio"
            assert d.bbox == BBox(x1=0, y1=0, x2=1, y2=1)
            assert d.model_name == "BirdNET"

    def test_detect_audio_keeps_best_confidence_per_species(self):
        """Across chunks, the highest confidence per species is kept."""
        from pyzm.ml.backends.birdnet import BirdnetBackend

        _, mock_cfg, mock_audio, mock_model, _, modules = _make_birdnet_modules()

        mock_cfg.LABELS = ["Turdus migratorius_American Robin"]

        mock_audio.open_audio_file.return_value = (np.zeros(48000), 48000)
        mock_audio.split_signal.return_value = [np.zeros(48000), np.zeros(48000)]

        mock_model.predict.side_effect = [
            np.array([np.zeros(1)]),
            np.array([np.zeros(1)]),
        ]
        mock_model.flat_sigmoid.side_effect = [
            np.array([0.7]),   # chunk 1: Robin=0.7
            np.array([0.9]),   # chunk 2: Robin=0.9
        ]
        mock_model.predict_filter.return_value = []

        config = _make_birdnet_config(birdnet_min_conf=0.5, birdnet_lat=-1.0, birdnet_lon=-1.0)
        backend = BirdnetBackend(config)
        backend._loaded = True

        with patch.dict("sys.modules", modules):
            detections = backend.detect_audio("/tmp/test.wav")

        assert len(detections) == 1
        assert detections[0].label == "American Robin"
        assert detections[0].confidence == 0.9

    def test_detect_audio_filters_below_min_confidence(self):
        """Species below birdnet_min_conf are excluded."""
        from pyzm.ml.backends.birdnet import BirdnetBackend

        _, mock_cfg, mock_audio, mock_model, _, modules = _make_birdnet_modules()

        mock_cfg.LABELS = [
            "Turdus migratorius_American Robin",
            "Passer domesticus_House Sparrow",
        ]

        mock_audio.open_audio_file.return_value = (np.zeros(48000), 48000)
        mock_audio.split_signal.return_value = [np.zeros(48000)]

        mock_model.predict.side_effect = [np.array([np.zeros(2)])]
        mock_model.flat_sigmoid.side_effect = [
            np.array([0.8, 0.3]),  # Robin above threshold, Sparrow below
        ]
        mock_model.predict_filter.return_value = []

        config = _make_birdnet_config(birdnet_min_conf=0.5, birdnet_lat=-1.0, birdnet_lon=-1.0)
        backend = BirdnetBackend(config)
        backend._loaded = True

        with patch.dict("sys.modules", modules):
            detections = backend.detect_audio("/tmp/test.wav")

        assert len(detections) == 1
        assert detections[0].label == "American Robin"

    def test_detect_audio_empty_when_no_species_found(self):
        """Returns empty list when no species meet the threshold."""
        from pyzm.ml.backends.birdnet import BirdnetBackend

        _, mock_cfg, mock_audio, mock_model, _, modules = _make_birdnet_modules()

        mock_cfg.LABELS = ["Passer domesticus_House Sparrow"]

        mock_audio.open_audio_file.return_value = (np.zeros(48000), 48000)
        mock_audio.split_signal.return_value = [np.zeros(48000)]

        mock_model.predict.side_effect = [np.array([np.zeros(1)])]
        mock_model.flat_sigmoid.side_effect = [np.array([0.1])]
        mock_model.predict_filter.return_value = []

        config = _make_birdnet_config(birdnet_min_conf=0.5, birdnet_lat=-1.0, birdnet_lon=-1.0)
        backend = BirdnetBackend(config)
        backend._loaded = True

        with patch.dict("sys.modules", modules):
            detections = backend.detect_audio("/tmp/test.wav")

        assert detections == []

    def test_detect_audio_empty_when_no_chunks(self):
        """Returns empty list when the audio file has no chunks."""
        from pyzm.ml.backends.birdnet import BirdnetBackend

        _, mock_cfg, mock_audio, mock_model, _, modules = _make_birdnet_modules()

        mock_audio.open_audio_file.return_value = (np.zeros(48000), 48000)
        mock_audio.split_signal.return_value = []
        mock_model.predict_filter.return_value = []

        backend = BirdnetBackend(_make_birdnet_config())
        backend._loaded = True

        with patch.dict("sys.modules", modules):
            detections = backend.detect_audio("/tmp/silent.wav")

        assert detections == []

    def test_detect_audio_handles_none_predictions(self):
        """Gracefully handles None predictions from model.predict()."""
        from pyzm.ml.backends.birdnet import BirdnetBackend

        _, mock_cfg, mock_audio, mock_model, _, modules = _make_birdnet_modules()

        mock_cfg.LABELS = ["Sci_Robin"]

        mock_audio.open_audio_file.return_value = (np.zeros(48000), 48000)
        mock_audio.split_signal.return_value = [np.zeros(48000), np.zeros(48000)]

        mock_model.predict.side_effect = [
            None,                         # chunk 1: None (skipped)
            np.array([np.zeros(1)]),      # chunk 2: valid
        ]
        mock_model.flat_sigmoid.side_effect = [np.array([0.8])]
        mock_model.predict_filter.return_value = []

        config = _make_birdnet_config(birdnet_min_conf=0.5, birdnet_lat=-1.0, birdnet_lon=-1.0)
        backend = BirdnetBackend(config)
        backend._loaded = True

        with patch.dict("sys.modules", modules):
            detections = backend.detect_audio("/tmp/test.wav")

        assert len(detections) == 1
        assert detections[0].label == "Robin"

    def test_detect_audio_handles_label_without_underscore(self):
        """Labels without underscore separator use the whole string as common name."""
        from pyzm.ml.backends.birdnet import BirdnetBackend

        _, mock_cfg, mock_audio, mock_model, _, modules = _make_birdnet_modules()

        mock_cfg.LABELS = ["UnknownBird"]

        mock_audio.open_audio_file.return_value = (np.zeros(48000), 48000)
        mock_audio.split_signal.return_value = [np.zeros(48000)]

        mock_model.predict.side_effect = [np.array([np.zeros(1)])]
        mock_model.flat_sigmoid.side_effect = [np.array([0.9])]
        mock_model.predict_filter.return_value = []

        config = _make_birdnet_config(birdnet_min_conf=0.5, birdnet_lat=-1.0, birdnet_lon=-1.0)
        backend = BirdnetBackend(config)
        backend._loaded = True

        with patch.dict("sys.modules", modules):
            detections = backend.detect_audio("/tmp/test.wav")

        assert len(detections) == 1
        assert detections[0].label == "UnknownBird"

    def test_detect_audio_results_sorted_by_confidence_descending(self):
        """Detections are sorted from highest to lowest confidence."""
        from pyzm.ml.backends.birdnet import BirdnetBackend

        _, mock_cfg, mock_audio, mock_model, _, modules = _make_birdnet_modules()

        mock_cfg.LABELS = ["Sci_Sparrow", "Sci_Robin", "Sci_Cardinal"]

        mock_audio.open_audio_file.return_value = (np.zeros(48000), 48000)
        mock_audio.split_signal.return_value = [np.zeros(48000)]

        mock_model.predict.side_effect = [np.array([np.zeros(3)])]
        mock_model.flat_sigmoid.side_effect = [
            np.array([0.6, 0.9, 0.75]),  # Sparrow=0.6, Robin=0.9, Cardinal=0.75
        ]
        mock_model.predict_filter.return_value = []

        config = _make_birdnet_config(birdnet_min_conf=0.5, birdnet_lat=-1.0, birdnet_lon=-1.0)
        backend = BirdnetBackend(config)
        backend._loaded = True

        with patch.dict("sys.modules", modules):
            detections = backend.detect_audio("/tmp/test.wav")

        confs = [d.confidence for d in detections]
        assert confs == sorted(confs, reverse=True)
        assert detections[0].label == "Robin"
        assert detections[1].label == "Cardinal"
        assert detections[2].label == "Sparrow"


# ===================================================================
# TestBirdnetLatLonFallback
# ===================================================================

class TestBirdnetLatLonFallback:
    """Tests for lat/lon config -> monitor DB fallback logic."""

    def _run_detect_audio(self, config, monitor_lat=-1.0, monitor_lon=-1.0):
        """Helper to run detect_audio and return the birdnet config mock."""
        from pyzm.ml.backends.birdnet import BirdnetBackend

        _, mock_cfg, mock_audio, mock_model, _, modules = _make_birdnet_modules()
        mock_audio.open_audio_file.return_value = (np.zeros(48000), 48000)
        mock_audio.split_signal.return_value = []
        mock_model.predict_filter.return_value = []

        backend = BirdnetBackend(config)
        backend._loaded = True

        with patch.dict("sys.modules", modules):
            backend.detect_audio(
                "/tmp/test.wav",
                event_week=20,
                monitor_lat=monitor_lat,
                monitor_lon=monitor_lon,
            )

        return mock_cfg, mock_model

    def test_config_latlon_used_when_set(self):
        """Config lat/lon take precedence over monitor lat/lon."""
        config = _make_birdnet_config(birdnet_lat=43.65, birdnet_lon=-79.38)
        mock_cfg, _ = self._run_detect_audio(config, monitor_lat=51.0, monitor_lon=-0.1)

        assert mock_cfg.LATITUDE == 43.65
        assert mock_cfg.LONGITUDE == -79.38

    def test_monitor_latlon_used_when_config_is_negative(self):
        """Monitor lat/lon used as fallback when config lat/lon are -1."""
        config = _make_birdnet_config(birdnet_lat=-1.0, birdnet_lon=-1.0)
        mock_cfg, _ = self._run_detect_audio(config, monitor_lat=51.0, monitor_lon=-0.1)

        assert mock_cfg.LATITUDE == 51.0
        assert mock_cfg.LONGITUDE == -0.1

    def test_both_negative_stays_negative(self):
        """When both config and monitor are -1, lat/lon stay -1."""
        config = _make_birdnet_config(birdnet_lat=-1.0, birdnet_lon=-1.0)
        mock_cfg, _ = self._run_detect_audio(config)

        assert mock_cfg.LATITUDE == -1.0
        assert mock_cfg.LONGITUDE == -1.0

    def test_partial_fallback_lat_only(self):
        """Config lat set but lon -1: lat from config, lon from monitor."""
        config = _make_birdnet_config(birdnet_lat=43.65, birdnet_lon=-1.0)
        mock_cfg, _ = self._run_detect_audio(config, monitor_lat=51.0, monitor_lon=-0.1)

        assert mock_cfg.LATITUDE == 43.65
        assert mock_cfg.LONGITUDE == -0.1

    def test_species_filter_called_when_latlon_set(self):
        """predict_filter is called when valid lat/lon are available."""
        config = _make_birdnet_config(birdnet_lat=43.65, birdnet_lon=-79.38)
        _, mock_model = self._run_detect_audio(config)

        mock_model.predict_filter.assert_called_once_with(43.65, -79.38, 20)

    def test_species_filter_not_called_when_no_latlon(self):
        """predict_filter is not called when lat/lon are -1."""
        config = _make_birdnet_config(birdnet_lat=-1.0, birdnet_lon=-1.0)
        _, mock_model = self._run_detect_audio(config)

        mock_model.predict_filter.assert_not_called()


# ===================================================================
# TestBirdnetConfigGlobals
# ===================================================================

class TestBirdnetConfigGlobals:
    """Tests verifying that BirdNET config globals are set correctly."""

    def _run_and_get_cfg(self, config, event_week=-1):
        from pyzm.ml.backends.birdnet import BirdnetBackend

        _, mock_cfg, mock_audio, mock_model, _, modules = _make_birdnet_modules()

        mock_audio.open_audio_file.return_value = (np.zeros(48000), 48000)
        mock_audio.split_signal.return_value = []
        mock_model.predict_filter.return_value = []

        backend = BirdnetBackend(config)
        backend._loaded = True

        with patch.dict("sys.modules", modules):
            backend.detect_audio("/tmp/test.wav", event_week=event_week)

        return mock_cfg

    def test_all_config_fields_set(self):
        """All birdnet_analyzer config fields are set from ModelConfig."""
        config = _make_birdnet_config(
            birdnet_lat=43.65,
            birdnet_lon=-79.38,
            birdnet_min_conf=0.7,
            birdnet_sensitivity=1.5,
            birdnet_overlap=0.5,
        )
        mock_cfg = self._run_and_get_cfg(config, event_week=25)

        assert mock_cfg.LATITUDE == 43.65
        assert mock_cfg.LONGITUDE == -79.38
        assert mock_cfg.WEEK == 25
        assert mock_cfg.MIN_CONFIDENCE == 0.7
        assert mock_cfg.SIGMOID_SENSITIVITY == 1.5
        assert mock_cfg.SIG_OVERLAP == 0.5

    def test_negative_week_passed_through(self):
        """When event_week is -1, WEEK is set to -1."""
        config = _make_birdnet_config(birdnet_lat=43.65, birdnet_lon=-79.38)
        mock_cfg = self._run_and_get_cfg(config, event_week=-1)
        assert mock_cfg.WEEK == -1

    def test_zero_week_treated_as_negative(self):
        """Week 0 is invalid, should be treated like -1."""
        config = _make_birdnet_config(birdnet_lat=43.65, birdnet_lon=-79.38)
        mock_cfg = self._run_and_get_cfg(config, event_week=0)
        assert mock_cfg.WEEK == -1
