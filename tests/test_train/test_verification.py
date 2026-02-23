"""Tests for pyzm.train.verification."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyzm.train.dataset import Annotation
from pyzm.train.verification import (
    DetectionStatus,
    ImageVerification,
    VerificationStore,
    VerifiedDetection,
)


# ---------------------------------------------------------------------------
# VerifiedDetection
# ---------------------------------------------------------------------------

class TestVerifiedDetection:
    def _make(self, **kw) -> VerifiedDetection:
        defaults = dict(
            detection_id="det_0",
            original=Annotation(class_id=0, cx=0.5, cy=0.5, w=0.3, h=0.4),
            original_label="person",
        )
        defaults.update(kw)
        return VerifiedDetection(**defaults)

    def test_effective_label_default(self):
        d = self._make()
        assert d.effective_label == "person"

    def test_effective_label_renamed(self):
        d = self._make(status=DetectionStatus.RENAMED, new_label="pedestrian")
        assert d.effective_label == "pedestrian"

    def test_effective_label_renamed_no_label_falls_back(self):
        d = self._make(status=DetectionStatus.RENAMED, new_label=None)
        assert d.effective_label == "person"

    def test_effective_annotation_default(self):
        d = self._make()
        assert d.effective_annotation is d.original

    def test_effective_annotation_reshaped(self):
        adj = Annotation(class_id=0, cx=0.6, cy=0.6, w=0.2, h=0.3)
        d = self._make(status=DetectionStatus.RESHAPED, adjusted=adj)
        assert d.effective_annotation is adj

    def test_effective_annotation_reshaped_no_adj_falls_back(self):
        d = self._make(status=DetectionStatus.RESHAPED, adjusted=None)
        assert d.effective_annotation is d.original

    def test_is_active_pending(self):
        d = self._make(status=DetectionStatus.PENDING)
        assert not d.is_active

    def test_is_active_deleted(self):
        d = self._make(status=DetectionStatus.DELETED)
        assert not d.is_active

    def test_is_active_approved(self):
        d = self._make(status=DetectionStatus.APPROVED)
        assert d.is_active

    def test_is_active_renamed(self):
        d = self._make(status=DetectionStatus.RENAMED, new_label="cat")
        assert d.is_active

    def test_is_active_reshaped(self):
        d = self._make(status=DetectionStatus.RESHAPED)
        assert d.is_active

    def test_is_active_added(self):
        d = self._make(status=DetectionStatus.ADDED)
        assert d.is_active

    def test_to_dict_minimal(self):
        d = self._make()
        result = d.to_dict()
        assert result["detection_id"] == "det_0"
        assert result["status"] == "pending"
        assert result["original"]["cx"] == pytest.approx(0.5)
        assert "new_label" not in result
        assert "adjusted" not in result

    def test_to_dict_with_rename(self):
        d = self._make(status=DetectionStatus.RENAMED, new_label="cat")
        result = d.to_dict()
        assert result["new_label"] == "cat"

    def test_to_dict_with_adjusted(self):
        adj = Annotation(class_id=1, cx=0.7, cy=0.8, w=0.1, h=0.2)
        d = self._make(status=DetectionStatus.RESHAPED, adjusted=adj)
        result = d.to_dict()
        assert result["adjusted"]["cx"] == pytest.approx(0.7)

    def test_roundtrip(self):
        adj = Annotation(class_id=1, cx=0.7, cy=0.8, w=0.1, h=0.2)
        orig = self._make(
            status=DetectionStatus.RESHAPED,
            new_label="dog",
            adjusted=adj,
        )
        rebuilt = VerifiedDetection.from_dict(orig.to_dict())
        assert rebuilt.detection_id == orig.detection_id
        assert rebuilt.status == orig.status
        assert rebuilt.new_label == orig.new_label
        assert rebuilt.original_label == orig.original_label
        assert rebuilt.original.cx == pytest.approx(orig.original.cx)
        assert rebuilt.adjusted is not None
        assert rebuilt.adjusted.cx == pytest.approx(adj.cx)


# ---------------------------------------------------------------------------
# ImageVerification
# ---------------------------------------------------------------------------

class TestImageVerification:
    def test_empty(self):
        iv = ImageVerification(image_name="test.jpg")
        assert iv.pending_count == 0
        assert iv.active_detections == []

    def test_pending_count(self):
        dets = [
            VerifiedDetection("d0", Annotation(0, .5, .5, .3, .3), DetectionStatus.PENDING, original_label="a"),
            VerifiedDetection("d1", Annotation(0, .5, .5, .3, .3), DetectionStatus.APPROVED, original_label="b"),
            VerifiedDetection("d2", Annotation(0, .5, .5, .3, .3), DetectionStatus.PENDING, original_label="c"),
        ]
        iv = ImageVerification(image_name="test.jpg", detections=dets)
        assert iv.pending_count == 2

    def test_active_detections(self):
        dets = [
            VerifiedDetection("d0", Annotation(0, .5, .5, .3, .3), DetectionStatus.APPROVED, original_label="a"),
            VerifiedDetection("d1", Annotation(0, .5, .5, .3, .3), DetectionStatus.DELETED, original_label="b"),
            VerifiedDetection("d2", Annotation(0, .5, .5, .3, .3), DetectionStatus.ADDED, original_label="c"),
        ]
        iv = ImageVerification(image_name="test.jpg", detections=dets)
        active = iv.active_detections
        assert len(active) == 2
        assert active[0].detection_id == "d0"
        assert active[1].detection_id == "d2"

    def test_roundtrip(self):
        dets = [
            VerifiedDetection("d0", Annotation(0, .5, .5, .3, .3), DetectionStatus.APPROVED, original_label="x"),
        ]
        iv = ImageVerification(image_name="img.jpg", detections=dets, fully_reviewed=True)
        rebuilt = ImageVerification.from_dict(iv.to_dict())
        assert rebuilt.image_name == "img.jpg"
        assert rebuilt.fully_reviewed is True
        assert len(rebuilt.detections) == 1
        assert rebuilt.detections[0].detection_id == "d0"


# ---------------------------------------------------------------------------
# VerificationStore
# ---------------------------------------------------------------------------

class TestVerificationStore:
    def test_empty_store(self, tmp_path: Path):
        store = VerificationStore(tmp_path)
        assert store.all_images() == []
        assert store.pending_count() == 0
        assert store.reviewed_images_count() == 0

    def test_set_and_get(self, tmp_path: Path):
        store = VerificationStore(tmp_path)
        iv = ImageVerification(image_name="a.jpg")
        store.set(iv)
        assert store.get("a.jpg") is iv
        assert store.get("nonexistent.jpg") is None

    def test_save_and_reload(self, tmp_path: Path):
        store = VerificationStore(tmp_path)
        det = VerifiedDetection(
            "d0", Annotation(0, .5, .5, .3, .3),
            DetectionStatus.APPROVED, original_label="person",
        )
        iv = ImageVerification("img.jpg", detections=[det], fully_reviewed=True)
        store.set(iv)
        store.save()

        store2 = VerificationStore(tmp_path)
        loaded = store2.get("img.jpg")
        assert loaded is not None
        assert loaded.fully_reviewed is True
        assert len(loaded.detections) == 1
        assert loaded.detections[0].original_label == "person"

    def test_pending_count(self, tmp_path: Path):
        store = VerificationStore(tmp_path)
        dets1 = [
            VerifiedDetection("d0", Annotation(0, .5, .5, .3, .3), DetectionStatus.PENDING, original_label="a"),
            VerifiedDetection("d1", Annotation(0, .5, .5, .3, .3), DetectionStatus.APPROVED, original_label="b"),
        ]
        dets2 = [
            VerifiedDetection("d0", Annotation(0, .5, .5, .3, .3), DetectionStatus.PENDING, original_label="c"),
        ]
        store.set(ImageVerification("a.jpg", detections=dets1))
        store.set(ImageVerification("b.jpg", detections=dets2))
        assert store.pending_count() == 2

    def test_reviewed_images_count(self, tmp_path: Path):
        store = VerificationStore(tmp_path)
        store.set(ImageVerification("a.jpg", fully_reviewed=True))
        store.set(ImageVerification("b.jpg", fully_reviewed=False))
        store.set(ImageVerification("c.jpg", fully_reviewed=True))
        assert store.reviewed_images_count() == 2

    def test_build_class_list(self, tmp_path: Path):
        store = VerificationStore(tmp_path)
        dets = [
            VerifiedDetection("d0", Annotation(0, .5, .5, .3, .3), DetectionStatus.APPROVED, original_label="car"),
            VerifiedDetection("d1", Annotation(0, .5, .5, .3, .3), DetectionStatus.APPROVED, original_label="person"),
            VerifiedDetection("d2", Annotation(0, .5, .5, .3, .3), DetectionStatus.DELETED, original_label="bus"),
            VerifiedDetection("d3", Annotation(0, .5, .5, .3, .3), DetectionStatus.PENDING, original_label="truck"),
            VerifiedDetection("d4", Annotation(0, .5, .5, .3, .3), DetectionStatus.RENAMED, new_label="vehicle", original_label="car"),
        ]
        store.set(ImageVerification("a.jpg", detections=dets))
        classes = store.build_class_list()
        assert "car" in classes
        assert "person" in classes
        assert "vehicle" in classes
        # deleted and pending should NOT appear
        assert "bus" not in classes
        assert "truck" not in classes

    def test_per_class_image_counts(self, tmp_path: Path):
        store = VerificationStore(tmp_path)
        store.set(ImageVerification("a.jpg", detections=[
            VerifiedDetection("d0", Annotation(0, .5, .5, .3, .3), DetectionStatus.APPROVED, original_label="car"),
            VerifiedDetection("d1", Annotation(0, .5, .5, .3, .3), DetectionStatus.APPROVED, original_label="car"),
            VerifiedDetection("d2", Annotation(0, .5, .5, .3, .3), DetectionStatus.APPROVED, original_label="person"),
        ]))
        store.set(ImageVerification("b.jpg", detections=[
            VerifiedDetection("d0", Annotation(0, .5, .5, .3, .3), DetectionStatus.APPROVED, original_label="car"),
        ]))
        counts = store.per_class_image_counts(["car", "person"])
        # car appears in both images, person in one
        assert counts["car"] == 2
        assert counts["person"] == 1

    def test_per_class_image_counts_no_double_count(self, tmp_path: Path):
        """Multiple detections of same class in one image count as 1 image."""
        store = VerificationStore(tmp_path)
        store.set(ImageVerification("a.jpg", detections=[
            VerifiedDetection("d0", Annotation(0, .5, .5, .3, .3), DetectionStatus.APPROVED, original_label="car"),
            VerifiedDetection("d1", Annotation(0, .2, .2, .1, .1), DetectionStatus.APPROVED, original_label="car"),
        ]))
        counts = store.per_class_image_counts(["car"])
        assert counts["car"] == 1

    def test_finalized_annotations(self, tmp_path: Path):
        store = VerificationStore(tmp_path)
        store.set(ImageVerification("a.jpg", detections=[
            VerifiedDetection("d0", Annotation(99, .5, .5, .3, .3), DetectionStatus.APPROVED, original_label="person"),
            VerifiedDetection("d1", Annotation(99, .2, .2, .1, .1), DetectionStatus.DELETED, original_label="car"),
            VerifiedDetection("d2", Annotation(99, .7, .7, .2, .2), DetectionStatus.RENAMED, new_label="vehicle", original_label="truck"),
            VerifiedDetection("d3", Annotation(99, .1, .1, .1, .1), DetectionStatus.PENDING, original_label="dog"),
        ]))

        class_map = {"person": 0, "vehicle": 1}
        anns = store.finalized_annotations("a.jpg", class_map)
        assert len(anns) == 2
        assert anns[0].class_id == 0  # person
        assert anns[0].cx == pytest.approx(0.5)
        assert anns[1].class_id == 1  # vehicle (renamed from truck)
        assert anns[1].cx == pytest.approx(0.7)

    def test_finalized_annotations_reshaped(self, tmp_path: Path):
        store = VerificationStore(tmp_path)
        adj = Annotation(class_id=0, cx=0.6, cy=0.6, w=0.2, h=0.2)
        store.set(ImageVerification("a.jpg", detections=[
            VerifiedDetection("d0", Annotation(0, .5, .5, .3, .3), DetectionStatus.RESHAPED, adjusted=adj, original_label="person"),
        ]))
        anns = store.finalized_annotations("a.jpg", {"person": 0})
        assert len(anns) == 1
        assert anns[0].cx == pytest.approx(0.6)

    def test_finalized_annotations_missing_image(self, tmp_path: Path):
        store = VerificationStore(tmp_path)
        assert store.finalized_annotations("nope.jpg", {"person": 0}) == []

    def test_all_images(self, tmp_path: Path):
        store = VerificationStore(tmp_path)
        store.set(ImageVerification("a.jpg"))
        store.set(ImageVerification("b.jpg"))
        assert sorted(store.all_images()) == ["a.jpg", "b.jpg"]

    def test_clear_all(self, tmp_path: Path):
        store = VerificationStore(tmp_path)
        store.set(ImageVerification("a.jpg", fully_reviewed=True))
        store.set(ImageVerification("b.jpg"))
        store.set(ImageVerification("c.jpg"))
        assert len(store.all_images()) == 3

        store.clear_all()
        assert store.all_images() == []
        assert store.pending_count() == 0
        assert store.reviewed_images_count() == 0

    def test_clear_all_empty_store(self, tmp_path: Path):
        store = VerificationStore(tmp_path)
        store.clear_all()  # should not raise
        assert store.all_images() == []

    def test_corrupt_json_starts_fresh(self, tmp_path: Path):
        (tmp_path / "verifications.json").write_text("{bad json")
        store = VerificationStore(tmp_path)
        assert store.all_images() == []

    def test_per_class_image_counts_auto_derive(self, tmp_path: Path):
        """When class_names is None, derive from data."""
        store = VerificationStore(tmp_path)
        store.set(ImageVerification("a.jpg", detections=[
            VerifiedDetection("d0", Annotation(0, .5, .5, .3, .3), DetectionStatus.APPROVED, original_label="car"),
        ]))
        counts = store.per_class_image_counts()
        assert counts == {"car": 1}


# ---------------------------------------------------------------------------
# corrected_classes
# ---------------------------------------------------------------------------

class TestCorrectedClasses:
    def _ann(self) -> Annotation:
        return Annotation(class_id=0, cx=0.5, cy=0.5, w=0.3, h=0.3)

    def test_no_corrections(self, tmp_path: Path):
        """All APPROVED -> empty dict."""
        store = VerificationStore(tmp_path)
        store.set(ImageVerification("a.jpg", detections=[
            VerifiedDetection("d0", self._ann(), DetectionStatus.APPROVED, original_label="person"),
        ]))
        assert store.corrected_classes() == {}

    def test_deleted_detection(self, tmp_path: Path):
        store = VerificationStore(tmp_path)
        store.set(ImageVerification("a.jpg", detections=[
            VerifiedDetection("d0", self._ann(), DetectionStatus.DELETED, original_label="person"),
        ]))
        result = store.corrected_classes()
        assert result == {"person": {"deleted": 1}}

    def test_renamed_detection(self, tmp_path: Path):
        store = VerificationStore(tmp_path)
        store.set(ImageVerification("a.jpg", detections=[
            VerifiedDetection("d0", self._ann(), DetectionStatus.RENAMED,
                              new_label="dog", original_label="cat"),
        ]))
        result = store.corrected_classes()
        assert result["cat"] == {"renamed_from": 1}
        assert result["dog"] == {"renamed_to": 1}

    def test_reshaped_detection(self, tmp_path: Path):
        store = VerificationStore(tmp_path)
        adj = Annotation(class_id=0, cx=0.6, cy=0.6, w=0.2, h=0.2)
        store.set(ImageVerification("a.jpg", detections=[
            VerifiedDetection("d0", self._ann(), DetectionStatus.RESHAPED,
                              adjusted=adj, original_label="person"),
        ]))
        result = store.corrected_classes()
        assert result == {"person": {"reshaped": 1}}

    def test_added_detection(self, tmp_path: Path):
        store = VerificationStore(tmp_path)
        store.set(ImageVerification("a.jpg", detections=[
            VerifiedDetection("d0", self._ann(), DetectionStatus.ADDED, original_label="dog"),
        ]))
        result = store.corrected_classes()
        assert result == {"dog": {"added": 1}}

    def test_mixed_corrections_across_images(self, tmp_path: Path):
        store = VerificationStore(tmp_path)
        store.set(ImageVerification("a.jpg", detections=[
            VerifiedDetection("d0", self._ann(), DetectionStatus.ADDED, original_label="dog"),
            VerifiedDetection("d1", self._ann(), DetectionStatus.DELETED, original_label="person"),
        ]))
        store.set(ImageVerification("b.jpg", detections=[
            VerifiedDetection("d0", self._ann(), DetectionStatus.ADDED, original_label="dog"),
            VerifiedDetection("d1", self._ann(), DetectionStatus.APPROVED, original_label="car"),
        ]))
        result = store.corrected_classes()
        assert result["dog"] == {"added": 2}
        assert result["person"] == {"deleted": 1}
        assert "car" not in result

    def test_pending_not_counted(self, tmp_path: Path):
        store = VerificationStore(tmp_path)
        store.set(ImageVerification("a.jpg", detections=[
            VerifiedDetection("d0", self._ann(), DetectionStatus.PENDING, original_label="person"),
        ]))
        assert store.corrected_classes() == {}


# ---------------------------------------------------------------------------
# classes_needing_upload
# ---------------------------------------------------------------------------

class TestClassesNeedingUpload:
    def _ann(self) -> Annotation:
        return Annotation(class_id=0, cx=0.5, cy=0.5, w=0.3, h=0.3)

    def test_no_corrections_no_needs(self, tmp_path: Path):
        store = VerificationStore(tmp_path)
        assert store.classes_needing_upload() == []

    def test_corrected_class_below_threshold(self, tmp_path: Path):
        store = VerificationStore(tmp_path)
        store.set(ImageVerification("a.jpg", detections=[
            VerifiedDetection("d0", self._ann(), DetectionStatus.ADDED, original_label="dog"),
        ]))
        result = store.classes_needing_upload(min_images=15)
        assert len(result) == 1
        assert result[0]["class_name"] == "dog"
        assert result[0]["current_images"] == 1
        assert result[0]["target_images"] == 15

    def test_reason_summary_present(self, tmp_path: Path):
        store = VerificationStore(tmp_path)
        store.set(ImageVerification("a.jpg", detections=[
            VerifiedDetection("d0", self._ann(), DetectionStatus.ADDED, original_label="dog"),
        ]))
        result = store.classes_needing_upload()
        assert result[0]["reason_summary"]  # non-empty string

    def test_renamed_only_includes_target_class(self, tmp_path: Path):
        """Renaming airplane→package should only ask for package uploads."""
        store = VerificationStore(tmp_path)
        det = VerifiedDetection(
            "d0", self._ann(), DetectionStatus.RENAMED,
            original_label="airplane", new_label="package",
        )
        store.set(ImageVerification("a.jpg", detections=[det]))
        result = store.classes_needing_upload()
        names = [e["class_name"] for e in result]
        assert "package" in names
        assert "airplane" not in names

    def test_deleted_class_excluded(self, tmp_path: Path):
        """A class that was only deleted should not need uploads."""
        store = VerificationStore(tmp_path)
        det = VerifiedDetection(
            "d0", self._ann(), DetectionStatus.DELETED, original_label="person",
        )
        store.set(ImageVerification("a.jpg", detections=[det]))
        result = store.classes_needing_upload()
        assert result == []

    def test_approved_only_class_excluded(self, tmp_path: Path):
        """Classes that were only approved (no edits) don't need uploads."""
        store = VerificationStore(tmp_path)
        store.set(ImageVerification("a.jpg", detections=[
            VerifiedDetection("d0", self._ann(), DetectionStatus.APPROVED, original_label="car"),
        ]))
        result = store.classes_needing_upload(min_images=10)
        assert result == []

    def test_class_at_threshold_not_included(self, tmp_path: Path):
        """A corrected class with enough images should not appear."""
        store = VerificationStore(tmp_path)
        for i in range(10):
            store.set(ImageVerification(f"img_{i}.jpg", detections=[
                VerifiedDetection(f"d0", self._ann(), DetectionStatus.ADDED, original_label="dog"),
            ]))
        result = store.classes_needing_upload(min_images=10)
        assert result == []

    def test_delete_class_detections(self, tmp_path: Path):
        """Bulk-delete all detections of a given class."""
        store = VerificationStore(tmp_path)
        store.set(ImageVerification("a.jpg", detections=[
            VerifiedDetection("d0", self._ann(), DetectionStatus.APPROVED, original_label="person"),
            VerifiedDetection("d1", self._ann(), DetectionStatus.PENDING, original_label="car"),
            VerifiedDetection("d2", self._ann(), DetectionStatus.APPROVED, original_label="person"),
        ]))
        store.set(ImageVerification("b.jpg", detections=[
            VerifiedDetection("d0", self._ann(), DetectionStatus.APPROVED, original_label="person"),
            VerifiedDetection("d1", self._ann(), DetectionStatus.DELETED, original_label="person"),
        ]))
        count = store.delete_class_detections("person")
        assert count == 3  # 2 from a.jpg + 1 from b.jpg (already-deleted one skipped)

        # Verify all person detections are now DELETED
        for iv in [store.get("a.jpg"), store.get("b.jpg")]:
            for det in iv.detections:
                if det.original_label == "person":
                    assert det.status == DetectionStatus.DELETED

        # car detection untouched
        a_iv = store.get("a.jpg")
        car_det = [d for d in a_iv.detections if d.original_label == "car"][0]
        assert car_det.status == DetectionStatus.PENDING

    def test_delete_class_detections_renamed(self, tmp_path: Path):
        """Bulk-delete uses effective_label, so renamed detections match."""
        store = VerificationStore(tmp_path)
        store.set(ImageVerification("a.jpg", detections=[
            VerifiedDetection("d0", self._ann(), DetectionStatus.RENAMED,
                              original_label="truck", new_label="vehicle"),
        ]))
        count = store.delete_class_detections("vehicle")
        assert count == 1
        assert store.get("a.jpg").detections[0].status == DetectionStatus.DELETED

    def test_delete_class_detections_nonexistent(self, tmp_path: Path):
        """Deleting a class that doesn't exist returns 0."""
        store = VerificationStore(tmp_path)
        store.set(ImageVerification("a.jpg", detections=[
            VerifiedDetection("d0", self._ann(), DetectionStatus.APPROVED, original_label="car"),
        ]))
        assert store.delete_class_detections("nonexistent") == 0

    def test_class_detection_counts(self, tmp_path: Path):
        """Count non-deleted detections per class."""
        store = VerificationStore(tmp_path)
        store.set(ImageVerification("a.jpg", detections=[
            VerifiedDetection("d0", self._ann(), DetectionStatus.APPROVED, original_label="person"),
            VerifiedDetection("d1", self._ann(), DetectionStatus.APPROVED, original_label="person"),
            VerifiedDetection("d2", self._ann(), DetectionStatus.APPROVED, original_label="car"),
            VerifiedDetection("d3", self._ann(), DetectionStatus.DELETED, original_label="car"),
        ]))
        store.set(ImageVerification("b.jpg", detections=[
            VerifiedDetection("d0", self._ann(), DetectionStatus.PENDING, original_label="person"),
        ]))
        counts = store.class_detection_counts()
        assert counts["person"] == {"detections": 3, "images": 2}
        assert counts["car"] == {"detections": 1, "images": 1}
        assert "car" in counts  # deleted one should not contribute
        # The deleted car detection should not be counted — only 1 of 2 car dets remains
        assert counts["car"]["detections"] == 1

    def test_class_detection_counts_empty(self, tmp_path: Path):
        """Empty store returns empty dict."""
        store = VerificationStore(tmp_path)
        assert store.class_detection_counts() == {}

    def test_class_detection_counts_all_deleted(self, tmp_path: Path):
        """All-deleted detections produce empty counts."""
        store = VerificationStore(tmp_path)
        store.set(ImageVerification("a.jpg", detections=[
            VerifiedDetection("d0", self._ann(), DetectionStatus.DELETED, original_label="person"),
        ]))
        assert store.class_detection_counts() == {}

    def test_approved_person_added_package_one_image(self, tmp_path: Path):
        """1 image: person approved, package added → needs package uploads."""
        store = VerificationStore(tmp_path)
        store.set(ImageVerification("img.jpg", detections=[
            VerifiedDetection("d0", self._ann(), DetectionStatus.APPROVED,
                              original_label="person"),
            VerifiedDetection("d1", self._ann(), DetectionStatus.ADDED,
                              original_label="package"),
        ], fully_reviewed=True))
        store.save()

        # Reload from disk (same path the upload phase takes)
        store2 = VerificationStore(tmp_path)
        result = store2.classes_needing_upload(min_images=10)
        assert len(result) == 1
        assert result[0]["class_name"] == "package"
        assert result[0]["current_images"] == 1
        # person should NOT appear (only approved, not corrected)
        names = [e["class_name"] for e in result]
        assert "person" not in names
