"""Tests for pyzm.serve.auth -- JWT authentication."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pyzm.models.config import ServerConfig
from pyzm.models.detection import BBox, Detection, DetectionResult


def _mock_detector():
    det = MagicMock()
    det._pipeline = True
    det._config = MagicMock()
    det._config.models = [MagicMock()]
    det.detect.return_value = DetectionResult(
        detections=[
            Detection(label="person", confidence=0.95, bbox=BBox(10, 20, 50, 80), model_name="yolov4")
        ],
        frame_id="single",
    )
    return det


@pytest.fixture
def auth_client():
    config = ServerConfig(
        models=["yolov4"],
        auth_enabled=True,
        auth_username="admin",
        auth_password="secret123",
        token_secret="test-secret",
        token_expiry_seconds=3600,
    )
    with patch("pyzm.serve.app.Detector") as MockDetector:
        mock_det = _mock_detector()
        MockDetector.return_value = mock_det
        mock_det._ensure_pipeline = MagicMock()
        from pyzm.serve.app import create_app
        application = create_app(config)
        from fastapi.testclient import TestClient
        with TestClient(application) as tc:
            yield tc


class TestLogin:
    def test_login_success(self, auth_client):
        resp = auth_client.post("/login", json={"username": "admin", "password": "secret123"})
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "expires" in data

    def test_login_bad_password(self, auth_client):
        resp = auth_client.post("/login", json={"username": "admin", "password": "wrong"})
        assert resp.status_code == 401

    def test_login_bad_username(self, auth_client):
        resp = auth_client.post("/login", json={"username": "hacker", "password": "secret123"})
        assert resp.status_code == 401


class TestAuthProtectedEndpoints:
    def _get_token(self, client):
        resp = client.post("/login", json={"username": "admin", "password": "secret123"})
        return resp.json()["access_token"]

    @pytest.mark.integration
    def test_detect_without_token_rejected(self, auth_client):
        import cv2, numpy as np
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        resp = auth_client.post("/detect", files={"file": ("t.jpg", buf.tobytes(), "image/jpeg")})
        assert resp.status_code in (401, 403)

    @pytest.mark.integration
    def test_detect_with_valid_token(self, auth_client):
        import cv2, numpy as np
        token = self._get_token(auth_client)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        resp = auth_client.post(
            "/detect",
            files={"file": ("t.jpg", buf.tobytes(), "image/jpeg")},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["labels"] == ["person"]

    @pytest.mark.integration
    def test_detect_with_bad_token(self, auth_client):
        import cv2, numpy as np
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        resp = auth_client.post(
            "/detect",
            files={"file": ("t.jpg", buf.tobytes(), "image/jpeg")},
            headers={"Authorization": "Bearer invalid-token-here"},
        )
        assert resp.status_code == 401

    def test_health_no_auth_required(self, auth_client):
        resp = auth_client.get("/health")
        assert resp.status_code == 200


# ===================================================================
# No-auth server still accepts /login
# ===================================================================

@pytest.fixture
def noauth_client():
    """Server with auth_enabled=False."""
    config = ServerConfig(models=["yolov4"], auth_enabled=False)

    with patch("pyzm.serve.app.Detector") as MockDetector:
        mock_det = _mock_detector()
        MockDetector.return_value = mock_det
        mock_det._ensure_pipeline = MagicMock()
        from pyzm.serve.app import create_app
        application = create_app(config)
        from fastapi.testclient import TestClient
        with TestClient(application) as tc:
            yield tc


class TestNoAuthLogin:
    """When the server runs without --auth, /login should still work."""

    def test_login_accepts_any_credentials(self, noauth_client):
        resp = noauth_client.post("/login", json={"username": "anyone", "password": "whatever"})
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data

    def test_login_returns_valid_token(self, noauth_client):
        resp = noauth_client.post("/login", json={"username": "admin", "password": "secret"})
        token = resp.json()["access_token"]
        # Token should be a non-empty string
        assert isinstance(token, str) and len(token) > 0

    @pytest.mark.integration
    def test_detect_works_without_token(self, noauth_client):
        """Endpoints don't require auth when auth is disabled."""
        import cv2, numpy as np
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        resp = noauth_client.post("/detect", files={"file": ("t.jpg", buf.tobytes(), "image/jpeg")})
        assert resp.status_code == 200

    @pytest.mark.integration
    def test_detect_works_with_token(self, noauth_client):
        """Endpoints accept (but don't require) a Bearer token when auth is disabled."""
        import cv2, numpy as np
        token = noauth_client.post("/login", json={"username": "x", "password": "y"}).json()["access_token"]
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        resp = noauth_client.post(
            "/detect",
            files={"file": ("t.jpg", buf.tobytes(), "image/jpeg")},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
