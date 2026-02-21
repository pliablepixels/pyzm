"""Abstract base class for all ML inference backends."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import portalocker

if TYPE_CHECKING:
    import numpy as np

    from pyzm.models.config import ModelConfig

from pyzm.models.detection import Detection

logger = logging.getLogger("pyzm.ml")


class MLBackend(ABC):
    """Abstract base for ML inference backends.

    Every concrete backend must implement :meth:`load`, :meth:`detect`, and
    the :attr:`name` property.  The host pipeline calls ``load()`` once and
    then ``detect()`` per-frame.
    """

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""

    @abstractmethod
    def detect(self, image: "np.ndarray") -> list[Detection]:
        """Run inference on a single image, return raw detections."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this backend instance."""

    @property
    def is_loaded(self) -> bool:
        return False

    @property
    def needs_exclusive_lock(self) -> bool:
        """True if this backend requires exclusive hardware (e.g. EdgeTPU)."""
        return False

    def acquire_lock(self) -> None:
        """Acquire the hardware lock. No-op by default."""

    def release_lock(self) -> None:
        """Release the hardware lock. No-op by default."""


class PortalockerMixin:
    """Shared portalocker-based locking for backends that need GPU/TPU serialization.

    Expects ``self._config: ModelConfig`` to be set before ``_init_lock()`` is called.
    Provides ``acquire_lock()`` / ``release_lock()`` that replace the identical
    boilerplate previously duplicated across YOLO, Face, and Coral backends.
    """

    _auto_lock: bool = True

    def _init_lock(self) -> None:
        """Initialize the portalocker BoundedSemaphore from ``self._config``."""
        cfg: ModelConfig = self._config  # type: ignore[attr-defined]
        self._is_locked = False
        self._disable_locks = cfg.disable_locks
        processor = cfg.processor.value
        self._lock_name = f"pyzm_uid{os.getuid()}_{processor}_lock"
        self._lock_maximum = cfg.max_processes
        self._lock_timeout = cfg.max_lock_wait

        if not self._disable_locks:
            logger.debug(
                "%s: portalocker: max=%d, name=%s, timeout=%d",
                getattr(self, "name", "?"),
                self._lock_maximum,
                self._lock_name,
                self._lock_timeout,
            )
            self._lock = portalocker.BoundedSemaphore(
                maximum=self._lock_maximum,
                name=self._lock_name,
                timeout=self._lock_timeout,
            )

    def acquire_lock(self) -> None:
        if self._disable_locks:
            return
        if self._is_locked:
            logger.debug("%s: %s portalocker already acquired", getattr(self, "name", "?"), self._lock_name)
            return
        try:
            logger.debug("%s: waiting for %s portalocker...", getattr(self, "name", "?"), self._lock_name)
            self._lock.acquire()
            logger.debug("%s: got %s portalocker", getattr(self, "name", "?"), self._lock_name)
            self._is_locked = True
        except portalocker.AlreadyLocked:
            logger.error(
                "%s: timeout waiting for %s portalocker for %d seconds",
                getattr(self, "name", "?"),
                self._lock_name,
                self._lock_timeout,
            )
            raise ValueError(
                f"{getattr(self, 'name', '?')}: timeout waiting for "
                f"{self._lock_name} portalocker for {self._lock_timeout} seconds"
            )

    def release_lock(self) -> None:
        if self._disable_locks:
            return
        if not self._is_locked:
            logger.debug("%s: %s portalocker already released", getattr(self, "name", "?"), self._lock_name)
            return
        self._lock.release()
        self._is_locked = False
        logger.debug("%s: released %s portalocker", getattr(self, "name", "?"), self._lock_name)
