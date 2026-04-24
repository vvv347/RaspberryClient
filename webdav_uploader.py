"""Background WebDAV uploader with a thread-safe queue."""

import logging
import os
import queue
import threading
from pathlib import Path, PurePosixPath

from webdav3.client import Client


class WebDAVUploader:
    def __init__(self, config: dict) -> None:
        login = os.environ.get("WEBDAV_LOGIN") or config.get("login", "")
        password = os.environ.get("WEBDAV_PASSWORD") or config.get("password", "")

        if not login or not password:
            raise ValueError(
                "WebDAV credentials missing. Set WEBDAV_LOGIN and WEBDAV_PASSWORD env vars."
            )

        self._client = Client({
            "webdav_hostname": config["hostname"],
            "webdav_login": login,
            "webdav_password": password,
            "webdav_root": config.get("root", "/webdav/"),
        })
        self._remote_base = config.get("remote_dir", "timelapse").strip("/")
        self._queue: queue.Queue = queue.Queue()
        self._errors = 0
        self._thread = threading.Thread(target=self._worker, daemon=True, name="webdav")
        self._thread.start()

    def check_connection(self) -> bool:
        try:
            self._client.check(self._remote_base) or self._client.mkdir(self._remote_base)
            logging.info("WebDAV connected: %s/%s", self._client.webdav.hostname, self._remote_base)
            return True
        except Exception as exc:
            logging.error("WebDAV connection failed: %s", exc)
            return False

    def _ensure_dirs(self, remote_path: str) -> None:
        parts = PurePosixPath(remote_path).parts
        current = ""
        for part in parts:
            current = f"{current}/{part}".lstrip("/")
            try:
                if not self._client.check(current):
                    self._client.mkdir(current)
            except Exception:
                pass

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            local_path, remote_path = item
            try:
                self._ensure_dirs(str(PurePosixPath(remote_path).parent))
                self._client.upload_sync(remote_path=remote_path, local_path=str(local_path))
                logging.debug("Uploaded → %s", remote_path)
            except Exception as exc:
                self._errors += 1
                logging.error("Upload failed [%s]: %s", local_path.name, exc)
            finally:
                self._queue.task_done()

    def upload_frame(self, local_path: Path, session: str) -> None:
        remote = f"{self._remote_base}/{session}/{local_path.name}"
        self._queue.put((local_path, remote))

    def upload_video(self, local_path: Path) -> None:
        remote = f"{self._remote_base}/{local_path.name}"
        self._queue.put((local_path, remote))

    def flush(self, timeout: float = 120.0) -> None:
        """Wait for all queued uploads to finish."""
        self._queue.join()

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=30)
        if self._errors:
            logging.warning("WebDAV total errors: %d", self._errors)
