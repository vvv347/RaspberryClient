#!/usr/bin/env python3
"""Time-lapse recorder for Microsoft LifeCam Studio on Raspberry Pi 4."""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

from webdav_uploader import WebDAVUploader


def setup_logging(debug: bool = False) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG if debug else logging.INFO,
    )


def load_config(path: str) -> dict:
    defaults = {
        "camera_device": 0,
        "interval": 5.0,
        "resolution": [1920, 1080],
        "output_dir": "output",
        "compile_on_exit": True,
        "video_fps": 24,
        "frame_format": "jpg",
        "jpeg_quality": 95,
        "webdav": {
            "hostname": "https://my-hdd-1.keenetic.link",
            "root": "/webdav/",
            "remote_dir": "timelapse"
        },
    }
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        # Deep-merge webdav section
        if "webdav" in data:
            defaults["webdav"].update(data.pop("webdav"))
        defaults.update(data)
    return defaults


class TimelapseRecorder:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.running = False
        self.frame_count = 0
        self.session_dir: Optional[Path] = None
        self.session_name: str = ""
        self.cap: Optional[cv2.VideoCapture] = None
        self.uploader: Optional[WebDAVUploader] = None

    def _setup_session(self) -> None:
        self.session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(self.config["output_dir"]) / self.session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Session directory: %s", self.session_dir)

    def _setup_webdav(self) -> None:
        try:
            self.uploader = WebDAVUploader(self.config["webdav"])
            if not self.uploader.check_connection():
                logging.warning("WebDAV unavailable — frames will be saved locally only")
                self.uploader = None
        except ValueError as exc:
            logging.warning("WebDAV disabled: %s", exc)

    def _open_camera(self) -> None:
        device = self.config["camera_device"]
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera: {device}")

        w, h = self.config["resolution"]
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        for _ in range(5):
            self.cap.grab()

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logging.info("Camera opened at %dx%d", actual_w, actual_h)

    def _capture_frame(self) -> bool:
        self.cap.grab()
        ret, frame = self.cap.retrieve()
        if not ret or frame is None:
            logging.warning("Failed to capture frame, skipping")
            return False

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        ext = self.config["frame_format"]
        filename = self.session_dir / f"frame_{self.frame_count:06d}_{ts}.{ext}"

        params = [cv2.IMWRITE_JPEG_QUALITY, self.config["jpeg_quality"]] if ext in ("jpg", "jpeg") else []
        cv2.imwrite(str(filename), frame, params)

        self.frame_count += 1
        if self.frame_count % 10 == 0:
            logging.info("Captured %d frames", self.frame_count)
        else:
            logging.debug("Frame %d: %s", self.frame_count, filename.name)

        if self.uploader:
            self.uploader.upload_frame(filename, self.session_name)

        return True

    def _compile_video(self) -> Optional[Path]:
        if self.frame_count == 0:
            logging.warning("No frames to compile")
            return None

        ext = self.config["frame_format"]
        frames = sorted(self.session_dir.glob(f"frame_*.{ext}"))
        if not frames:
            return None

        list_file = self.session_dir / "frames.txt"
        frame_duration = 1.0 / self.config["video_fps"]
        with open(list_file, "w") as f:
            for fp in frames:
                f.write(f"file '{fp.absolute()}'\n")
                f.write(f"duration {frame_duration:.6f}\n")

        output_video = Path(self.config["output_dir"]) / f"timelapse_{self.session_name}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(output_video),
        ]

        logging.info("Compiling %d frames → %s", self.frame_count, output_video)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logging.error("ffmpeg failed:\n%s", result.stderr[-800:])
            return None

        logging.info("Video saved: %s", output_video)
        return output_video

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False

    def run(self) -> None:
        self._setup_session()
        self._setup_webdav()
        self._open_camera()
        self.running = True

        interval = self.config["interval"]
        logging.info("Recording every %.1fs — press Ctrl+C to stop", interval)

        try:
            while self.running:
                t0 = time.monotonic()
                self._capture_frame()
                sleep = interval - (time.monotonic() - t0)
                if sleep > 0:
                    time.sleep(sleep)
        except KeyboardInterrupt:
            pass
        finally:
            logging.info("Stopped. Total frames: %d", self.frame_count)
            if self.cap:
                self.cap.release()

            video_path = None
            if self.config.get("compile_on_exit") and self.frame_count > 0:
                video_path = self._compile_video()

            if self.uploader:
                if video_path:
                    self.uploader.upload_video(video_path)
                logging.info("Waiting for uploads to finish...")
                self.uploader.flush()
                self.uploader.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Time-lapse recorder — Raspberry Pi 4 + LifeCam Studio")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument("--device", type=int, help="Camera device index (default: 0)")
    parser.add_argument("--interval", type=float, help="Seconds between frames")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--fps", type=int, help="Output video FPS")
    parser.add_argument("--no-compile", action="store_true", help="Skip video compilation on exit")
    parser.add_argument("--no-webdav", action="store_true", help="Disable WebDAV upload")
    parser.add_argument("--debug", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    setup_logging(args.debug)

    config = load_config(args.config)
    if args.device is not None:
        config["camera_device"] = args.device
    if args.interval is not None:
        config["interval"] = args.interval
    if args.output:
        config["output_dir"] = args.output
    if args.fps is not None:
        config["video_fps"] = args.fps
    if args.no_compile:
        config["compile_on_exit"] = False
    if args.no_webdav:
        config.pop("webdav", None)

    recorder = TimelapseRecorder(config)
    signal.signal(signal.SIGTERM, lambda _s, _f: recorder.stop())

    recorder.run()


if __name__ == "__main__":
    main()
