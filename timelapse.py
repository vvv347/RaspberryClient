#!/usr/bin/env python3
"""Time-lapse recorder for Microsoft LifeCam Studio on Raspberry Pi 4."""

import argparse
import fcntl
import json
import logging
import os
import signal
import struct
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from webdav_uploader import WebDAVUploader


def setup_logging(debug: bool = False) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG if debug else logging.INFO,
    )


def load_config(path: str) -> dict:
    defaults = {
        "camera_device": "/dev/video0",
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
            "remote_dir": "timelapse",
        },
    }
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
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
        self.device: str = ""
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
                logging.warning("WebDAV unavailable — saving locally only")
                self.uploader = None
        except ValueError as exc:
            logging.warning("WebDAV disabled: %s", exc)

    @staticmethod
    def _is_capture_device(path: str) -> bool:
        """VIDIOC_QUERYCAP via O_RDWR|O_NONBLOCK — required by V4L2 spec."""
        # struct v4l2_capability: driver[16] card[32] bus_info[32] version(4) capabilities(4)
        try:
            fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
            try:
                buf = bytearray(104)
                fcntl.ioctl(fd, 0x80685600, buf)           # VIDIOC_QUERYCAP
                caps = struct.unpack_from("<I", buf, 84)[0]
                return bool(caps & 0x00000001)              # V4L2_CAP_VIDEO_CAPTURE
            finally:
                os.close(fd)
        except OSError:
            return False

    @staticmethod
    def _ffmpeg_probe(path: str) -> bool:
        """Fallback: attempt to grab one frame with ffmpeg."""
        r = subprocess.run(
            ["ffmpeg", "-y", "-f", "v4l2", "-i", path, "-frames:v", "1", "-f", "null", "-"],
            capture_output=True, timeout=8,
        )
        return r.returncode == 0

    @staticmethod
    def _find_camera_device(device) -> str:
        preferred = f"/dev/video{device}" if isinstance(device, int) else device
        all_nodes = sorted(Path("/dev").glob("video*"), key=lambda p: int(p.name[5:]))

        capture_nodes = [str(p) for p in all_nodes if TimelapseRecorder._is_capture_device(str(p))]

        if not capture_nodes:
            logging.warning("ioctl probe found nothing — falling back to ffmpeg probe (may take a moment)")
            capture_nodes = [str(p) for p in all_nodes if TimelapseRecorder._ffmpeg_probe(str(p))]

        if not capture_nodes:
            raise RuntimeError(
                "No usable V4L2 capture device found.\n"
                "If you see permission errors, run:\n"
                "  sudo usermod -aG video $USER  && newgrp video\n"
                f"Nodes checked: {[str(p) for p in all_nodes]}"
            )

        if preferred in capture_nodes:
            return preferred

        chosen = capture_nodes[0]
        logging.warning("'%s' not usable — using '%s'", preferred, chosen)
        return chosen

    def _open_camera(self) -> None:
        self.device = self._find_camera_device(self.config["camera_device"])
        w, h = self.config["resolution"]
        # Quick probe: capture one test frame to validate device + resolution
        test_file = self.session_dir / ".probe.jpg"
        result = subprocess.run(
            self._ffmpeg_capture_cmd(str(test_file), w, h),
            capture_output=True, timeout=10,
        )
        test_file.unlink(missing_ok=True)
        if result.returncode != 0:
            err = result.stderr.decode(errors="replace")[-400:]
            raise RuntimeError(f"Camera probe failed on {self.device}:\n{err}")
        logging.info("Camera ready: %s at %dx%d", self.device, w, h)

    def _ffmpeg_capture_cmd(self, output: str, w: int, h: int) -> list:
        # ffmpeg JPEG -q:v scale: 1 (best) – 31 (worst)
        q = max(1, round(31 * (100 - self.config["jpeg_quality"]) / 100))
        return [
            "ffmpeg", "-y",
            "-f", "v4l2",
            "-video_size", f"{w}x{h}",
            "-i", self.device,
            "-frames:v", "1",
            "-q:v", str(q),
            output,
        ]

    def _capture_frame(self) -> bool:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        ext = self.config["frame_format"]
        filename = self.session_dir / f"frame_{self.frame_count:06d}_{ts}.{ext}"
        w, h = self.config["resolution"]

        result = subprocess.run(
            self._ffmpeg_capture_cmd(str(filename), w, h),
            capture_output=True, timeout=15,
        )
        if result.returncode != 0:
            logging.warning("Frame capture failed, skipping")
            return False

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
            logging.error("ffmpeg compile failed:\n%s", result.stderr[-800:])
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
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--device", help="Camera device path (e.g. /dev/video2)")
    parser.add_argument("--interval", type=float, help="Seconds between frames")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--fps", type=int, help="Output video FPS")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-webdav", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    setup_logging(args.debug)

    config = load_config(args.config)
    if args.device:
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
