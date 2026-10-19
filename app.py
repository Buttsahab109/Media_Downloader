#!/usr/bin/env python3
"""
Media Downloader & Audio Converter Suite - Web Edition
======================================================

A small Flask server that exposes the same two tools as the desktop app
through a browser interface:

1. Video Downloader - yt-dlp downloads on the server, you save the result.
2. Audio Converter  - upload a local video, get back MP3 or WAV (FFmpeg).

Run locally:
    pip install -r requirements.txt
    python app.py              ->  http://127.0.0.1:5000

Production (single worker is REQUIRED: job state lives in memory):
    gunicorn -w 1 --threads 8 -b 0.0.0.0:8000 --timeout 300 app:app

Environment variables (all optional):
    HOST, PORT               bind address for `python app.py`
    MEDIA_WORK_DIR           where temporary job files are stored
    MAX_UPLOAD_MB            max upload size for the converter   (500)
    MAX_DOWNLOAD_MB          max size of one downloaded video    (2048)
    MAX_DURATION_MIN         max video length in minutes         (180)
    MAX_ACTIVE_JOBS          jobs allowed to run at once         (4)
    RATE_LIMIT_PER_MIN       requests per IP per minute          (12)
    JOB_TTL_MIN              minutes finished files are kept     (30)
    TRUST_PROXY=1            read client IP from X-Forwarded-For
"""

from __future__ import annotations

import collections
import ipaddress
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

from flask import Flask, Response, jsonify, request, send_file
from werkzeug.utils import secure_filename

try:
    import yt_dlp
except ImportError:  # reported through /api/health
    yt_dlp = None


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


WORK_DIR = (Path(os.environ.get("MEDIA_WORK_DIR", tempfile.gettempdir()))
            / "media_suite_web")
MAX_UPLOAD_MB = _env_int("MAX_UPLOAD_MB", 500)
MAX_DOWNLOAD_MB = _env_int("MAX_DOWNLOAD_MB", 2048)
MAX_DURATION_S = _env_int("MAX_DURATION_MIN", 180) * 60
MAX_ACTIVE_JOBS = _env_int("MAX_ACTIVE_JOBS", 4)
RATE_LIMIT_PER_MIN = _env_int("RATE_LIMIT_PER_MIN", 12)
JOB_TTL_S = _env_int("JOB_TTL_MIN", 30) * 60
TRUST_PROXY = os.environ.get("TRUST_PROXY") == "1"

VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v",
    ".mpg", ".mpeg", ".3gp", ".ts", ".m2ts", ".ogv",
}
MP3_BITRATES = ("128k", "192k", "256k", "320k")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


# --------------------------------------------------------------------------- #
# Errors, validation, rate limiting
# --------------------------------------------------------------------------- #
class ApiError(Exception):
    """An error that is safe to show to the user."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


class CancelledByUser(Exception):
    """Raised inside yt-dlp hooks to abort a download."""


@app.errorhandler(ApiError)
def _handle_api_error(exc: ApiError):
    return jsonify(error=exc.message), exc.status


@app.errorhandler(413)
def _handle_too_large(_exc):
    return jsonify(error=f"That file is too large (limit {MAX_UPLOAD_MB} MB)."
                   ), 413


@app.errorhandler(404)
def _handle_404(_exc):
    return jsonify(error="Not found."), 404


@app.errorhandler(500)
def _handle_500(_exc):
    return jsonify(error="Unexpected server error."), 500


@app.after_request
def _security_headers(response: Response) -> Response:
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' https: data:; "
        "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
        "frame-ancestors 'none'")
    return response


def clean_error(exc: BaseException) -> str:
    text = ANSI_RE.sub("", str(exc)).strip()
    text = re.sub(r"^ERROR:\s*", "", text)
    return text[:400] or exc.__class__.__name__


def validate_url(raw: object) -> str:
    """Return a safe http(s) URL or raise ApiError.

    Hostnames that resolve to private/loopback/link-local addresses are
    refused so the server cannot be used to probe its own network (SSRF).
    """
    url = raw.strip() if isinstance(raw, str) else ""
    if not url:
        raise ApiError("Please enter a video URL.")
    if len(url) > 2000:
        raise ApiError("That link is too long.")
    parsed = urlparse(url)
    host = parsed.hostname
    if parsed.scheme not in ("http", "https") or not host or "." not in host:
        raise ApiError("That doesn't look like a valid link. "
                       "It must start with http:// or https://")
    try:
        infos = socket.getaddrinfo(host, parsed.port or 443,
                                   proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError):
        raise ApiError("Could not find that website. Check the link.")
    for info in infos:
        address = ipaddress.ip_address(info[4][0].split("%")[0])
        if not address.is_global:
            raise ApiError("That address is not allowed.")
    return url


_hits: dict[str, collections.deque] = collections.defaultdict(
    collections.deque)
_hits_lock = threading.Lock()


def _client_ip() -> str:
    if TRUST_PROXY:
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def check_rate_limit() -> None:
    ip = _client_ip()
    now = time.time()
    with _hits_lock:
        window = _hits[ip]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= RATE_LIMIT_PER_MIN:
            raise ApiError("Too many requests. Please wait a minute.", 429)
        window.append(now)


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #
class Job:
    """One download or conversion running in a background thread."""

    def __init__(self, kind: str) -> None:
        self.id = uuid.uuid4().hex
        self.kind = kind
        self.status = "queued"  # queued | running | done | error | cancelled
        self.progress = 0.0
        self.message = "Queued"
        self.error = ""
        self.file: Path | None = None
        self.filename = ""
        self.created = time.time()
        self.finished: float | None = None
        self.cancel = threading.Event()
        self.proc: subprocess.Popen | None = None
        self.dir = WORK_DIR / self.id
        self.dir.mkdir(parents=True, exist_ok=True)

    def finish(self, status: str, message: str, error: str = "") -> None:
        self.status = status
        self.message = message
        self.error = error
        self.finished = time.time()
        if status == "done":
            self.progress = 1.0

    def to_dict(self) -> dict:
        size = None
        if self.file and self.file.exists():
            size = self.file.stat().st_size
        return {
            "id": self.id, "kind": self.kind, "status": self.status,
            "progress": round(self.progress, 4), "message": self.message,
            "error": self.error, "filename": self.filename, "size": size,
        }


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()


def create_job(kind: str) -> Job:
    with JOBS_LOCK:
        active = sum(1 for j in JOBS.values()
                     if j.status in ("queued", "running"))
        if active >= MAX_ACTIVE_JOBS:
            raise ApiError("The server is busy right now. "
                           "Please try again in a minute.", 503)
        job = Job(kind)
        JOBS[job.id] = job
        return job


def get_job(job_id: str) -> Job:
    job = JOBS.get(job_id)
    if job is None:
        raise ApiError("That job has expired or does not exist.", 404)
    return job


def _cleanup_loop() -> None:
    """Delete finished jobs (and their files) after JOB_TTL_S."""
    while True:
        time.sleep(60)
        now = time.time()
        expired = []
        with JOBS_LOCK:
            for job_id, job in JOBS.items():
                if job.finished and now - job.finished > JOB_TTL_S:
                    expired.append(job_id)
                elif not job.finished and now - job.created > 12 * 3600:
                    job.cancel.set()  # stuck job safety net
            removed = [JOBS.pop(job_id) for job_id in expired]
        for job in removed:
            shutil.rmtree(job.dir, ignore_errors=True)


def _start_background() -> None:
    shutil.rmtree(WORK_DIR, ignore_errors=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=_cleanup_loop, daemon=True).start()


# --------------------------------------------------------------------------- #
# Downloader
# --------------------------------------------------------------------------- #
class QuietLogger:
    """Keep yt-dlp from writing to the server console."""

    def debug(self, msg: str) -> None:
        pass

    def info(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        pass

    def error(self, msg: str) -> None:
        pass


def format_speed(value: float | None) -> str:
    if not value:
        return ""
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024:
            return f"{value:.1f} {unit}/s"
        value /= 1024
    return f"{value:.1f} TB/s"


def format_eta(value: int | float | None) -> str:
    if value is None:
        return ""
    minutes, seconds = divmod(int(value), 60)
    hours, minutes = divmod(minutes, 60)
    return (f"{hours}:{minutes:02d}:{seconds:02d}" if hours
            else f"{minutes:02d}:{seconds:02d}")


def build_format(height: int | None, have_ffmpeg: bool) -> str:
    """yt-dlp format selector for the chosen maximum height."""
    if height is None:
        return "bestvideo+bestaudio/best" if have_ffmpeg else "best"
    if have_ffmpeg:
        return (f"bestvideo[height<={height}]+bestaudio/"
                f"best[height<={height}]/best")
    return f"best[height<={height}]/best"


def parse_resolution(value: object) -> int | None:
    if value in (None, "", "Best available", "best"):
        return None
    match = re.fullmatch(r"(\d{3,4})p?", str(value).strip())
    if not match:
        raise ApiError("Unknown resolution.")
    return int(match.group(1))


def download_worker(job: Job, url: str, height: int | None) -> None:
    def hook(data: dict) -> None:
        if job.cancel.is_set():
            raise CancelledByUser()
        status = data.get("status")
        if status == "downloading":
            total = data.get("total_bytes") or data.get("total_bytes_estimate")
            done = data.get("downloaded_bytes", 0)
            fraction = (done / total) if total else 0.0
            job.progress = min(max(fraction, 0.0), 1.0) * 0.97
            parts = [f"Downloading {fraction * 100:4.1f}%",
                     format_speed(data.get("speed")),
                     f"ETA {format_eta(data.get('eta'))}"
                     if data.get("eta") is not None else ""]
            job.message = "  ·  ".join(p for p in parts if p)
        elif status == "finished":
            job.progress = 0.98
            job.message = "Processing..."

    opts = {
        "format": build_format(height, bool(FFMPEG)),
        "outtmpl": str(job.dir / "%(title).120B [%(id)s].%(ext)s"),
        "progress_hooks": [hook],
        "noplaylist": True,
        "playlistend": 1,
        "quiet": True,
        "no_warnings": True,
        "windowsfilenames": True,
        "socket_timeout": 30,
        "retries": 3,
        "max_filesize": MAX_DOWNLOAD_MB * 1024 * 1024,
        # "<=?" lets live/unknown-length videos through the duration check.
        "match_filter": yt_dlp.utils.match_filter_func(
            f"duration <=? {MAX_DURATION_S}"),
        "ffmpeg_location": os.path.dirname(FFMPEG) if FFMPEG else None,
        "logger": QuietLogger(),
    }
    if FFMPEG:
        opts["merge_output_format"] = "mp4"

    job.status = "running"
    job.message = "Starting..."
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.extract_info(url, download=True)
        if job.cancel.is_set():
            raise CancelledByUser()
        candidates = [p for p in job.dir.iterdir()
                      if p.is_file() and p.suffix not in
                      (".part", ".ytdl", ".temp", ".tmp")]
        if not candidates:
            raise RuntimeError(
                "Nothing was downloaded. The video may exceed this server's "
                "size or length limit.")
        output = max(candidates, key=lambda p: p.stat().st_size)
        job.file, job.filename = output, output.name
        job.finish("done", "Ready to save")
    except CancelledByUser:
        job.finish("cancelled", "Cancelled")
    except Exception as exc:  # noqa: BLE001
        if job.cancel.is_set():
            job.finish("cancelled", "Cancelled")
        else:
            job.finish("error", "Failed", clean_error(exc))


# --------------------------------------------------------------------------- #
# Converter
# --------------------------------------------------------------------------- #
def probe_duration(path: Path) -> float | None:
    if not FFPROBE:
        return None
    try:
        result = subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30,
            creationflags=CREATE_NO_WINDOW)
        return float(result.stdout.strip())
    except (ValueError, OSError, subprocess.SubprocessError):
        return None


def convert_worker(job: Job, source: Path, output: Path, fmt: str,
                   bitrate: str) -> None:
    job.status = "running"
    job.message = "Converting..."
    job.progress = 0.0
    command = [FFMPEG, "-y", "-i", str(source), "-vn"]
    if fmt == "mp3":
        command += ["-acodec", "libmp3lame", "-b:a", bitrate]
    else:
        command += ["-acodec", "pcm_s16le"]
    command += ["-progress", "pipe:1", "-nostats", "-loglevel", "error",
                str(output)]

    duration = probe_duration(source)
    tail: list[str] = []
    code = -1
    try:
        job.proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            creationflags=CREATE_NO_WINDOW)
        for line in job.proc.stdout:
            line = line.strip()
            if line.startswith("out_time_ms=") and duration:
                try:
                    seconds = int(line.split("=", 1)[1]) / 1_000_000
                except ValueError:
                    continue
                fraction = min(max(seconds / duration, 0.0), 1.0)
                job.progress = fraction * 0.99
                job.message = f"Converting {fraction * 100:4.1f}%"
            elif line and "=" not in line:
                tail.append(line)
        job.proc.wait()
        code = job.proc.returncode
    except OSError as exc:
        tail.append(str(exc))
    finally:
        job.proc = None
        source.unlink(missing_ok=True)

    if job.cancel.is_set():
        output.unlink(missing_ok=True)
        job.finish("cancelled", "Cancelled")
    elif code == 0 and output.exists():
        job.file, job.filename = output, output.name
        job.finish("done", "Ready to save")
    else:
        output.unlink(missing_ok=True)
        # FFmpeg's own messages contain server file paths, so keep them out
        # of the response.
        job.finish("error", "Failed",
                   "Could not convert this file. It may be corrupted or "
                   "have no audio track.")


# --------------------------------------------------------------------------- #
# API routes
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def api_health():
    return jsonify(
        ytdlp=yt_dlp is not None, ffmpeg=bool(FFMPEG),
        max_upload_mb=MAX_UPLOAD_MB,
        max_duration_min=MAX_DURATION_S // 60,
        extensions=sorted(VIDEO_EXTENSIONS))


@app.post("/api/info")
def api_info():
    check_rate_limit()
    if yt_dlp is None:
        raise ApiError("yt-dlp is not installed on the server.", 503)
    data = request.get_json(silent=True) or {}
    url = validate_url(data.get("url"))
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True,
            "skip_download": True, "socket_timeout": 20,
            "logger": QuietLogger()}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:  # noqa: BLE001
        raise ApiError(clean_error(exc), 422)
    if not info:
        raise ApiError("No information found for that link.", 422)
    if info.get("_type") == "playlist":
        raise ApiError("Playlists aren't supported. "
                       "Paste a link to a single video.", 422)

    duration = info.get("duration")
    heights = sorted(
        {f["height"] for f in info.get("formats") or []
         if f.get("height") and f.get("vcodec") != "none"}, reverse=True)
    thumbnail = info.get("thumbnail") or ""
    if not thumbnail.startswith("https://"):
        thumbnail = ""
    return jsonify(
        title=info.get("title") or "Untitled video",
        uploader=info.get("uploader") or "",
        duration=duration, thumbnail=thumbnail, heights=heights,
        too_long=bool(duration and duration > MAX_DURATION_S),
        max_duration_min=MAX_DURATION_S // 60)


@app.post("/api/download")
def api_download():
    check_rate_limit()
    if yt_dlp is None:
        raise ApiError("yt-dlp is not installed on the server.", 503)
    data = request.get_json(silent=True) or {}
    url = validate_url(data.get("url"))
    height = parse_resolution(data.get("resolution"))
    job = create_job("download")
    threading.Thread(target=download_worker, args=(job, url, height),
                     daemon=True).start()
    return jsonify(job=job.to_dict()), 202


@app.post("/api/convert")
def api_convert():
    check_rate_limit()
    if not FFMPEG:
        raise ApiError("FFmpeg is not installed on the server.", 503)
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        raise ApiError("Please choose a video file.")
    suffix = Path(upload.filename).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise ApiError("Unsupported file type. Choose a video file such as "
                       "MP4, MKV, MOV, AVI or WEBM.")
    fmt = (request.form.get("format") or "mp3").lower()
    if fmt not in ("mp3", "wav"):
        raise ApiError("Format must be MP3 or WAV.")
    bitrate = request.form.get("bitrate") or "192k"
    if bitrate not in MP3_BITRATES:
        bitrate = "192k"

    stem = secure_filename(Path(upload.filename).stem) or "audio"
    job = create_job("convert")
    source = job.dir / f"input{suffix}"
    try:
        upload.save(source)
    except OSError:
        job.finish("error", "Failed", "The upload could not be saved.")
        raise ApiError("The upload could not be saved.", 500)
    output = job.dir / f"{stem[:120]}.{fmt}"
    threading.Thread(target=convert_worker,
                     args=(job, source, output, fmt, bitrate),
                     daemon=True).start()
    return jsonify(job=job.to_dict()), 202


@app.get("/api/jobs/<job_id>")
def api_job(job_id: str):
    return jsonify(get_job(job_id).to_dict())


@app.post("/api/jobs/<job_id>/cancel")
def api_cancel(job_id: str):
    job = get_job(job_id)
    if job.status in ("queued", "running"):
        job.cancel.set()
        proc = job.proc
        if proc and proc.poll() is None:
            proc.terminate()
    return jsonify(job.to_dict())


@app.get("/api/jobs/<job_id>/file")
def api_file(job_id: str):
    job = get_job(job_id)
    if job.status != "done" or not job.file or not job.file.exists():
        raise ApiError("That file is not ready.", 409)
    return send_file(job.file, as_attachment=True,
                     download_name=job.filename, max_age=0)


@app.get("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


# --------------------------------------------------------------------------- #
# Frontend (single page, no external assets)
# --------------------------------------------------------------------------- #
INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Media Downloader &amp; Audio Converter</title>
<style>
  :root {
    --bg: #f4f5f7; --panel: #ffffff; --text: #1b1f27; --muted: #6a7384;
    --line: #dfe2e8; --accent: #2f6fed; --accent-text: #ffffff;
    --ok: #1f9d55; --bad: #d64545; --track: #e7eaf0;
  }
  :root[data-theme="dark"] {
    --bg: #14161b; --panel: #1d2026; --text: #e8eaf0; --muted: #98a0b2;
    --line: #2e323b; --accent: #5b8cff; --accent-text: #0d1220;
    --ok: #3ec27a; --bad: #ff7070; --track: #2a2e37;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #14161b; --panel: #1d2026; --text: #e8eaf0; --muted: #98a0b2;
      --line: #2e323b; --accent: #5b8cff; --accent-text: #0d1220;
      --ok: #3ec27a; --bad: #ff7070; --track: #2a2e37;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font: 15px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }
  .wrap { max-width: 860px; margin: 0 auto; padding: 20px 16px 48px; }
  header { display: flex; align-items: center; gap: 12px; margin-bottom: 18px; }
  h1 { font-size: 20px; margin: 0; flex: 1; }
  button, select, input[type=text] {
    font: inherit; color: var(--text); background: var(--panel);
    border: 1px solid var(--line); border-radius: 8px; padding: 8px 12px;
  }
  button { cursor: pointer; }
  button.primary {
    background: var(--accent); color: var(--accent-text);
    border-color: var(--accent); font-weight: 600;
  }
  button.danger { color: var(--bad); }
  button:disabled { opacity: .5; cursor: not-allowed; }
  input[type=text] { width: 100%; }
  .tabs { display: flex; gap: 6px; margin-bottom: 12px; }
  .tab { border-radius: 8px; background: transparent; }
  .tab[aria-selected="true"] {
    background: var(--panel); border-color: var(--accent); font-weight: 600;
  }
  .panel {
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 12px; padding: 18px;
  }
  .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  .row + .row { margin-top: 12px; }
  .grow { flex: 1; min-width: 200px; }
  label.small { color: var(--muted); font-size: 13px; }
  .alert {
    margin-top: 12px; padding: 10px 12px; border-radius: 8px;
    border: 1px solid var(--bad); color: var(--bad); font-size: 14px;
  }
  .alert.warn { border-color: #c98a1b; color: #c98a1b; }
  [hidden] { display: none !important; }
  .meta { display: flex; gap: 14px; margin-top: 14px; align-items: center; }
  .meta img { width: 140px; border-radius: 8px; background: var(--track); }
  .meta .t { font-weight: 600; }
  .meta .s { color: var(--muted); font-size: 13px; }
  .drop {
    border: 2px dashed var(--line); border-radius: 12px; padding: 28px 12px;
    text-align: center; color: var(--muted); cursor: pointer;
  }
  .drop.over { border-color: var(--accent); color: var(--text); }
  .chips { display: flex; flex-direction: column; gap: 6px; margin-top: 12px; }
  .chip {
    display: flex; align-items: center; gap: 8px; padding: 6px 10px;
    border: 1px solid var(--line); border-radius: 8px; font-size: 14px;
  }
  .chip span.n { flex: 1; overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap; }
  .chip button { padding: 2px 8px; }
  .jobs { margin-top: 16px; display: flex; flex-direction: column; gap: 10px; }
  .job { border: 1px solid var(--line); border-radius: 10px; padding: 12px; }
  .job-title { font-weight: 600; overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap; }
  .bar { height: 8px; background: var(--track); border-radius: 99px;
    margin: 8px 0 6px; overflow: hidden; }
  .bar span { display: block; height: 100%; width: 0; background: var(--accent);
    transition: width .25s; }
  .job.done .bar span { background: var(--ok); }
  .job.error .bar span, .job.cancelled .bar span { background: var(--bad); }
  .status { font-size: 13px; color: var(--muted); }
  .job.error .status { color: var(--bad); }
  .actions { margin-top: 8px; display: flex; gap: 8px; }
  a.btn {
    display: inline-block; text-decoration: none; background: var(--ok);
    color: #fff; font-weight: 600; border-radius: 8px; padding: 8px 14px;
  }
  .logbox { margin-top: 18px; }
  .logbox h2 { font-size: 14px; margin: 0 0 6px; display: flex; gap: 8px;
    align-items: center; }
  .logbox h2 span { flex: 1; }
  #log {
    height: 150px; overflow: auto; background: var(--panel);
    border: 1px solid var(--line); border-radius: 10px; padding: 8px 10px;
    font: 12px/1.5 ui-monospace, Consolas, monospace; white-space: pre-wrap;
  }
  footer { margin-top: 20px; color: var(--muted); font-size: 12px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Media Downloader &amp; Audio Converter</h1>
    <button id="themeBtn" title="Switch light / dark">Theme</button>
  </header>

  <div id="banner" class="alert warn" hidden></div>

  <div class="tabs" role="tablist">
    <button class="tab" role="tab" id="tab-dl" aria-selected="true">
      Video Downloader</button>
    <button class="tab" role="tab" id="tab-cv" aria-selected="false">
      Audio Converter</button>
  </div>

  <!-- Downloader -->
  <section class="panel" id="panel-dl">
    <div class="row">
      <div class="grow"><input type="text" id="url" autocomplete="off"
        placeholder="Paste a video link (https://...)"></div>
      <button id="fetchBtn">Fetch info</button>
    </div>
    <div class="meta" id="meta" hidden>
      <img id="thumb" alt="" referrerpolicy="no-referrer">
      <div><div class="t" id="mTitle"></div><div class="s" id="mSub"></div></div>
    </div>
    <div class="row">
      <label class="small" for="res">Resolution</label>
      <select id="res"></select>
      <button class="primary" id="dlBtn">Download</button>
    </div>
    <div class="alert" id="dlAlert" hidden></div>
    <div class="jobs" id="dlJobs"></div>
  </section>

  <!-- Converter -->
  <section class="panel" id="panel-cv" hidden>
    <div class="drop" id="drop">
      <div id="dropText">Drag &amp; drop video files here, or click to choose</div>
      <input type="file" id="file" multiple hidden>
    </div>
    <div class="chips" id="chips"></div>
    <div class="row">
      <label class="small" for="fmt">Format</label>
      <select id="fmt"><option value="mp3">MP3</option>
        <option value="wav">WAV</option></select>
      <span id="brWrap" class="row" style="margin:0">
        <label class="small" for="br">Bitrate</label>
        <select id="br"><option>128k</option><option selected>192k</option>
          <option>256k</option><option>320k</option></select>
      </span>
      <button class="primary" id="cvBtn" disabled>Convert</button>
    </div>
    <div class="alert" id="cvAlert" hidden></div>
    <div class="jobs" id="cvJobs"></div>
  </section>

  <div class="logbox">
    <h2><span>Activity log</span><button id="clearLog">Clear</button></h2>
    <div id="log" aria-live="polite"></div>
  </div>
  <footer>Only download or convert content you have the right to use.
    Finished files are deleted from the server automatically.</footer>
</div>

<script>
"use strict";
const $ = (sel) => document.querySelector(sel);
let config = { max_upload_mb: 500, extensions: [], ffmpeg: true, ytdlp: true };
let pendingFiles = [];

/* ---------- tiny helpers ---------- */
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid != null) el.append(kid);
  return el;
}
function log(msg) {
  const box = $("#log");
  const t = new Date().toLocaleTimeString();
  box.append(document.createTextNode(`[${t}] ${msg}\n`));
  box.scrollTop = box.scrollHeight;
}
function fmtDuration(s) {
  if (s == null) return "";
  s = Math.round(s);
  const h_ = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), x = s % 60;
  const p = (n) => String(n).padStart(2, "0");
  return h_ ? `${h_}:${p(m)}:${p(x)}` : `${m}:${p(x)}`;
}
function fmtBytes(n) {
  if (n == null) return "";
  const u = ["B", "KB", "MB", "GB"]; let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i ? 1 : 0)} ${u[i]}`;
}
function looksLikeUrl(v) {
  try { const u = new URL(v); return u.protocol === "http:" || u.protocol === "https:"; }
  catch { return false; }
}
function showAlert(id, msg) {
  const el = $(id);
  el.textContent = msg || "";
  el.hidden = !msg;
}
async function api(path, options) {
  let res;
  try { res = await fetch(path, options); }
  catch { throw new Error("Could not reach the server."); }
  let data = {};
  try { data = await res.json(); } catch { /* not json */ }
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}
function postJson(path, body) {
  return api(path, { method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body) });
}

/* ---------- theme & tabs ---------- */
function applyTheme(mode) {
  document.documentElement.dataset.theme = mode;
  try { localStorage.setItem("theme", mode); } catch { /* ignore */ }
}
(function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("theme"); } catch { /* ignore */ }
  if (saved) document.documentElement.dataset.theme = saved;
})();
$("#themeBtn").onclick = () => {
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  const cur = document.documentElement.dataset.theme || (dark ? "dark" : "light");
  applyTheme(cur === "dark" ? "light" : "dark");
};
function selectTab(which) {
  const dl = which === "dl";
  $("#tab-dl").setAttribute("aria-selected", String(dl));
  $("#tab-cv").setAttribute("aria-selected", String(!dl));
  $("#panel-dl").hidden = !dl;
  $("#panel-cv").hidden = dl;
}
$("#tab-dl").onclick = () => selectTab("dl");
$("#tab-cv").onclick = () => selectTab("cv");
$("#clearLog").onclick = () => { $("#log").textContent = ""; };

/* ---------- job cards ---------- */
function createCard(title) {
  const fill = h("span");
  const status = h("div", { class: "status" }, "Starting...");
  const actions = h("div", { class: "actions" });
  const el = h("div", { class: "job" },
    h("div", { class: "job-title", title }, title),
    h("div", { class: "bar" }, fill), status, actions);
  return {
    el,
    progress(f) { fill.style.width = `${Math.max(0, Math.min(1, f)) * 100}%`; },
    text(t) { status.textContent = t; },
    state(s) { el.className = "job " + s; },
    actions(...nodes) { actions.replaceChildren(...nodes); },
  };
}

function track(jobId, card, label) {
  const finalStates = ["done", "error", "cancelled"];
  const cancelBtn = h("button", { class: "danger" }, "Cancel");
  cancelBtn.onclick = async () => {
    cancelBtn.disabled = true;
    try { await api(`/api/jobs/${jobId}/cancel`, { method: "POST" }); }
    catch (e) { log(`Cancel failed: ${e.message}`); }
  };
  card.actions(cancelBtn);

  const tick = async () => {
    let job;
    try { job = await api(`/api/jobs/${jobId}`); }
    catch (e) {
      card.state("error"); card.text(e.message); card.actions();
      log(`${label}: ${e.message}`);
      return;
    }
    card.progress(job.progress);
    card.state(job.status);
    if (job.status === "done") {
      card.text(`Ready: ${job.filename} (${fmtBytes(job.size)})`);
      const save = h("a", { class: "btn", href: `/api/jobs/${jobId}/file`,
        download: job.filename }, "Save file");
      card.actions(save);
      log(`${label}: finished -> ${job.filename}`);
      return;
    }
    if (job.status === "error") {
      card.text(job.error || "Failed"); card.actions();
      log(`${label}: failed - ${job.error}`);
      return;
    }
    if (job.status === "cancelled") {
      card.text("Cancelled"); card.actions();
      log(`${label}: cancelled`);
      return;
    }
    card.text(job.message);
    if (!finalStates.includes(job.status)) setTimeout(tick, 700);
  };
  tick();
}

/* ---------- downloader ---------- */
function fillResolutions(heights) {
  const sel = $("#res");
  sel.replaceChildren(h("option", { value: "Best available" }, "Best available"));
  const list = heights && heights.length ? heights
    : [2160, 1440, 1080, 720, 480, 360, 240, 144];
  for (const hgt of list) sel.append(h("option", { value: `${hgt}p` }, `${hgt}p`));
}
fillResolutions();

let fetchedUrl = "", fetchedTitle = "";
$("#fetchBtn").onclick = async () => {
  showAlert("#dlAlert", "");
  const url = $("#url").value.trim();
  if (!url) return showAlert("#dlAlert", "Please enter a video URL.");
  if (!looksLikeUrl(url)) return showAlert("#dlAlert",
    "That doesn't look like a valid link. It must start with http:// or https://");
  const btn = $("#fetchBtn");
  btn.disabled = true; btn.textContent = "Fetching...";
  log(`Fetching info: ${url}`);
  try {
    const info = await postJson("/api/info", { url });
    fetchedUrl = url; fetchedTitle = info.title;
    $("#mTitle").textContent = info.title;
    $("#mSub").textContent = [info.uploader, fmtDuration(info.duration)]
      .filter(Boolean).join("  ·  ");
    const img = $("#thumb");
    img.hidden = !info.thumbnail;
    if (info.thumbnail) img.src = info.thumbnail;
    $("#meta").hidden = false;
    fillResolutions(info.heights);
    if (info.too_long) showAlert("#dlAlert",
      `This video is longer than the ${info.max_duration_min}-minute limit ` +
      "and will probably be refused.");
    log(`Found: ${info.title}`);
  } catch (e) {
    showAlert("#dlAlert", e.message);
    log(`Could not fetch info: ${e.message}`);
  } finally {
    btn.disabled = false; btn.textContent = "Fetch info";
  }
};
$("#url").addEventListener("keydown", (e) => {
  if (e.key === "Enter") $("#fetchBtn").click();
});

$("#dlBtn").onclick = async () => {
  showAlert("#dlAlert", "");
  const url = $("#url").value.trim();
  if (!url) return showAlert("#dlAlert", "Please enter a video URL.");
  if (!looksLikeUrl(url)) return showAlert("#dlAlert",
    "That doesn't look like a valid link. It must start with http:// or https://");
  const btn = $("#dlBtn");
  btn.disabled = true;
  try {
    const { job } = await postJson("/api/download",
      { url, resolution: $("#res").value });
    const label = (url === fetchedUrl && fetchedTitle) ? fetchedTitle : url;
    const card = createCard(label);
    $("#dlJobs").prepend(card.el);
    log(`Starting download (${$("#res").value}): ${url}`);
    track(job.id, card, label);
  } catch (e) {
    showAlert("#dlAlert", e.message);
    log(`Download not started: ${e.message}`);
  } finally {
    btn.disabled = false;
  }
};

/* ---------- converter ---------- */
function renderChips() {
  const box = $("#chips");
  box.replaceChildren(...pendingFiles.map((f, i) => {
    const rm = h("button", { title: "Remove" }, "×");
    rm.onclick = () => { pendingFiles.splice(i, 1); renderChips(); };
    return h("div", { class: "chip" },
      h("span", { class: "n" }, f.name),
      h("span", { class: "s" }, fmtBytes(f.size)), rm);
  }));
  $("#cvBtn").disabled = pendingFiles.length === 0;
}
function addFiles(fileList) {
  showAlert("#cvAlert", "");
  const problems = [];
  for (const f of fileList) {
    const ext = "." + (f.name.split(".").pop() || "").toLowerCase();
    if (config.extensions.length && !config.extensions.includes(ext)) {
      problems.push(`${f.name}: not a supported video type`); continue;
    }
    if (f.size > config.max_upload_mb * 1024 * 1024) {
      problems.push(`${f.name}: larger than ${config.max_upload_mb} MB`); continue;
    }
    pendingFiles.push(f);
  }
  if (problems.length) showAlert("#cvAlert", problems.join(" · "));
  renderChips();
}
const drop = $("#drop");
drop.onclick = () => $("#file").click();
$("#file").onchange = (e) => { addFiles(e.target.files); e.target.value = ""; };
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => {
  e.preventDefault(); drop.classList.add("over");
}));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => {
  e.preventDefault(); drop.classList.remove("over");
}));
drop.addEventListener("drop", (e) => addFiles(e.dataTransfer.files));
$("#fmt").onchange = () => { $("#brWrap").hidden = $("#fmt").value !== "mp3"; };

function uploadFile(file, fmt, bitrate, card) {
  return new Promise((resolve, reject) => {
    const fd = new FormData();
    fd.append("file", file); fd.append("format", fmt); fd.append("bitrate", bitrate);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/convert");
    xhr.upload.onprogress = (e) => {
      if (!e.lengthComputable) return;
      card.progress(e.loaded / e.total);
      card.text(`Uploading ${Math.round((e.loaded / e.total) * 100)}%`);
    };
    xhr.onload = () => {
      let data = {};
      try { data = JSON.parse(xhr.responseText); } catch { /* ignore */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data);
      else reject(new Error(data.error || `Upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error("Network error during upload."));
    xhr.send(fd);
  });
}

$("#cvBtn").onclick = async () => {
  const files = pendingFiles.slice();
  pendingFiles = []; renderChips();
  const fmt = $("#fmt").value, bitrate = $("#br").value;
  $("#cvBtn").disabled = true;
  for (const file of files) {
    const card = createCard(file.name);
    $("#cvJobs").prepend(card.el);
    log(`Uploading ${file.name}`);
    try {
      const { job } = await uploadFile(file, fmt, bitrate, card);
      card.progress(0);
      track(job.id, card, file.name);
    } catch (e) {
      card.state("error"); card.text(e.message);
      log(`${file.name}: ${e.message}`);
    }
  }
  $("#cvBtn").disabled = pendingFiles.length === 0;
};

/* ---------- startup ---------- */
(async function init() {
  try {
    config = await api("/api/health");
    const notes = [];
    if (!config.ytdlp) notes.push("yt-dlp is not installed on the server, so downloads are disabled.");
    if (!config.ffmpeg) notes.push("FFmpeg is not installed on the server: the converter is disabled and downloads may be lower quality.");
    if (notes.length) { $("#banner").textContent = notes.join(" "); $("#banner").hidden = false; }
    $("#dropText").textContent =
      `Drag & drop video files here, or click to choose (max ${config.max_upload_mb} MB each)`;
    if (!config.ffmpeg) showAlert("#cvAlert",
      "FFmpeg is not installed on the server, so conversion is unavailable.");
    log("Ready.");
  } catch (e) { log(`Could not reach the server: ${e.message}`); }
})();
</script>
</body>
</html>
"""


_start_background()

if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"),
            port=_env_int("PORT", 5000), threaded=True)
