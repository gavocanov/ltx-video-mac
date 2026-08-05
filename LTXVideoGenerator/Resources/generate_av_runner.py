#!/usr/bin/env python3
"""LTX-2 Unified AV generation runner.

Invoked by the macOS app as a standalone script with explicit CLI arguments.
Runs the vendored generate_av.py and streams progress/status tokens on stderr
so the Swift side can report real-time progress.
"""
import os
import sys
import json
import subprocess
import time
import select
import signal
import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--log-file", required=True)
    p.add_argument("--model-repo", required=True)
    p.add_argument("--text-encoder-repo", required=True)
    p.add_argument("--use-local-pref", type=int, default=0)
    p.add_argument("--image-path", default="")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--num-frames", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--fps", type=int, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--cfg-scale", type=float, required=True)
    p.add_argument("--output-path", required=True)
    p.add_argument("--tiling", required=True)
    p.add_argument("--preview-every", type=int, default=0)
    p.add_argument("--preview-dir", default="")
    p.add_argument("--disable-audio", action="store_true")
    p.add_argument("--image-strength", type=float, default=0.0)
    p.add_argument("--lora-path", default="")
    p.add_argument("--lora-strength", type=float, default=0.0)
    p.add_argument("--save-audio-separately", action="store_true")
    p.add_argument("--resources-path", required=True)
    return p.parse_args()


def main():
    args = parse_args()

    log_file = open(args.log_file, "w")

    def log(msg):
        print(msg, file=log_file, flush=True)
        print(msg, file=sys.stderr, flush=True)

    try:
        log("=== LTX-2 Unified AV Generation Started ===")
        log(f"Python: {sys.executable}")

        # Check MLX
        import mlx.core as mx
        log("MLX device: Apple Silicon")

        model_repo = args.model_repo
        text_encoder_repo = args.text_encoder_repo
        log(f"Model: {model_repo}")
        log(f"Text encoder: {text_encoder_repo}")
        local_mlx_video_repo = os.path.expanduser("~/projects/mlx-video-with-audio")
        local_has_mlx = os.path.exists(os.path.join(local_mlx_video_repo, "mlx_video", "generate_av.py"))

        def _mlx_version_subprocess(extra_env):
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            for k, v in extra_env.items():
                env[k] = v
            r = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import mlx_video.version; print(mlx_video.version.__version__)",
                ],
                capture_output=True,
                text=True,
                env=env,
                timeout=120,
            )
            if r.returncode != 0:
                return None
            return (r.stdout or "").strip() or None

        def _parse_version_tuple(s):
            if not s:
                return None
            parts = []
            for seg in s.split("."):
                digits = "".join(c for c in seg if c.isdigit())
                try:
                    parts.append(int(digits) if digits else 0)
                except Exception:
                    parts.append(0)
            return tuple(parts)

        pip_ver = _mlx_version_subprocess({})
        local_ver = _mlx_version_subprocess({"PYTHONPATH": local_mlx_video_repo}) if local_has_mlx else None
        use_local_pref = bool(args.use_local_pref)
        force_local = os.environ.get("LTX_FORCE_LOCAL_MLX_VIDEO") == "1" or use_local_pref

        if not local_has_mlx:
            use_local_mlx_video_repo = False
        elif force_local:
            use_local_mlx_video_repo = True
        elif pip_ver and local_ver:
            use_local_mlx_video_repo = _parse_version_tuple(local_ver) > _parse_version_tuple(pip_ver)
        else:
            use_local_mlx_video_repo = bool(local_ver) and not pip_ver

        log(
            "mlx-video-with-audio versions: pip=%r local_repo=%r -> use_local_repo=%s"
            % (pip_ver, local_ver, use_local_mlx_video_repo)
        )
        if local_has_mlx and not use_local_mlx_video_repo and pip_ver and local_ver:
            if _parse_version_tuple(pip_ver) >= _parse_version_tuple(local_ver):
                log(
                    "Using pip/site-packages (newer or same as ~/projects/mlx-video-with-audio). "
                    "Preferences: enable 'Use local mlx-video-with-audio repo' or set LTX_FORCE_LOCAL_MLX_VIDEO=1 to override."
                )

        # Image-to-video mode
        source_image_path = args.image_path or None
        mode = "image-to-video" if source_image_path else "text-to-video"

        prompt = args.prompt
        negative_prompt = args.negative_prompt
        log(f"Prompt: {prompt[:100]}...")
        log(f"Size: {args.width}x{args.height}, {args.num_frames} frames")
        log(f"Seed: {args.seed}")

        disable_audio = args.disable_audio

        # Run our vendored generate_av.py (patched for latent previews) instead of
        # `python -m mlx_video.generate_av`, so we can emit PREVIEW frames mid-denoise.
        generate_av_script = os.path.join(args.resources_path, "generate_av.py")
        if not os.path.exists(generate_av_script):
            raise RuntimeError(f"Vendored generate_av.py not found at {generate_av_script}")

        cmd = [
            sys.executable, generate_av_script,
            "--prompt", prompt,
            "--height", str(args.height),
            "--width", str(args.width),
            "--num-frames", str(args.num_frames),
            "--seed", str(args.seed),
            "--fps", str(args.fps),
            "--steps", str(args.steps),
            "--cfg-scale", str(args.cfg_scale),
            "--output-path", args.output_path,
            "--model-repo", model_repo,
            "--text-encoder-repo", text_encoder_repo,
            "--tiling", args.tiling,
        ]
        if args.preview_every > 0:
            cmd.extend(["--preview-every", str(args.preview_every)])
            cmd.extend(["--preview-dir", args.preview_dir])
        if negative_prompt.strip():
            cmd.extend(["--negative-prompt", negative_prompt])
        if disable_audio:
            cmd.append("--no-audio")
            log(f"Mode: {mode} (audio disabled; video-only mux)")
        else:
            log(f"Mode: {mode} (with audio)")

        # Add image conditioning if provided
        if source_image_path:
            cmd.extend(["--image", source_image_path])
            cmd.extend(["--image-strength", str(args.image_strength)])
            log(f"Image conditioning: {source_image_path}")

        # Add LoRA adapter if provided
        lora_path = args.lora_path
        if lora_path:
            cmd.extend(["--lora-path", lora_path])
            cmd.extend(["--lora-strength", str(args.lora_strength)])
            log(f"LoRA: {lora_path} (strength {args.lora_strength})")

        if (not disable_audio) and args.save_audio_separately:
            cmd.append("--save-audio-separately")
            log("Saving audio track separately")

        log("Starting generation...")
        log(f"Command: {' '.join(cmd)}")

        # Pre-download model + text encoder so mlx_video.generate_av finds them cached
        # and we can report real progress instead of silent downloads.
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
            from huggingface_hub.constants import HF_HUB_CACHE
            # Suppress huggingface_hub's own tqdm bars; we emit our own progress lines.
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

            # huggingface_hub >= 1.26 does NOT resume interrupted downloads across
            # process restarts (it uses a process-unique temp file). Any partial left
            # by a killed/cancelled run is dead weight that would otherwise accumulate
            # and never be reused. Remove stale *.incomplete blobs for these repos so
            # disk isn't wasted and downloads start clean.
            try:
                import glob as _glob
                for repo in (model_repo, text_encoder_repo):
                    repo_dir = os.path.join(HF_HUB_CACHE, "models--" + repo.replace("/", "--"), "blobs")
                    for inc in _glob.glob(os.path.join(repo_dir, "*.incomplete")):
                        try:
                            os.remove(inc)
                            log(f"Removed stale incomplete: {os.path.basename(inc)}")
                        except OSError:
                            pass
            except Exception as e:
                log(f"Incomplete-cache cleanup skipped: {e}")

            for repo, label in ((model_repo, "model"), (text_encoder_repo, "text encoder")):
                print(f"DOWNLOAD:START:{repo}", file=sys.stderr, flush=True)
                files = list_repo_files(repo)
                total = len(files)
                for idx, filename in enumerate(files, start=1):
                    print(
                        f"DOWNLOAD:PROGRESS:{idx}:{total}:{repo}:{filename}",
                        file=sys.stderr,
                        flush=True,
                    )
                    hf_hub_download(repo_id=repo, filename=filename)
                print(f"DOWNLOAD:COMPLETE:{repo}", file=sys.stderr, flush=True)
        except Exception as e:
            log(f"Pre-download failed (will rely on lazy download): {e}")

        child_env = os.environ.copy()
        # Drop inherited PYTHONPATH so venv site-packages wins unless we explicitly use a local checkout.
        child_env.pop("PYTHONPATH", None)
        if use_local_mlx_video_repo:
            child_env["PYTHONPATH"] = local_mlx_video_repo

        # Run the CLI module and stream combined output (binary read so we see tqdm \r updates)
        process = subprocess.Popen(
            cmd,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )

        # Unbuffered read: use os.read() on the raw fd so every line the inner process
        # flushes is available immediately.  process.stdout.read(n) uses Python's
        # BufferedReader which blocks until n bytes accumulate, starving the progress loop.
        line_buf = ""
        interactivity_watchdog = False
        download_in_progress = False
        last_download_activity = None  # None = not in download phase yet; set when first download line seen
        # Large models can go quiet between tqdm updates; heartbeats + long window avoid false kills.
        download_stall_timeout = 7200  # 2 hours without *any* subprocess output while downloading
        stdout_fd = process.stdout.fileno()
        while True:
            if process.stdout is None:
                break
            ready, _, _ = select.select([process.stdout], [], [], 1.0)
            if ready:
                try:
                    raw = os.read(stdout_fd, 8192)
                except (ValueError, OSError):
                    raw = b""
                if not raw:
                    if process.poll() is not None:
                        break
                    continue
                # Decode and treat any received data as activity when we're in download phase
                try:
                    chunk = raw.decode("utf-8", errors="replace")
                except Exception:
                    chunk = ""
                if download_in_progress and last_download_activity is not None:
                    last_download_activity = time.time()
                line_buf += chunk
                # Partial tqdm line (e.g. "  3%|") also counts as download activity
                if "%" in line_buf and "|" in line_buf:
                    download_in_progress = True
                    if last_download_activity is None:
                        last_download_activity = time.time()
                _nl = "\n"
                _cr = "\r"
                while _nl in line_buf or _cr in line_buf:
                    line, sep, rest = line_buf.partition(_nl)
                    if not sep:
                        line, sep, rest = line_buf.partition(_cr)
                    line_buf = rest if sep else line_buf
                    if not sep:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    log(line)
                    low = line.lower()
                    if ("impacting interactivity" in low) or ("kiogpucommandbuffercallbackerrorimpactinginteractivity" in low):
                        interactivity_watchdog = True
                    if ("fetching" in low) or ("downloading" in low) or line.startswith("DOWNLOAD:") or ("%" in line and "|" in line):
                        download_in_progress = True
                        if last_download_activity is None:
                            last_download_activity = time.time()
                    if line.startswith("STAGE:") or "generation..." in low or "decoding" in low:
                        download_in_progress = False
                        last_download_activity = None
                    # Emit explicit phase statuses so UI doesn't look frozen after denoising
                    if "decoding video" in low:
                        print("STATUS:Decoding video...", file=sys.stderr, flush=True)
                    elif "video encoded" in low:
                        print("STATUS:Saving video frames...", file=sys.stderr, flush=True)
                    elif "decoding audio" in low:
                        print("STATUS:Decoding audio...", file=sys.stderr, flush=True)
                    elif "combining video and audio" in low:
                        print("STATUS:Saving final video...", file=sys.stderr, flush=True)
                    elif "saved video with audio" in low:
                        print("STATUS:Saving final video...", file=sys.stderr, flush=True)
                    print(line, file=sys.stderr, flush=True)
            else:
                if process.poll() is not None:
                    break
                # Only enforce stall when we've seen download start and then no data for timeout
                if download_in_progress and last_download_activity is not None:
                    stalled_for = int(time.time() - last_download_activity)
                    if stalled_for >= download_stall_timeout:
                        print(f"DOWNLOAD:STALL:{stalled_for}", file=sys.stderr, flush=True)
                        log(f"ERROR: model download stalled for {stalled_for}s (no data received)")
                        process.kill()
                        raise TimeoutError(f"Model download stalled for {stalled_for}s")

        process.wait()

        if process.returncode != 0:
            if process.returncode < 0:
                signal_num = -process.returncode
                signal_name = signal.Signals(signal_num).name if signal_num in signal.Signals._value2member_map_ else f"signal {signal_num}"
                if signal_num == signal.SIGKILL:
                    log("DIAGNOSTIC_SIGKILL: mlx_video.generate_av killed by SIGKILL (exit code -9); likely macOS memory pressure during text encoder / model eval.")
                    raise RuntimeError(
                        "mlx_video.generate_av was killed by SIGKILL (exit code -9). "
                        "macOS often sends SIGKILL under unified memory pressure (jetsam). "
                        "Try a smaller text encoder in Preferences, aggressive VAE tiling, lower resolution or frames, or close other apps. "
                        "Full output is in /tmp/ltx_generation.log."
                    )
                if signal_num == signal.SIGABRT:
                    if interactivity_watchdog:
                        log("DIAGNOSTIC_METAL_INTERACTIVITY: SIGABRT after Metal Impacting Interactivity watchdog.")
                        raise RuntimeError(
                            "DIAGNOSTIC_METAL_INTERACTIVITY: mlx_video.generate_av aborted with SIGABRT (code -6) after "
                            "[METAL] Impacting Interactivity / kIOGPUCommandBufferCallbackErrorImpactingInteractivity. "
                            "Try VAE tiling auto or conservative instead of aggressive; reduce resolution or frames."
                        )
                    raise RuntimeError(
                        "mlx_video.generate_av aborted with SIGABRT (code -6). "
                        "This is usually a native MLX/Metal abort, often from peak unified-memory pressure "
                        "or the macOS Metal watchdog terminating a long-running command buffer. "
                        "Update mlx-video-with-audio, then retry with lower-memory settings if needed. "
                        "Full subprocess output is in /tmp/ltx_generation.log."
                    )
                raise RuntimeError(
                    f"mlx_video.generate_av was terminated by {signal_name} (code {process.returncode}). "
                    "Full subprocess output is in /tmp/ltx_generation.log."
                )
            raise RuntimeError(f"mlx_video.generate_av failed with code {process.returncode}")

        log(f"Video with audio saved to: {args.output_path}")
        log("Generation complete!")
        log_file.close()
        print(json.dumps({"video_path": args.output_path, "seed": args.seed, "mode": mode, "has_audio": not disable_audio}))
    except Exception as e:
        log(f"ERROR: {e}")
        import traceback
        log(traceback.format_exc())
        log_file.close()
        sys.exit(1)


if __name__ == "__main__":
    main()
