#!/usr/bin/env python3
"""Preview enhanced prompt without running full video generation.

Outputs JSON: {"enhanced_prompt": "..."} or {"error": "..."}

When enhancement returns empty (e.g. safety filter), auto-retries with suspected
filtered words replaced by placeholders, then merges originals back into the result.

Always uses MLX uncensored Gemma via mlx_lm (~7GB, no Lightricks/LTX-2 download).
"""

import argparse
import json
import re
import sys
from pathlib import Path

ENHANCER_MODEL = "TheCluster/amoral-gemma-3-12B-v2-mlx-4bit"

# Words that commonly trigger Gemma/content filters (lowercase)
SUSPECTED_FILTERED_WORDS = [
    "piss",
    "urine",
    "blood",
    "gore",
    "corpse",
    "dead body",
    "vomit",
    "vomiting",
    "naked",
    "nude",
    "sex",
    "sexual",
]


def _sanitize_prompt(prompt: str) -> tuple[str, dict[str, str]]:
    """Replace suspected filtered words with placeholders. Returns (sanitized, {placeholder: original})."""
    replacements: dict[str, str] = {}
    result = prompt
    for i, word in enumerate(SUSPECTED_FILTERED_WORDS):
        pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.I)
        for m in reversed(list(pattern.finditer(result))):
            ph = f"__X{i}__"
            replacements[ph] = m.group()
            result = result[: m.start()] + ph + result[m.end() :]
    return result, replacements


def _merge_back(enhanced: str, replacements: dict[str, str]) -> str:
    """Restore original words from placeholders."""
    result = enhanced
    for ph, orig in replacements.items():
        result = result.replace(ph, orig)
    return result


def _apply_chat_template(tokenizer, system_prompt: str, user_content: str) -> str:
    """Apply the tokenizer's native chat template with a proper system turn."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _enhance_with_mlx_lm(
    prompt: str,
    model_repo: str,
    system_prompt: str | None,
    temperature: float,
    seed: int,
    max_tokens: int,
    verbose: bool,
) -> str:
    """Enhance prompt using mlx_lm with given MLX model. No Lightricks/LTX-2 download."""
    try:
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler
    except ImportError:
        print("mlx-lm not available. Install: pip install mlx-lm", file=sys.stderr)
        return prompt

    print(
        f"Loading prompt enhancer ({model_repo}, first run ~7GB)...",
        file=sys.stderr,
        flush=True,
    )
    # Pre-download the model so we can report real progress instead of a silent
    # hang inside load(). Emit DOWNLOAD:* tokens on stderr for the Swift side.
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        from tqdm import tqdm

        class _ByteProgress(tqdm):
            """Emit per-file byte progress as DOWNLOAD:BYTES tokens."""
            def __init__(self, *args, **kwargs):
                kwargs.setdefault("disable", False)
                super().__init__(*args, **kwargs)

            def update(self, n=1):
                super().update(n)
                if self.total:
                    pct = int(100 * self.n / self.total)
                    print(
                        f"DOWNLOAD:BYTES:{pct}:{self.n}:{self.total}",
                        file=sys.stderr,
                        flush=True,
                    )

        # NOTE: do NOT set HF_HUB_DISABLE_PROGRESS_BARS here — we rely on tqdm
        # running so _ByteProgress.update() emits DOWNLOAD:BYTES tokens.
        # Remove stale *.incomplete blobs so interrupted downloads don't cause
        # lock contention or dead weight; huggingface_hub >= 1.26 doesn't resume them.
        try:
            import glob as _glob
            from huggingface_hub.constants import HF_HUB_CACHE
            repo_dir = os.path.join(HF_HUB_CACHE, "models--" + model_repo.replace("/", "--"), "blobs")
            for inc in _glob.glob(os.path.join(repo_dir, "*.incomplete")):
                try:
                    os.remove(inc)
                    print(f"CLEANED:{os.path.basename(inc)}", file=sys.stderr, flush=True)
                except OSError:
                    pass
        except Exception as e:
            print(f"Incomplete-cache cleanup skipped: {e}", file=sys.stderr, flush=True)

        print(f"DOWNLOAD:START:{model_repo}", file=sys.stderr, flush=True)
        files = list_repo_files(model_repo)
        total = len(files)
        for idx, filename in enumerate(files, start=1):
            print(
                f"DOWNLOAD:PROGRESS:{idx}:{total}:{model_repo}:{filename}",
                file=sys.stderr,
                flush=True,
            )
            try:
                hf_hub_download(
                    repo_id=model_repo,
                    filename=filename,
                    tqdm_class=_ByteProgress,
                )
            except Exception as e:
                print(f"FILE_FAILED:{filename}:{e}", file=sys.stderr, flush=True)
        print(f"DOWNLOAD:COMPLETE:{model_repo}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"PREDOWNLOAD_ERROR:{type(e).__name__}:{e}", file=sys.stderr, flush=True)

    print("STATUS:Loading prompt enhancer model...", file=sys.stderr, flush=True)
    model, tokenizer = load(model_repo)

    if system_prompt is None:
        try:
            from mlx_video.models.ltx.enhance_prompt import _load_system_prompt

            system_prompt = _load_system_prompt("gemma_t2v_system_prompt.txt")
        except Exception:
            system_prompt = "You are a creative writer. Expand the user's short video prompt into a detailed, vivid description suitable for AI video generation. Include lighting, camera movement, and atmosphere."

    user_content = prompt
    formatted = _apply_chat_template(tokenizer, system_prompt, user_content)

    import mlx.core as mx

    mx.random.seed(seed)

    # mlx-lm 0.25+ uses sampler instead of temp kwarg (generate_step rejects temp)
    sampler = make_sampler(temperature, 1.0, 0.0, 1, top_k=0)

    # Stream generation and stop cleanly at the model's end-of-turn token.
    from mlx_lm import stream_generate

    eos_id = tokenizer.eos_token_id
    pieces: list[str] = []
    for response in stream_generate(
        model,
        tokenizer,
        prompt=formatted,
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        if response.finish_reason is not None:
            break
        if eos_id is not None and response.token == eos_id:
            break
        pieces.append(response.text)
    response_text = "".join(pieces).strip()

    del model
    mx.clear_cache()
    return response_text


def main():
    parser = argparse.ArgumentParser(description="Preview Gemma-enhanced prompt")
    parser.add_argument("--prompt", "-p", required=True, help="User prompt to enhance")
    parser.add_argument(
        "--model-repo",
        default="notapalindrome/ltx23-mlx-av-q4",
        help="Generation model repository (from app selection)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature for enhancement",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Image path for I2V (uses i2v system prompt if set)",
    )
    parser.add_argument(
        "--resources-path",
        default=None,
        help="App Resources path for bundled prompts (pre-flight injection)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    try:
        # Pre-flight: inject bundled prompts if mlx_video is missing them
        if args.resources_path:
            try:
                from pathlib import Path as P
                import shutil

                resources_path = P(args.resources_path)
                bundled_prompts = resources_path / "prompts"
                import mlx_video.models.ltx.text_encoder as te

                target_dir = P(te.__file__).parent / "prompts"
                for name in [
                    "gemma_t2v_system_prompt.txt",
                    "gemma_i2v_system_prompt.txt",
                ]:
                    src = bundled_prompts / name
                    dst = target_dir / name
                    if src.exists() and not dst.exists():
                        target_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dst)
            except Exception:
                pass

        # Always use MLX uncensored Gemma via mlx_lm
        model_repo = ENHANCER_MODEL

        system_prompt = None
        if args.image:
            try:
                from mlx_video.models.ltx.enhance_prompt import _load_system_prompt

                system_prompt = _load_system_prompt("gemma_i2v_system_prompt.txt")
            except Exception:
                pass

        def do_enhance(p: str):
            return _enhance_with_mlx_lm(
                p,
                model_repo=model_repo,
                system_prompt=system_prompt,
                temperature=args.temperature,
                seed=args.seed,
                max_tokens=256,
                verbose=False,
            )

        enhanced = do_enhance(args.prompt)

        # Auto-retry with sanitized prompt when enhancement returns empty (filtered)
        if not enhanced or not enhanced.strip():
            sanitized, replacements = _sanitize_prompt(args.prompt)
            if replacements:
                enhanced = do_enhance(sanitized)
                if enhanced and enhanced.strip():
                    enhanced = _merge_back(enhanced, replacements)

        print(json.dumps({"enhanced_prompt": enhanced or ""}))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
