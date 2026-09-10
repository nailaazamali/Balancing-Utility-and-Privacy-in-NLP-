from __future__ import annotations

import re

import torch

from .metrics import structural_quality


def build_prompt(models, source: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Rewrite the offensive or hateful tweet into a non-hateful and non-offensive "
                "version. Preserve the principal topic and modify the wording only as much as "
                "necessary. Keep the rewrite fluent and natural. Output only the rewritten text."
            ),
        },
        {"role": "user", "content": str(source)},
    ]
    return models.gen_tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _clean(text: str) -> str:
    text = re.sub(r"^(assistant:|Assistant:)\s*", "", str(text).strip()).strip()
    return text.strip('"').strip("'").strip()


@torch.inference_mode()
def _generate(models, config, source: str, *, do_sample: bool, seed: int, temperature=None):
    prompt = build_prompt(models, source)
    enc = models.gen_tok(prompt, return_tensors="pt", truncation=True, padding=True)
    enc = {k: v.to(models.device) for k, v in enc.items()}
    torch.manual_seed(int(seed))
    kwargs = {
        "max_new_tokens": int(config.max_new_tokens),
        "do_sample": bool(do_sample),
        "eos_token_id": models.gen_tok.eos_token_id,
        "pad_token_id": models.gen_tok.pad_token_id,
    }
    if do_sample:
        kwargs["temperature"] = float(temperature)
        kwargs["top_p"] = float(config.top_p)
    out = models.gen.generate(**enc, **kwargs)
    prompt_len = enc["input_ids"].shape[1]
    return _clean(models.gen_tok.decode(out[0, prompt_len:], skip_special_tokens=True))


def generate_candidates(models, config, source: str, source_seed: int) -> list[dict]:
    if int(config.K_gen) != 1 + len(config.temperatures):
        raise ValueError("K_gen must match the disclosed generation schedule.")

    rows = [{
        "text": _generate(models, config, source, do_sample=False, seed=source_seed),
        "generation_mode": "deterministic",
        "temperature": None,
        "generation_seed": int(source_seed),
    }]
    for offset, temperature in enumerate(config.temperatures, start=1):
        seed = int(source_seed + offset)
        rows.append({
            "text": _generate(
                models, config, source, do_sample=True, seed=seed, temperature=temperature
            ),
            "generation_mode": "stochastic",
            "temperature": float(temperature),
            "generation_seed": seed,
        })

    kept, seen = [], set()
    for row in rows:
        text = row["text"].strip()
        key = re.sub(r"\s+", " ", text.lower()).strip()
        if not text or key in seen:
            continue
        if not structural_quality(models, config, source, text):
            continue
        seen.add(key)
        kept.append(row)
    return kept
