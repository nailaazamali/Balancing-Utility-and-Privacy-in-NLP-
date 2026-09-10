from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import spacy
import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


def _entailment_index(model) -> int:
    labels = {int(k): str(v) for k, v in model.config.id2label.items()}
    matches = [idx for idx, label in labels.items() if "entail" in label.lower()]
    if len(matches) != 1:
        raise RuntimeError(f"Could not identify entailment label from {labels}")
    return matches[0]


@dataclass
class ModelBundle:
    config: object
    device: str
    dtype: torch.dtype
    clf_tok: object
    clf: object
    sbert: object
    gen_tok: object
    gen: object
    sens_tok: object
    sens_model: object
    sens_entail_idx: int
    nli_tok: object
    nli_model: object
    nli_entail_idx: int
    mlm_tok: object
    mlm: object
    nlp: object

    @classmethod
    def load(cls, config, device: str | None = None) -> "ModelBundle":
        device = device or getattr(config, "device", "cpu")
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be either 'cpu' or 'cuda'")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        dtype = torch.float16 if device == "cuda" else torch.float32
        common = {"torch_dtype": dtype, "low_cpu_mem_usage": True}

        clf_tok = AutoTokenizer.from_pretrained(config.classifier, use_fast=True)
        clf = AutoModelForSequenceClassification.from_pretrained(
            config.classifier, **common
        ).to(device).eval()

        sbert = SentenceTransformer(config.sbert, device=device)

        gen_tok = AutoTokenizer.from_pretrained(config.generator, use_fast=True)
        if gen_tok.pad_token is None:
            gen_tok.pad_token = gen_tok.eos_token
        gen = AutoModelForCausalLM.from_pretrained(
            config.generator, **common
        ).to(device).eval()

        sens_tok = AutoTokenizer.from_pretrained(
            config.sensitivity_context_model, use_fast=True
        )
        sens_model = AutoModelForSequenceClassification.from_pretrained(
            config.sensitivity_context_model, **common
        ).to(device).eval()

        nli_tok = AutoTokenizer.from_pretrained(config.nli_model, use_fast=True)
        nli_model = AutoModelForSequenceClassification.from_pretrained(
            config.nli_model, **common
        ).to(device).eval()

        mlm_tok = AutoTokenizer.from_pretrained(config.mlm_model, use_fast=True)
        mlm = AutoModelForMaskedLM.from_pretrained(
            config.mlm_model, **common
        ).to(device).eval()
        if mlm_tok.mask_token is None:
            raise RuntimeError("The selected masked-language model has no mask token.")

        nlp = spacy.load(config.spacy_model)

        return cls(
            config=config,
            device=device,
            dtype=dtype,
            clf_tok=clf_tok,
            clf=clf,
            sbert=sbert,
            gen_tok=gen_tok,
            gen=gen,
            sens_tok=sens_tok,
            sens_model=sens_model,
            sens_entail_idx=_entailment_index(sens_model),
            nli_tok=nli_tok,
            nli_model=nli_model,
            nli_entail_idx=_entailment_index(nli_model),
            mlm_tok=mlm_tok,
            mlm=mlm,
            nlp=nlp,
        )

    @torch.inference_mode()
    def toxic_probabilities(self, texts: list[str], batch_size: int = 16):
        p_non_all, p_tox_all = [], []
        for start in range(0, len(texts), batch_size):
            batch = [str(x) for x in texts[start:start + batch_size]]
            enc = self.clf_tok(
                batch, return_tensors="pt", truncation=True, padding=True
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            probs = torch.sigmoid(self.clf(**enc).logits.float()).cpu().numpy()
            p_tox = probs.max(axis=1)
            p_tox_all.append(p_tox.astype(np.float32))
            p_non_all.append((1.0 - p_tox).astype(np.float32))
        return np.concatenate(p_non_all), np.concatenate(p_tox_all)

    def sentence_embeddings(self, texts: list[str]):
        return self.sbert.encode(
            [str(x) for x in texts],
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    @torch.inference_mode()
    def nli_entailment(self, source: str, claims: list[str], batch_size: int = 12):
        values = []
        for start in range(0, len(claims), batch_size):
            batch = claims[start:start + batch_size]
            enc = self.nli_tok(
                [str(source)] * len(batch),
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            probs = torch.softmax(self.nli_model(**enc).logits.float(), dim=-1)
            values.extend(probs[:, self.nli_entail_idx].cpu().tolist())
        return [float(x) for x in values]

    @torch.inference_mode()
    def contextual_sensitivity(self, premises: list[str], hypotheses: list[str]):
        enc = self.sens_tok(
            premises,
            hypotheses,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.sens_model(**enc).logits.float()
        return logits[:, self.sens_entail_idx].cpu().numpy()
