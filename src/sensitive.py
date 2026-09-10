from __future__ import annotations

import re

import numpy as np


REGEX_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "phone": r"(?<!\w)(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}(?!\w)",
    "url": r"(?:https?://|www\.)[^\s]+",
    "handle": r"(?<!\w)@\w+",
    "hashtag": r"(?<!\w)#\w+",
}

DIRECT_TYPES = {"email", "phone", "url", "handle"}

SENSITIVE_TERMS = {
    "religion": [
        "muslim", "muslims", "christian", "christians", "jew", "jews",
        "jewish", "hindu", "hindus", "sikh", "sikhs",
    ],
    "race_ethnicity_nationality_immigration": [
        "black", "white", "asian", "arab", "latino", "mexican",
        "immigrant", "immigrants", "refugee", "refugees",
    ],
    "gender_sexuality": [
        "woman", "women", "man", "men", "female", "male", "gay",
        "lesbian", "trans", "transgender", "queer",
    ],
    "disability_health": [
        "disabled", "disability", "autism", "deaf", "blind", "cancer",
        "chemotherapy", "depression", "pregnant", "pregnancy", "hospital", "clinic",
    ],
    "location_context": [
        "london", "glasgow", "quetta", "manchester", "school",
        "university", "workplace", "office",
    ],
    "family_context": [
        "wife", "husband", "child", "daughter", "son", "mother", "father", "family",
    ],
    "legal_context": ["police", "court", "arrest", "complaint", "case"],
}

PRIVACY_WEIGHTS = {
    "email": 1.00,
    "phone": 1.00,
    "possible_person_name": 0.85,
    "disability_health": 0.85,
    "location_context": 0.70,
    "family_context": 0.70,
    "legal_context": 0.70,
    "handle": 0.60,
    "religion": 0.55,
    "race_ethnicity_nationality_immigration": 0.55,
    "gender_sexuality": 0.55,
    "url": 0.40,
    "organisation_context": 0.30,
    "hashtag": 0.25,
}

CONTEXT_LABELS = {
    "religion": "a person's religion or religious belief",
    "race_ethnicity_nationality_immigration":
        "a person's race, ethnicity, nationality or immigration context",
    "gender_sexuality": "a person's gender identity or sexual orientation",
    "disability_health": "a person's health condition, disability, medical status or pregnancy",
    "location_context": "information revealing a person's private or identifying location",
    "family_context": "information revealing a person's family or personal relationship",
    "legal_context": "information revealing a person's legal situation or involvement",
    "possible_person_name": "information identifying a specific person",
    "organisation_context": "information revealing a person's organisation, institution or workplace",
    "hashtag": "a hashtag revealing privacy-sensitive personal information",
}


def _key(item: dict) -> tuple[str, str]:
    return item["category"].strip().lower(), item["span"].strip().lower()


def _raw_candidates(models, text: str) -> list[dict]:
    text = str(text)
    found = []
    for category, pattern in REGEX_PATTERNS.items():
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            found.append({
                "category": category,
                "span": match.group(0).lower().strip(),
                "start": match.start(),
                "end": match.end(),
                "source": "regex",
            })

    for category, terms in SENSITIVE_TERMS.items():
        for term in terms:
            for match in re.finditer(r"\b" + re.escape(term) + r"\b", text, flags=re.IGNORECASE):
                found.append({
                    "category": category,
                    "span": match.group(0).lower(),
                    "start": match.start(),
                    "end": match.end(),
                    "source": "lexicon",
                })

    doc = models.nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            category = "possible_person_name"
        elif ent.label_ in {"GPE", "LOC", "FAC"}:
            category = "location_context"
        elif ent.label_ == "ORG":
            category = "organisation_context"
        else:
            continue
        found.append({
            "category": category,
            "span": ent.text.lower().strip(),
            "start": ent.start_char,
            "end": ent.end_char,
            "source": "ner",
        })

    unique = {}
    for item in found:
        unique[(item["category"], item["span"], item["start"], item["end"])] = item
    return list(unique.values())


def detect_sensitive_items(models, config, text: str) -> list[dict]:
    candidates = _raw_candidates(models, text)
    retained = []
    pending = []

    for item in candidates:
        if item["category"] in DIRECT_TYPES:
            kept = dict(item)
            kept["context_score"] = 1.0
            retained.append(kept)
        else:
            pending.append(item)

    if pending:
        premises, hypotheses = [], []
        for item in pending:
            label = CONTEXT_LABELS.get(item["category"], "privacy-sensitive personal information")
            context = (
                f'Text: "{text}"\nCandidate span: "{item["span"]}".\n'
                f'The candidate was initially detected as "{item["category"]}".\n'
                "Determine whether this candidate reveals privacy-sensitive personal information "
                "about a person in this specific context."
            )
            premises.extend([context, context])
            hypotheses.extend([
                f"This text contains {label}.",
                "This text contains not privacy-sensitive personal information.",
            ])

        logits = models.contextual_sensitivity(premises, hypotheses).reshape(-1, 2)
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / probs.sum(axis=1, keepdims=True)

        for item, row in zip(pending, probs):
            sensitive_score, nonsensitive_score = float(row[0]), float(row[1])
            if sensitive_score >= float(config.tau_sens) and sensitive_score > nonsensitive_score:
                kept = dict(item)
                kept["context_score"] = sensitive_score
                retained.append(kept)

    return retained


def normalized_key(item: dict) -> tuple[str, str]:
    return _key(item)
