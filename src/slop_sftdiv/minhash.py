from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable


TOKEN_RE = re.compile(r"\b[\w']+\b", re.UNICODE)
MAX_U64 = (1 << 64) - 1


def normalize_for_minhash(text: str) -> str:
    return " ".join(text.casefold().split())


def token_shingles(text: str, *, shingle_size: int = 5) -> frozenset[str]:
    normalized = normalize_for_minhash(text)
    tokens = TOKEN_RE.findall(normalized)
    if not tokens:
        return frozenset()
    if len(tokens) <= shingle_size:
        return frozenset({" ".join(tokens)})
    return frozenset(
        " ".join(tokens[index : index + shingle_size])
        for index in range(len(tokens) - shingle_size + 1)
    )


def minhash_signature(
    text: str,
    *,
    num_hashes: int = 64,
    shingle_size: int = 5,
) -> tuple[int, ...]:
    if num_hashes <= 0:
        raise ValueError("num_hashes must be positive")
    if shingle_size <= 0:
        raise ValueError("shingle_size must be positive")
    shingles = token_shingles(text, shingle_size=shingle_size)
    if not shingles:
        return tuple()
    signature: list[int] = []
    for seed in range(num_hashes):
        signature.append(min(_stable_hash(shingle, seed=seed) for shingle in shingles))
    return tuple(signature)


def estimate_jaccard(signature_a: Iterable[int], signature_b: Iterable[int]) -> float:
    left = tuple(signature_a)
    right = tuple(signature_b)
    if not left or not right:
        return 0.0
    width = min(len(left), len(right))
    matches = sum(1 for index in range(width) if left[index] == right[index])
    return matches / width


def signature_bands(signature: tuple[int, ...], *, band_size: int = 4) -> list[tuple[int, tuple[int, ...]]]:
    if band_size <= 0:
        raise ValueError("band_size must be positive")
    return [
        (index // band_size, signature[index : index + band_size])
        for index in range(0, len(signature), band_size)
        if len(signature[index : index + band_size]) == band_size
    ]


def _stable_hash(value: str, *, seed: int) -> int:
    digest = hashlib.blake2b(f"{seed}\0{value}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & MAX_U64
