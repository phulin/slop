from slop_sftdiv.minhash import estimate_jaccard, minhash_signature, token_shingles


def test_token_shingles_normalizes_case_and_whitespace():
    assert token_shingles("  Explain   Robust TESTING now ", shingle_size=2) == frozenset(
        {"explain robust", "robust testing", "testing now"}
    )


def test_minhash_similarity_tracks_overlap():
    left = minhash_signature("explain robust testing with concrete examples", num_hashes=64, shingle_size=1)
    near = minhash_signature("explain robust testing with examples", num_hashes=64, shingle_size=1)
    far = minhash_signature("write a short poem about winter", num_hashes=64, shingle_size=1)

    assert estimate_jaccard(left, near) > 0.5
    assert estimate_jaccard(left, far) < 0.3
