from __future__ import annotations

import pytest

from slop_sftdiv.cli.pangram_detector_sample_search import sample_objective


def test_sample_objective_combines_detector_lm_and_quality_terms() -> None:
    objective = sample_objective(
        editlens_score=0.8,
        mean_lm_log_probability=-2.0,
        sequence_quality_penalty_value=1.5,
        lm_logprob_coeff=0.1,
        sequence_quality_penalty_coeff=0.2,
    )

    assert objective == pytest.approx(0.8 - 0.2 - 0.3)
