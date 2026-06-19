from slop_sftdiv.generation_text import final_answer_text, feature_text_for_mode


def test_final_answer_text_strips_xml_reasoning_blocks():
    text = "<think>delve into hidden analysis</think>\n\nFinal answer: plain result."

    assert final_answer_text(text) == "Final answer: plain result."


def test_final_answer_text_strips_fenced_reasoning_blocks():
    text = "```thinking\nCertainly, hidden chain\n```\nVisible answer."

    assert final_answer_text(text) == "Visible answer."


def test_final_answer_text_drops_unclosed_reasoning_tail():
    text = "Visible prefix.\n<think>unfinished hidden delve"

    assert final_answer_text(text) == "Visible prefix."


def test_feature_text_for_mode_raw_preserves_text():
    text = "<think>hidden</think> answer"

    assert feature_text_for_mode(text, "raw") == text
