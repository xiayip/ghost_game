import json

import pytest

from img2doc.card_schema import CardValidationError, parse_and_validate_card, serialize_result


def valid_card():
    return {
        "character_name": "镜面幽灵",
        "codename": "007",
        "character_gender": "未知",
        "role": "网络侦察员",
        "cyberware_level": {"code": "C2", "name": "增强级"},
        "introduction": "游走于城市网络边缘的侦察员，借助感知增强接口追踪异常信号，并为行动小组建立安全的数据通道，在复杂街区中持续记录潜在威胁。",
    }


def test_valid_card_round_trip():
    parsed = parse_and_validate_card(json.dumps(valid_card(), ensure_ascii=False))
    assert parsed["character_gender"] == "未知"
    assert json.loads(serialize_result(parsed)) == parsed


def test_rejects_mismatched_cyberware_level():
    card = valid_card()
    card["cyberware_level"] = {"code": "C2", "name": "战术级"}
    with pytest.raises(CardValidationError):
        parse_and_validate_card(json.dumps(card, ensure_ascii=False))


def test_accepts_json_code_fence_for_resilience():
    content = "```json\n" + json.dumps(valid_card(), ensure_ascii=False) + "\n```"
    assert parse_and_validate_card(content)["codename"] == "007"


def test_rejects_text_codename():
    card = valid_card()
    card["codename"] = "GHOST-07"
    with pytest.raises(CardValidationError):
        parse_and_validate_card(json.dumps(card, ensure_ascii=False))


def test_short_introduction_is_extended_for_the_web_card():
    card = valid_card()
    card["introduction"] = "负责追踪深网中的异常信号。"
    parsed = parse_and_validate_card(json.dumps(card, ensure_ascii=False))
    assert 50 <= len(parsed["introduction"]) <= 90
    assert parsed["introduction"].startswith("负责追踪深网中的异常信号。")


def test_long_introduction_is_trimmed_on_a_sentence_boundary():
    card = valid_card()
    first_sentence = "负责在复杂城区追踪深网异常信号并维护行动小组的安全通信链路，持续记录潜在威胁。"
    card["introduction"] = first_sentence + "额外的冗长描述。" * 10
    parsed = parse_and_validate_card(json.dumps(card, ensure_ascii=False))
    assert 50 <= len(parsed["introduction"]) <= 90
    assert parsed["introduction"].endswith("。")


def test_long_unbroken_introduction_uses_ellipsis():
    card = valid_card()
    card["introduction"] = "网" * 120
    parsed = parse_and_validate_card(json.dumps(card, ensure_ascii=False))
    assert len(parsed["introduction"]) == 90
    assert parsed["introduction"].endswith("…")
