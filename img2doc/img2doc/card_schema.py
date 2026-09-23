from __future__ import annotations

import json
import re
from typing import Any, Dict


class CardValidationError(ValueError):
    pass


CYBERWARE_LEVELS = {
    "C0": "原生级",
    "C1": "辅助级",
    "C2": "增强级",
    "C3": "战术级",
    "C4": "深度义体化",
    "C5": "全身义体",
}
GENDERS = {"男", "女", "未知"}


def _strip_json_fence(content: str) -> str:
    value = content.strip()
    if value.startswith("```") and value.endswith("```"):
        lines = value.splitlines()
        if len(lines) >= 3:
            value = "\n".join(lines[1:-1]).strip()
    return value


def parse_and_validate_card(content: str) -> Dict[str, Any]:
    try:
        raw = json.loads(_strip_json_fence(content))
    except (json.JSONDecodeError, TypeError) as exc:
        raise CardValidationError(f"模型返回内容不是有效 JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise CardValidationError("模型返回 JSON 顶层必须是对象")

    required = (
        "character_name",
        "codename",
        "character_gender",
        "role",
        "cyberware_level",
        "introduction",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise CardValidationError(f"模型返回缺少字段: {', '.join(missing)}")

    cleaned: Dict[str, str] = {}
    for key in ("character_name", "codename", "role", "introduction"):
        value = raw[key]
        if not isinstance(value, str) or not value.strip():
            raise CardValidationError(f"字段 {key} 必须是非空字符串")
        cleaned[key] = value.strip()

    if not re.fullmatch(r"[\u3400-\u9fff]{2,8}", cleaned["character_name"]):
        raise CardValidationError("character_name 必须是 2 至 8 个中文字符")
    if not re.fullmatch(r"[0-9]{3,8}", cleaned["codename"]):
        raise CardValidationError("codename 必须是 3 至 8 位纯数字编号字符串")
    if len(cleaned["role"]) > 16:
        raise CardValidationError("role 不能超过 16 个字符")
    if not 50 <= len(cleaned["introduction"]) <= 90:
        raise CardValidationError("introduction 必须为 50 至 90 个字符")

    gender = raw["character_gender"]
    if gender not in GENDERS:
        raise CardValidationError("character_gender 只能是男、女或未知")
    level = raw["cyberware_level"]
    if not isinstance(level, dict):
        raise CardValidationError("cyberware_level 必须是对象")
    code = str(level.get("code", "")).strip()
    name = str(level.get("name", "")).strip()
    if code not in CYBERWARE_LEVELS or CYBERWARE_LEVELS[code] != name:
        raise CardValidationError("义体等级 code 与 name 不匹配")
    return {
        "character_name": cleaned["character_name"],
        "codename": cleaned["codename"],
        "character_gender": gender,
        "role": cleaned["role"],
        "cyberware_level": {"code": code, "name": name},
        "introduction": cleaned["introduction"],
    }


def serialize_result(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
