#!/usr/bin/env python3
"""
update_models_2026.py
Хирургическое обновление названий моделей в KeryxHunter.
Только имена моделей. Ничего больше не меняется.
"""

from pathlib import Path
import re

# ==================== ТОЧНЫЙ МАППИНГ 2026 ====================
MODEL_MAP = {
    # Cloud (remote.py)
    "claude-opus-4.6":          "claude-opus-4.6",
    "claude-sonnet-4.6":        "claude-sonnet-4.6",
    "claude-haiku-4.6":         "claude-haiku-4.6",
    "gpt-5.4":                   "gpt-5.4",
    "gpt-5.4-mini":              "gpt-5.4-mini",
    "gpt-5-turbo":              "gpt-5-turbo",
    "llama-4-70b-versatile":  "llama-4-70b-versatile",
    "deepseek-v4":            "deepseek-v4",
    "deepseek-r1":           "deepseek-r1",

    # Local + Swarm
    "llama-4-70b":                "llama-4-70b",
    "qwen3.5-coder-32b":                 "qwen3.5-coder-32b",
    "llama-4-13b":                "llama-4-13b",
    "qwen3-coder-8b":                 "qwen3-coder-8b",
    "phi-4-mini":               "phi-4-mini",
    "gemma-3-2b":                 "gemma-3-2b",
}

def update_file(file_path: Path) -> bool:
    original = file_path.read_text(encoding="utf-8")
    updated = original

    changed = 0
    for old, new in MODEL_MAP.items():
        # Только целое слово (word boundary)
        pattern = r'\b' + re.escape(old) + r'\b'
        new_text, count = re.subn(pattern, new, updated)
        if count > 0:
            updated = new_text
            changed += count

    if changed == 0:
        return False

    # Бэкап
    backup = file_path.with_suffix(".py.bak")
    backup.write_text(original, encoding="utf-8")

    # Запись
    file_path.write_text(updated, encoding="utf-8")

    print(f"✅ Обновлён: {file_path.name}  (+{changed} замен)")
    return True


if __name__ == "__main__":
    project_root = Path("/Users/serhiihrynko/Documents/Helga/Keryx_Hunter")
    keryx_dir = project_root / "keryx"

    print("🔧 Хирургическое обновление моделей на 2026...\n")

    updated_count = 0
    for py_file in keryx_dir.rglob("*.py"):
        if py_file.name.endswith(".bak"):
            continue
        if update_file(py_file):
            updated_count += 1

    print("\n" + "="*70)
    print(f"Готово! Обновлено файлов: {updated_count}")
    print("Все бэкапы сохранены как *.bak")
    print("Только названия моделей были изменены.")
    print("="*70)