#!/usr/bin/env python3
"""
clean_invisible.py
Удаляет только невидимые Unicode-символы из всех .py файлов.
Не меняет логику, отступы, кавычки и структуру кода.
Создаёт .bak бэкапы перед изменением.
"""

import re
from pathlib import Path

# Невидимые и проблемные Unicode-символы, которые часто ломают подсветку
INVISIBLE_CHARS = [
    '\u200B',  # Zero Width Space
    '\u200C',  # Zero Width Non-Joiner
    '\u200D',  # Zero Width Joiner
    '\u200E',  # Left-to-Right Mark
    '\u200F',  # Right-to-Left Mark
    '\u00A0',  # Non-breaking space
    '\uFEFF',  # Byte Order Mark
    '\u2028',  # Line Separator
    '\u2029',  # Paragraph Separator
]

# Регулярное выражение для удаления всех этих символов
INVISIBLE_PATTERN = re.compile('[' + ''.join(INVISIBLE_CHARS) + ']')


def clean_file(file_path: Path):
    original_text = file_path.read_text(encoding='utf-8')

    cleaned_text = INVISIBLE_PATTERN.sub('', original_text)

    if cleaned_text == original_text:
        print(f"✓ Уже чистый: {file_path.name}")
        return 0

    # Создаём бэкап
    backup_path = file_path.with_suffix('.py.bak')
    backup_path.write_text(original_text, encoding='utf-8')

    # Перезаписываем чистой версией
    file_path.write_text(cleaned_text, encoding='utf-8')

    removed_count = len(original_text) - len(cleaned_text)
    print(f"✅ Очищен: {file_path.name}  (удалено {removed_count} невидимых символов)")
    return removed_count


if __name__ == "__main__":
    project_root = Path("/Users/serhiihrynko/Documents/Helga/Keryx_Hunter")
    target_dir = project_root / "keryx"

    print("🔧 Запуск очистки невидимых Unicode-символов...\n")

    total_removed = 0
    files_processed = 0

    for py_file in target_dir.rglob("*.py"):
        if py_file.name.endswith(".bak"):
            continue
        removed = clean_file(py_file)
        total_removed += removed
        files_processed += 1

    print("\n" + "="*70)
    print(f"Завершено! Обработано файлов: {files_processed}")
    print(f"Всего удалено невидимых символов: {total_removed}")
    print("Бэкапы сохранены как *.bak")
    print("="*70)
