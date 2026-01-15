#!/usr/bin/env python3
"""
Export SkyClip source code to a markdown file.
Only actual code - no configs, no JSON, no lock files.
"""

from pathlib import Path
from datetime import datetime

# Only actual source code
INCLUDE_EXTENSIONS = {
    '.rs',      # Rust
    '.tsx',     # React TypeScript
    '.ts',      # TypeScript
    '.py',      # Python
    '.css',     # Styles
}

# Directories to skip
SKIP_DIRS = {
    'node_modules',
    'target',
    'dist',
    '.git',
    '.vscode',
    '__pycache__',
    '.pytest_cache',
    'build',
}

# Files to skip
SKIP_FILES = {
    'export_code.py',
    'vite-env.d.ts',
}

def should_include(file_path: Path, root: Path) -> bool:
    name = file_path.name
    rel_path = file_path.relative_to(root)

    if name in SKIP_FILES:
        return False

    for part in rel_path.parts:
        if part in SKIP_DIRS:
            return False

    if file_path.suffix not in INCLUDE_EXTENSIONS:
        return False

    return True

def get_language(file_path: Path) -> str:
    ext_map = {
        '.rs': 'rust',
        '.tsx': 'tsx',
        '.ts': 'typescript',
        '.py': 'python',
        '.css': 'css',
    }
    return ext_map.get(file_path.suffix, '')

def export_to_markdown(root_dir: Path, output_file: Path):
    files_content = []

    for file_path in sorted(root_dir.rglob('*')):
        if not file_path.is_file():
            continue
        if not should_include(file_path, root_dir):
            continue

        try:
            content = file_path.read_text(encoding='utf-8')
            rel_path = file_path.relative_to(root_dir)
            files_content.append((str(rel_path), content, get_language(file_path)))
        except Exception as e:
            print(f"Skipping {file_path}: {e}")

    lines = [
        "# SkyClip Source Code",
        "",
    ]

    for rel_path, content, lang in files_content:
        lines.append(f"## {rel_path}")
        lines.append("")
        lines.append(f"```{lang}")
        lines.append(content.rstrip())
        lines.append("```")
        lines.append("")

    output_file.write_text('\n'.join(lines), encoding='utf-8')
    print(f"Exported {len(files_content)} files to {output_file}")

if __name__ == '__main__':
    root = Path(__file__).parent
    output = root / 'skyclip_code_export.md'
    export_to_markdown(root, output)
