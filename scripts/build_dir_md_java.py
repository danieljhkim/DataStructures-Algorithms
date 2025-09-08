#!/usr/bin/env python3
import os
import re
from collections.abc import Iterator

EXCLUDES_DIR = {"scripts", "target", "build", "out"}
SKIP_CONTAINS = ("venv",)


def good_file_paths(top_dir: str = ".", exts=(".java",)) -> Iterator[str]:
    for dir_path, dir_names, filenames in os.walk(top_dir):
        dir_names[:] = [
            d
            for d in dir_names
            if d not in EXCLUDES_DIR
            and d[0] not in "._"
            and not any(s in d for s in SKIP_CONTAINS)
        ]
        for filename in filenames:
            if filename.startswith("."):
                continue
            if os.path.splitext(filename)[1] in exts:
                yield os.path.join(dir_path, filename).lstrip("./")


def md_prefix(indent: int) -> str:
    return f"{indent * '  '}*" if indent else "\n##"


def print_path(old_path: str, new_path: str) -> str:
    old_parts = old_path.split(os.sep)
    for i, new_part in enumerate(new_path.split(os.sep)):
        if (i + 1 > len(old_parts) or old_parts[i] != new_part) and new_part:
            print(f"{md_prefix(i)} {new_part.replace('_', ' ').title()}")
    return new_path


def format_display(name: str) -> str:
    stem, _ext = os.path.splitext(name)
    display = stem.replace("_", " ")
    display = re.sub(r"(?<!^)(?=[A-Z])", " ", display)
    parts = [w.capitalize() for w in display.split()]
    return " ".join(parts)


def print_java_directory_md(top_dir: str = ".") -> None:
    old_path = ""
    for fullpath in sorted(good_file_paths(top_dir, exts=(".java",))):
        dirpath, filename = os.path.split(fullpath)
        if dirpath != old_path:
            old_path = print_path(old_path, dirpath)
        indent = (dirpath.count(os.sep) + 1) if dirpath else 0
        url = f"{dirpath}/{filename}".replace(" ", "%20") if dirpath else filename
        display = format_display(filename)
        print(f"{md_prefix(indent)} [{display}]({url})")


if __name__ == "__main__":
    print_java_directory_md(".")
