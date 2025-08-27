# [auto-fix] Insert strict-interleaving stubs, rename unused loop var, and migrate Ruff config.
# It is safe to run multiple times; original files are backed up once as *.bak.

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

TS = datetime.now(timezone(timedelta(hours=2))).isoformat(timespec="minutes")  # Europe/Berlin

ROOT = Path(__file__).resolve().parents[1]
VIS = ROOT / "src" / "garden_planner" / "visualizer.py"
TOML = ROOT / "pyproject.toml"

STUBS = f"""
# [{TS}] @KernelKale
# Temporary stubs for strict interleaving. Replace with real logic later.

def _enforce_interleaving_strict(clusters, gap=0.01, max_passes=1):
    \"\"\"No-op: return clusters unchanged until proper logic is implemented.\"\"\"
    return clusters


def _validate_interleaving_strict(clusters) -> int:
    \"\"\"No-op: return 0 = no violations.\"\"\"
    return 0
""".lstrip()


def patch_visualizer():
    text = VIS.read_text(encoding="utf-8")
    original = text

    # 1) Insert stubs above the CLI header (or above def main(...) as fallback).
    if "_enforce_interleaving_strict" not in text and "_validate_interleaving_strict" not in text:
        cli_header_pat = re.compile(r"^\s*#\s*-+\s*CLI\s*-+\s*$", re.MULTILINE)
        main_pat = re.compile(r"^def\s+main\s*\(", re.MULTILINE)

        m = cli_header_pat.search(text) or main_pat.search(text)
        if not m:
            raise SystemExit(
                "Could not find CLI header or def main(. Please insert stubs manually."
            )
        insert_at = m.start()
        text = text[:insert_at] + STUBS + "\n" + text[insert_at:]

    # 2) Rename unused loop variable x -> _x for the zip(base, pts, strict=False) loop.
    # Make it robust to whitespace.
    loop_pat = re.compile(
        r"(for\s*\(\s*Xp\s*,\s*Yp\s*\)\s*,\s*\(\s*)x(\s*,\s*y\s*,\s*c\s*\)\s*in\s*zip\(\s*base\s*,\s*pts\s*,\s*strict=False\s*\)\s*:)",
        re.MULTILINE,
    )
    text, n = loop_pat.subn(r"\1_x\2  # x unused", text)

    if text != original:
        if not VIS.with_suffix(".py.bak").exists():
            VIS.with_suffix(".py.bak").write_text(original, encoding="utf-8")
        VIS.write_text(text, encoding="utf-8")
        print(
            f"[OK] visualizer.py updated ({'stubs added, ' if '_enforce_interleaving_strict' in text else ''}{n} loop rename(s))."
        )
    else:
        print("[SKIP] visualizer.py already in desired state.")


def patch_toml():
    t = TOML.read_text(encoding="utf-8")
    original = t

    # Ensure [tool.ruff] exists
    if "[tool.ruff]" not in t:
        raise SystemExit("pyproject.toml missing [tool.ruff] section.")

    # Capture select/ignore under [tool.ruff], then remove them there.
    # Later we will create [tool.ruff.lint] with those values (or defaults).
    select_pat = re.compile(r"(?m)^(select\s*=\s*\[.*?\])\s*$")
    ignore_pat = re.compile(r"(?m)^(ignore\s*=\s*\[.*?\])\s*$")

    # Work only inside the [tool.ruff] block
    ruff_block_pat = re.compile(r"(?ms)^\[tool\.ruff\]\s*(.+?)(?=^\[|\Z)")
    m = ruff_block_pat.search(t)
    if not m:
        raise SystemExit("Could not parse [tool.ruff] block.")
    ruff_block = m.group(1)

    found_select = select_pat.search(ruff_block)
    found_ignore = ignore_pat.search(ruff_block)

    new_ruff_block = select_pat.sub("", ruff_block)
    new_ruff_block = ignore_pat.sub("", new_ruff_block)

    # Reassemble with cleaned [tool.ruff]
    t = t[: m.start(1)] + new_ruff_block + t[m.end(1) :]

    # If [tool.ruff.lint] already exists, do nothing; else insert with values.
    if "[tool.ruff.lint]" not in t:
        select_val = (
            found_select.group(1).split("=", 1)[1].strip()
            if found_select
            else '["E","F","I","UP","B"]'
        )
        ignore_val = found_ignore.group(1).split("=", 1)[1].strip() if found_ignore else '["E501"]'
        lint_block = f"""
[tool.ruff.lint]
select = {select_val}
ignore = {ignore_val}
""".lstrip()
        # Insert right after [tool.ruff] block
        t = t[: m.end()] + "\n" + lint_block + t[m.end() :]

    if t != original:
        if not TOML.with_suffix(".toml.bak").exists():
            TOML.with_suffix(".toml.bak").write_text(original, encoding="utf-8")
        TOML.write_text(t, encoding="utf-8")
        print("[OK] pyproject.toml updated (moved select/ignore to [tool.ruff.lint]).")
    else:
        print("[SKIP] pyproject.toml already in desired state.")


def main():
    if not VIS.exists():
        raise SystemExit(f"Not found: {VIS}")
    if not TOML.exists():
        raise SystemExit(f"Not found: {TOML}")
    patch_visualizer()
    patch_toml()
    print("[DONE] Apply fixes complete.")


if __name__ == "__main__":
    main()
