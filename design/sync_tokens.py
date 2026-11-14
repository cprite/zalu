"""Single source of truth for ZALU design tokens: frontend/src/app/globals.css (@theme block).

    python design/sync_tokens.py          # writes design/tokens.json and updates the TOKENS block in the Figma plugin
    python design/sync_tokens.py --check  # exit 1 if either file is out of date (for CI)

tokens.json follows the W3C Design Tokens (DTCG) format, importable with Tokens Studio or Figma variable importers.
It also reports WCAG contrast for every *-ink / text token so accessibility regressions are visible.
"""
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSS = ROOT / "frontend" / "src" / "app" / "globals.css"
PLUGIN = ROOT / "design" / "figma-plugin" / "code.js"
OUT = ROOT / "design" / "tokens.json"


def read(p):
    return io.open(p, encoding="utf-8").read()


def parse():
    theme = re.search(r"@theme\s*{(.*?)\n}", read(CSS), re.S).group(1)
    colors = dict(re.findall(r"--color-([a-z0-9-]+):\s*(#[0-9A-Fa-f]{6})", theme))
    radius = {k: int(v) for k, v in re.findall(r"--radius-([a-z]+):\s*(\d+)px", theme)}
    return {k: v.upper() for k, v in colors.items()}, radius


def luminance(h):
    c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def contrast(a, b):
    la, lb = sorted([luminance(a), luminance(b)], reverse=True)
    return round((la + 0.05) / (lb + 0.05), 2)


def build():
    colors, radius = parse()
    doc = {
        "$description": "ZALU design tokens. Generated from frontend/src/app/globals.css by design/sync_tokens.py. Do not edit.",
        "color": {k: {"$type": "color", "$value": v, "$extensions": {"web": f"var(--color-{k})"}} for k, v in colors.items()},
        "spacing": {str(n): {"$type": "dimension", "$value": f"{n}px"} for n in (4, 8, 12, 16, 24, 32, 48)},
        "radius": {k: {"$type": "dimension", "$value": f"{v}px", "$extensions": {"web": f"var(--radius-{k})"}} for k, v in radius.items()},
        "fontFamily": {"sans": {"$type": "fontFamily", "$value": ["Inter", "system-ui", "sans-serif"]}},
    }
    checks = {}
    for k, v in colors.items():
        if k in ("ink", "ink-2", "muted") or k.endswith("-ink"):
            bg = colors.get(k.replace("-ink", "-soft"), colors["surface"]) if k.endswith("-ink") else colors["bg"]
            checks[k] = {"on": bg, "ratio": contrast(v, bg), "AA_text": contrast(v, bg) >= 4.5}
    checks["control"] = {"on": colors["surface"], "ratio": contrast(colors["control"], colors["surface"]), "AA_ui": contrast(colors["control"], colors["surface"]) >= 3}
    doc["$extensions"] = {"wcag": checks}
    js = "const TOKENS = " + json.dumps({"color": colors, "radius": radius}, indent=2) + ";"
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n", js, checks


def main():
    tokens_json, js, checks = build()
    plugin = read(PLUGIN)
    new_plugin = re.sub(r"(// <TOKENS>[^\n]*\n).*?(\n// </TOKENS>)", lambda m: m.group(1) + js + m.group(2), plugin, flags=re.S)
    stale = (not OUT.exists() or read(OUT) != tokens_json) or new_plugin != plugin
    fails = [k for k, c in checks.items() if not (c.get("AA_text", True) and c.get("AA_ui", True))]
    if "--check" in sys.argv:
        if stale or fails:
            sys.exit(f"tokens out of date: {stale}; contrast failures: {fails}")
        print("tokens up to date, contrast OK")
        return
    io.open(OUT, "w", encoding="utf-8", newline="\n").write(tokens_json)
    io.open(PLUGIN, "w", encoding="utf-8", newline="\n").write(new_plugin)
    print(f"wrote {OUT.relative_to(ROOT)} and updated {PLUGIN.relative_to(ROOT)}")
    for k, c in checks.items():
        print(f"  {k:12} {c['ratio']:>5}:1 on {c['on']}  {'OK' if k not in fails else 'FAIL'}")


if __name__ == "__main__":
    main()
