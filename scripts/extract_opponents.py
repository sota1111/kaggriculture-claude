"""Reproducibly rebuild opponents/ from public Kaggle kernels (SOT-2383).

The opponent pool for the replay-league (scripts/eval_league.py) is a frozen
snapshot of the current *public top-N* kaggriculture kernels. This script pulls
each kernel and extracts its committed agent policy into opponents/<name>.py, then
writes opponents/MANIFEST.json with provenance (kernel ref, sha256, extraction
method). The committed opponents/*.py ARE the authoritative frozen artifact — this
script documents how they were produced and lets a maintainer refresh the pool.

Requires: kaggle CLI auth + network. Run from the repo root:
  python scripts/extract_opponents.py [--pull-dir /tmp/kpull]

Each public kernel stores its policy differently, hence per-kernel extractors:
  - prvsiyan frontier tapes : `AGENT_SOURCE = '<escaped main.py>'` cell literal
  - kaitofukami v17         : base85 + zlib compressed main.py payload
  - pilkwang economic ctrl  : `%%agentfile` cell magic (body is the policy module)
  - roman hamburger         : base64 + gzip `ANCHOR_BLOB` (their submitted anchor)
"""
import json, ast, re, base64, zlib, gzip, hashlib, sys, importlib.util, subprocess, argparse
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "opponents"

# ref -> (output filename, extraction method key)
KERNELS = {
    "prvsiyan/kaggriculture-frontier-the-moon-counts-melons": ("moon_counts_melons.py", "agent_source"),
    "prvsiyan/kaggriculture-frontier-the-soil-remembers-rain": ("soil_remembers_rain.py", "agent_source"),
    "kaitofukami/29-30-current-holdout-v17-learned-market-ranker": ("kaitofukami_v17_market_ranker.py", "b85_zlib"),
    "pilkwang/kaggriculture-observable-economic-control": ("pilkwang_economic_control.py", "agentfile"),
    "romantamrazov/kaggriculture-hamburger": ("roman_hamburger_anchor.py", "anchor_blob"),
}


def _cells(nb_path):
    return ["".join(c.get("source", [])) for c in json.load(open(nb_path))["cells"]
            if c.get("cell_type") == "code"]


def _extract_agent_source(cells):
    for src in cells:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "AGENT_SOURCE" for t in node.targets):
                return ast.literal_eval(node.value)
    raise ValueError("AGENT_SOURCE not found")


def _extract_b85_zlib(cells):
    for src in cells:
        if "b85decode" in src and "zlib.decompress" in src:
            m = re.search(r'payload\s*=\s*""\.join\(\[(.*?)\]\)', src, re.S)
            if m:
                payload = "".join(ast.literal_eval("[" + m.group(1) + "]"))
                return zlib.decompress(base64.b85decode(payload)).decode("utf-8")
    raise ValueError("b85+zlib payload not found")


def _extract_agentfile(cells):
    for src in cells:
        if src.lstrip().startswith("%%agentfile"):
            return src.split("\n", 1)[1]
    raise ValueError("%%agentfile cell not found")


def _extract_anchor_blob(cells):
    for src in cells:
        if "ANCHOR_BLOB" in src:
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == "ANCHOR_BLOB" for t in node.targets):
                    blob = ast.literal_eval(node.value)
                    return gzip.decompress(base64.b64decode(blob)).decode("utf-8")
    raise ValueError("ANCHOR_BLOB not found")


EXTRACTORS = {
    "agent_source": _extract_agent_source,
    "b85_zlib": _extract_b85_zlib,
    "agentfile": _extract_agentfile,
    "anchor_blob": _extract_anchor_blob,
}


def _loads_ok(path):
    uniq = "checkopp_" + hashlib.sha1(str(path).encode()).hexdigest()[:8]
    spec = importlib.util.spec_from_file_location(uniq, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[uniq] = mod
    spec.loader.exec_module(mod)
    return callable(getattr(mod, "agent", None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull-dir", default="/tmp/kpull")
    args = ap.parse_args()
    pull = Path(args.pull_dir)
    OUT.mkdir(exist_ok=True)
    manifest = []
    for ref, (fname, method) in KERNELS.items():
        d = pull / ref.replace("/", "_")
        d.mkdir(parents=True, exist_ok=True)
        if not list(d.glob("*.ipynb")):
            subprocess.run(["kaggle", "kernels", "pull", ref, "-p", str(d)], check=True)
        nb = list(d.glob("*.ipynb"))[0]
        src = EXTRACTORS[method](_cells(nb))
        (OUT / fname).write_text(src, encoding="utf-8")
        ok = _loads_ok(OUT / fname)
        sha = hashlib.sha256(src.encode()).hexdigest()
        print(f"[{fname}] loads={ok} lines={src.count(chr(10))+1} sha256={sha[:16]} <- {ref} ({method})")
        manifest.append({"file": fname, "kernel_ref": ref, "extraction": method,
                         "sha256": sha, "lines": src.count("\n") + 1, "loads": ok})
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {OUT/'MANIFEST.json'} ({len(manifest)} opponents)")


if __name__ == "__main__":
    main()
