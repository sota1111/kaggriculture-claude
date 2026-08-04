"""Held-out replay-league eval harness for kaggriculture (SOT-2383).

Oracle re-anchoring. The existing local oracle (scripts/eval.py) only measures a
candidate against the *committed champion* (self-mirror + vs-own-champion). That
field does not reflect the real public arena: the current champion wins 20/20 in
self-mirror yet sits out-of-rank on the public LB. This harness follows the
"time-split replay league" idea (kaitofukami): freeze the current *public top-N*
kernels' policies as an opponent pool (opponents/*.py) and score a candidate
against that real field, both seats, on a held-out screen/confirm seed split.

Each game loads a FRESH instance of both the candidate and the opponent module so
tape agents that keep module-global state cannot leak between games (determinism +
independence, matching the arena's fresh-process-per-episode model). Opponent
modules are registered in sys.modules under a unique name before exec so their
module-level dataclasses / helpers resolve correctly.

Usage:
  python scripts/eval_league.py [candidate.py] [--seeds screen|confirm|all|s1,s2]
                                [--opponents dir] [--json out.json] [--check-determinism]

Default candidate is the committed main.py. Reports, per opponent and both seats,
the candidate win-rate / diff_min / diff_mean / candidate-min-money, plus an
overall field win-rate. Prints a self-mirror row (candidate vs itself) as the
reference point for the self-mirror <-> league divergence.
"""
import sys, os, json, argparse, importlib.util, statistics, hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Held-out seed split. CONFIRM mirrors the SOT-2381 confirm set for continuity so
# league numbers are directly comparable to the recorded self-mirror A/B.
SCREEN_SEEDS = [101, 202]
CONFIRM_SEEDS = [7, 42, 303, 404, 505, 777, 1234, 2026, 5555, 9001]


def _load_agent(path):
    """Load a fresh agent callable from a python file, isolated in sys.modules."""
    path = Path(path)
    uniq = "leagueagent_%s" % hashlib.sha1(str(path).encode()).hexdigest()[:10]
    # Ensure a fresh module object each call (tape agents may hold global state).
    sys.modules.pop(uniq, None)
    spec = importlib.util.spec_from_file_location(uniq, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[uniq] = mod              # register BEFORE exec (dataclass/global resolution)
    spec.loader.exec_module(mod)
    return mod.agent


def _run_pair(cand_path, opp_path, seed):
    """Run one game with FRESH instances; return (cand_money, opp_money) from
    the candidate's perspective, run at BOTH seats. Returns (dA, dB, rows)."""
    from kaggle_environments import make
    out = []
    # seat A: candidate is player 0
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.run([_load_agent(cand_path), _load_agent(opp_path)])
    s = env.state
    cA, oA = float(s[0].observation.farms[0]["money"]), float(s[0].observation.farms[1]["money"])
    statusA = [p.status for p in s]
    # seat B: candidate is player 1
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.run([_load_agent(opp_path), _load_agent(cand_path)])
    s = env.state
    cB, oB = float(s[0].observation.farms[1]["money"]), float(s[0].observation.farms[0]["money"])
    statusB = [p.status for p in s]
    return (cA - oA, cB - oB), [(seed, "A", cA, oA, statusA), (seed, "B", cB, oB, statusB)]


def evaluate(cand_path, opp_path, seeds, label=""):
    diffs, rows, cand_moneys, opp_moneys, statuses = [], [], [], [], []
    for sd in seeds:
        (dA, dB), rr = _run_pair(cand_path, opp_path, sd)
        diffs += [dA, dB]
        for _, seat, cm, om, st in rr:
            cand_moneys.append(cm); opp_moneys.append(om); statuses.append(st)
        rows += rr
    wins = sum(1 for d in diffs if d > 0)
    n = len(diffs)
    all_done = all(st == ["DONE", "DONE"] for st in statuses)
    summ = {
        "opponent": Path(opp_path).name,
        "label": label,
        "games": n,
        "wins": wins,
        "win_rate": wins / n if n else 0.0,
        "diff_min": min(diffs) if diffs else 0.0,
        "diff_mean": statistics.mean(diffs) if diffs else 0.0,
        "cand_min_money": min(cand_moneys) if cand_moneys else 0.0,
        "cand_mean_money": statistics.mean(cand_moneys) if cand_moneys else 0.0,
        "opp_mean_money": statistics.mean(opp_moneys) if opp_moneys else 0.0,
        "sign": "ALL WIN" if diffs and min(diffs) > 0 else "ALL LOSE" if diffs and max(diffs) < 0 else "MIXED",
        "all_done": all_done,
    }
    return summ, rows


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _parse_subject(spec):
    """Parse a calibration subject spec `path[:label[:real_lb]]`.

    real_lb may be empty (unknown candidate). Paths never contain ':' here.
    """
    parts = spec.split(":")
    path = parts[0]
    label = parts[1] if len(parts) > 1 and parts[1] else Path(path).stem
    lb = None
    if len(parts) > 2 and parts[2] not in ("", "None"):
        lb = float(parts[2])
    return {"path": path, "label": label, "lb": lb, "sha": _sha256(path)}


def _field_stats(per_opp, opp_names):
    """Aggregate a field win-rate / games-weighted diff_mean over a subset of
    opponents (by file name). Returns (wins, games, win_rate, diff_mean, diff_min)."""
    wins = sum(per_opp[o]["wins"] for o in opp_names if o in per_opp)
    games = sum(per_opp[o]["games"] for o in opp_names if o in per_opp)
    dmean = (sum(per_opp[o]["diff_mean"] * per_opp[o]["games"]
                 for o in opp_names if o in per_opp) / games) if games else 0.0
    dmin = min((per_opp[o]["diff_min"] for o in opp_names if o in per_opp), default=0.0)
    return wins, games, (wins / games if games else 0.0), dmean, dmin


def calibrate(subjects, opp_paths, seeds, json_out=""):
    """Field-oracle calibration against known real-LB anchors.

    Each subject is scored vs every opponent whose bytes differ from it (self-play
    guard, so an anchor that is also in the pool — e.g. soil — never mirror-matches
    itself). Real LB is a *field* ranking (each agent vs a diverse field), NOT a
    head-to-head between two agents: soil beats moon-198 head-to-head (self-mirror
    oracle) yet is field-inferior on the real LB (600 vs 2543.4). So the calibration
    metric is FIELD win-rate, compared on the common opponent set two subjects share.
    Calibration PASSES iff the league's field ordering is monotone with real LB for
    every LB-known subject pair.
    """
    opp_sha = {p: _sha256(p) for p in opp_paths}
    subj_shas = {s["sha"] for s in subjects}
    print(f"# CALIBRATE :: subjects={[s['label'] for s in subjects]} :: "
          f"seeds={seeds} :: opponents={len(opp_paths)}")

    per_subject = {}
    for s in subjects:
        per_opp = {}
        for opp in opp_paths:
            if opp_sha[opp] == s["sha"]:
                continue  # self-play guard: skip the pool entry identical to this subject
            summ, _ = evaluate(s["path"], opp, seeds, label=Path(opp).name)
            per_opp[Path(opp).name] = summ
        fw, fg, fr, fdm, fdmin = _field_stats(per_opp, list(per_opp))
        per_subject[s["label"]] = {
            "label": s["label"], "path": s["path"], "sha16": s["sha"][:16],
            "real_lb": s["lb"], "per_opp": per_opp,
            "field_wins": fw, "field_games": fg, "field_win_rate": fr,
            "field_diff_mean": fdm, "field_diff_min": fdmin,
        }
        print(f"\n== subject {s['label']} (sha {s['sha'][:16]}, real_lb={s['lb']}) ==")
        for o, summ in per_opp.items():
            print(f"  vs {o:<38} {summ['wins']:>2}/{summ['games']:<2} "
                  f"({summ['win_rate']*100:5.1f}%)  diff_mean={summ['diff_mean']:+11.1f}  {summ['sign']}")
        print(f"  -> FULL field {fw}/{fg} = {fr*100:.1f}%  diff_mean={fdm:+.1f}  diff_min={fdmin:+.1f}")

    # Pairwise comparisons on the COMMON opponent set (apples-to-apples).
    labels = list(per_subject)
    print("\n#### PAIRWISE COMMON-FIELD COMPARISONS ####")
    comparisons = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = per_subject[labels[i]], per_subject[labels[j]]
            common = sorted(set(a["per_opp"]) & set(b["per_opp"]))
            aw, ag, ar, adm, admin = _field_stats(a["per_opp"], common)
            bw, bg, br, bdm, bdmin = _field_stats(b["per_opp"], common)
            cmp = {
                "a": a["label"], "b": b["label"], "common_opponents": common,
                "a_win_rate": ar, "b_win_rate": br, "a_diff_mean": adm, "b_diff_mean": bdm,
                "a_lb": a["real_lb"], "b_lb": b["real_lb"],
                "field_delta_a_minus_b": adm - bdm,
            }
            comparisons.append(cmp)
            print(f"\n  [{a['label']}] vs [{b['label']}] on common field {common}:")
            print(f"    {a['label']}: {aw}/{ag}={ar*100:.1f}%  diff_mean={adm:+.1f}")
            print(f"    {b['label']}: {bw}/{bg}={br*100:.1f}%  diff_mean={bdm:+.1f}")
            if a["real_lb"] is not None and b["real_lb"] is not None:
                lb_order = "a>b" if a["real_lb"] > b["real_lb"] else "a<b" if a["real_lb"] < b["real_lb"] else "a==b"
                fld_order = "a>b" if ar > br else "a<b" if ar < br else "a==b"
                ok = (lb_order == fld_order) or lb_order == "a==b"
                cmp["real_lb_order"] = lb_order
                cmp["field_order"] = fld_order
                cmp["monotone"] = ok
                print(f"    real_lb: {a['label']}={a['real_lb']} {b['label']}={b['real_lb']} "
                      f"({lb_order}) | field: ({fld_order}) -> "
                      f"{'REPRODUCES real-LB order' if ok else 'CONTRADICTS real-LB order'}")

    lb_cmps = [c for c in comparisons if "monotone" in c]
    calibrated = bool(lb_cmps) and all(c["monotone"] for c in lb_cmps)
    print("\n#### CALIBRATION VERDICT ####")
    print(f"  LB-anchored pairs checked: {len(lb_cmps)}  "
          f"monotone: {sum(1 for c in lb_cmps if c['monotone'])}/{len(lb_cmps)}")
    print(f"  CALIBRATED = {calibrated}  "
          f"({'league field ordering reproduces the known real-LB ordering' if calibrated else 'league does NOT reproduce real-LB ordering — oracle still unreliable'})")

    if json_out:
        Path(json_out).write_text(json.dumps({
            "mode": "calibrate", "seeds": seeds,
            "subjects": {k: {kk: vv for kk, vv in v.items() if kk != "per_opp"} | {
                "per_opp": {o: s for o, s in v["per_opp"].items()}} for k, v in per_subject.items()},
            "comparisons": comparisons,
            "calibrated": calibrated,
        }, indent=2))
        print(f"\n# wrote {json_out}")
    return calibrated


def _seeds_arg(val):
    if val == "screen":
        return SCREEN_SEEDS
    if val == "confirm":
        return CONFIRM_SEEDS
    if val == "all":
        return SCREEN_SEEDS + CONFIRM_SEEDS
    return [int(x) for x in val.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", nargs="?", default=str(REPO / "main.py"))
    ap.add_argument("--seeds", default="all")
    ap.add_argument("--opponents", default=str(REPO / "opponents"))
    ap.add_argument("--json", default="")
    ap.add_argument("--check-determinism", action="store_true")
    ap.add_argument("--self-mirror", action="store_true",
                    help="also score candidate vs itself as the self-mirror reference")
    ap.add_argument("--calibrate", action="store_true",
                    help="calibration mode: rank --subjects by field win-rate and check "
                         "monotonicity vs known real-LB scores")
    ap.add_argument("--subjects", default="",
                    help="comma-separated calibration subjects `path[:label[:real_lb]]`")
    args = ap.parse_args()

    seeds = _seeds_arg(args.seeds)

    if args.calibrate:
        opp_dir = Path(args.opponents)
        opp_paths = sorted(str(p) for p in opp_dir.glob("*.py") if p.name != "__init__.py")
        if not args.subjects:
            print("ERROR: --calibrate requires --subjects", file=sys.stderr)
            sys.exit(2)
        subjects = [_parse_subject(x.strip()) for x in args.subjects.split(",") if x.strip()]
        calibrate(subjects, opp_paths, seeds, json_out=args.json)
        return
    cand = args.candidate
    opp_dir = Path(args.opponents)
    opp_paths = sorted(str(p) for p in opp_dir.glob("*.py") if p.name != "__init__.py")

    print(f"# replay-league :: candidate={Path(cand).name} :: seeds={seeds} :: opponents={len(opp_paths)}")
    print(f"# candidate sha256={hashlib.sha256(Path(cand).read_bytes()).hexdigest()[:16]}")

    if args.check_determinism and opp_paths:
        (d1, _), _ = _run_pair(cand, opp_paths[0], seeds[0])
        (d2, _), _ = _run_pair(cand, opp_paths[0], seeds[0])
        print(f"# determinism check on {Path(opp_paths[0]).name} seed {seeds[0]}: "
              f"run1={d1} run2={d2} -> {'DETERMINISTIC' if d1 == d2 else 'NON-DETERMINISTIC!'}")

    results = []
    all_diffs = []
    for opp in opp_paths:
        summ, rows = evaluate(cand, opp, seeds, label=Path(opp).name)
        results.append(summ)
        for (_, seat, cm, om, st) in rows:
            all_diffs.append(cm - om)
        print(f"\n== vs {summ['opponent']} ==")
        for (sd, seat, cm, om, st) in rows:
            d = cm - om
            print(f"  seed {sd:>5} seat {seat}: cand={cm:10.1f}  opp={om:10.1f}  diff={d:+11.1f}  "
                  f"{'WIN' if d > 0 else 'LOSE' if d < 0 else 'TIE'}  {st}")
        print(f"  -> win_rate={summ['win_rate']*100:.0f}% ({summ['wins']}/{summ['games']})  "
              f"diff_min={summ['diff_min']:+.1f}  diff_mean={summ['diff_mean']:+.1f}  "
              f"cand_min_money={summ['cand_min_money']:.1f}  sign={summ['sign']}  all_done={summ['all_done']}")

    if args.self_mirror:
        summ, rows = evaluate(cand, cand, seeds, label="SELF-MIRROR")
        results.append(summ)
        print(f"\n== SELF-MIRROR (cand vs cand) ==")
        print(f"  -> win_rate={summ['win_rate']*100:.0f}%  diff_min={summ['diff_min']:+.1f}  "
              f"diff_mean={summ['diff_mean']:+.1f}  (near 0 expected)  all_done={summ['all_done']}")

    field_wins = sum(1 for d in all_diffs if d > 0)
    field_n = len(all_diffs)
    print(f"\n#### FIELD SUMMARY (excl. self-mirror) ####")
    print(f"  opponents={len(opp_paths)}  games={field_n}  "
          f"field_win_rate={field_wins}/{field_n}={field_wins/field_n*100:.1f}%  "
          f"field_diff_min={min(all_diffs):+.1f}  field_diff_mean={statistics.mean(all_diffs):+.1f}")
    beaten = [r['opponent'] for r in results if r['label'] != 'SELF-MIRROR' and r['win_rate'] == 1.0]
    lost = [r['opponent'] for r in results if r['label'] != 'SELF-MIRROR' and r['win_rate'] == 0.0]
    mixed = [r['opponent'] for r in results if r['label'] != 'SELF-MIRROR' and 0.0 < r['win_rate'] < 1.0]
    print(f"  ALL-WIN vs: {beaten}")
    print(f"  ALL-LOSE vs: {lost}")
    print(f"  MIXED vs: {mixed}")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "candidate": Path(cand).name,
            "candidate_sha256": hashlib.sha256(Path(cand).read_bytes()).hexdigest(),
            "seeds": seeds,
            "per_opponent": results,
            "field_win_rate": field_wins / field_n if field_n else 0.0,
            "field_diff_min": min(all_diffs) if all_diffs else 0.0,
            "field_diff_mean": statistics.mean(all_diffs) if all_diffs else 0.0,
        }, indent=2))
        print(f"\n# wrote {args.json}")


if __name__ == "__main__":
    main()
