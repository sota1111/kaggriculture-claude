"""Local eval harness for kaggriculture agents (SOT-2260).

Runs head-to-head env.run([A, B]) mirrors and reports player-0 money and the
mean/min diff over a seed list. Agents are loaded from a python file exporting
`agent(obs)`; the champion is the committed main.py, candidates are given paths.
"""
import sys, importlib.util, statistics
from kaggle_environments import make

def load_agent(path):
    spec = importlib.util.spec_from_file_location("cand_%d" % (abs(hash(path)) % 10**6), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent

def run_pair(a, b, seed):
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.run([a, b])
    s = env.state
    m0 = s[0].observation.farms[0]["money"]
    m1 = s[0].observation.farms[1]["money"]
    return float(m0), float(m1)

def evaluate(cand_path, opp_path, seeds, label=""):
    a = load_agent(cand_path)
    b = load_agent(opp_path)
    rows = []
    for sd in seeds:
        m0, m1 = run_pair(a, b, sd)
        rows.append((sd, m0, m1, m0 - m1))
    diffs = [r[3] for r in rows]
    means = statistics.mean([r[1] for r in rows])
    print(f"== {label} :: cand={cand_path.split('/')[-1]} vs opp={opp_path.split('/')[-1]} ==")
    for sd, m0, m1, d in rows:
        print(f"  seed {sd:>4}: cand={m0:9.1f}  opp={m1:9.1f}  diff={d:+9.1f}  {'WIN' if d>0 else 'LOSE' if d<0 else 'TIE'}")
    print(f"  cand_mean={means:.1f}  diff_mean={statistics.mean(diffs):+.1f}  diff_min={min(diffs):+.1f}  "
          f"sign={'ALL WIN' if min(diffs)>0 else 'ALL LOSE' if max(diffs)<0 else 'MIXED'}")
    return rows

if __name__ == "__main__":
    cand = sys.argv[1]
    opp = sys.argv[2]
    seeds = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1,2,3,4,5]
    label = sys.argv[4] if len(sys.argv) > 4 else ""
    evaluate(cand, opp, seeds, label)
