"""Validate a Kaggriculture main.py without requiring the game package."""

import importlib.util
import sys
from pathlib import Path


path = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("submission", path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
assert callable(module.agent)

obs = {
    "player": 0,
    "day": 0,
    "farms": [{"money": 100, "farmer": [0, 0], "tiles": [[None]]}],
    "private": {"shed": {}, "seeds": {"WHEAT": 1}, "inventories": [[]]},
}
result = module.agent(obs)
assert set(result) == {"farmer", "hands", "market"}
assert isinstance(result["farmer"], list)
assert isinstance(result["hands"], list)
assert isinstance(result["market"], list)
print("Kaggriculture submission contract: PASS")
