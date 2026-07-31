"""Deterministic Kaggriculture Claude baseline.

The agent maintains a small wheat loop: buy seed, plant, water, harvest, and
sell produce from the shed. Invalid actions are avoided and every observation
returns the competition's required action dictionary.
"""


def agent(obs):
    player = int(obs["player"])
    me = obs["farms"][player]
    private = obs["private"]
    fx, fy = me["farmer"]
    tile = me["tiles"][fy][fx]

    market = []
    wheat = int(private["shed"].get("WHEAT", 0))
    if wheat:
        market.append(["SELL", "WHEAT", wheat])
    if int(private["seeds"].get("WHEAT", 0)) == 0 and int(me["money"]) >= 10:
        market.append(["BUY_SEED", "WHEAT", 1])

    farmer = ["PASS"]
    if tile is None and int(private["seeds"].get("WHEAT", 0)) > 0:
        farmer = ["PLANT", "WHEAT"]
    elif isinstance(tile, dict) and tile.get("kind") == "PLANT":
        age = int(obs["day"]) - int(tile["planted_day"])
        if age >= 2:
            farmer = ["HARVEST"]
        elif not tile.get("watered_today", False):
            farmer = ["WATER"]

    return {"farmer": farmer, "hands": [], "market": market}
