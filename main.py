"""Multi-tile farmer routing engine for Kaggriculture (Claude champion).

The previous champion farmed a single tile with a plain wheat loop. This agent
turns the farmer into a state machine that patrols the whole unlocked NW
quadrant (5x5), so many wheat tiles are planted, watered and harvested in
parallel instead of one at a time. Produce accumulates in the farmer's
inventory (auto-dropped to the shed at end of day) and is sold from the shed
every turn via the market channel, which runs independently of the farmer's
single physical action per turn.

Routing each turn (greedy, nearest-task first):
  HARVEST a mature tile under the farmer > WATER a plant that needs it >
  DIG a weed blocking a slot > PLANT on an empty slot > otherwise step one tile
  toward the nearest tile that needs service.

Wheat timing (env constants): first_yield_day=2, max_yield_day=4, so a plant is
watered on age 0 (keep-alive; a plant weeds after two unwatered days and the
planting day already counts as one) and on age 2,3,4 (each watering inside the
[(max_yield_day+1)//2, max_yield_day] window adds a yield unit), then harvested
at age 4 with a full 4-unit yield. Only self-contained Python is used so the
file runs under Kaggle's exec harness (no imports, no __file__, no cwd use).
"""

# Wheat parameters, mirrored from the competition's CROPS table.
CROP = "WHEAT"
SEED_COST = 10
FIRST_YIELD_DAY = 2
MAX_YIELD_DAY = 4
HARVEST_AGE = MAX_YIELD_DAY  # harvest once the full yield has accrued
SEED_BUFFER = 8              # keep enough seed on hand to fill open slots


def agent(obs):
    player = int(obs["player"])
    me = obs["farms"][player]
    private = obs["private"]
    day = int(obs["day"])
    tiles = me["tiles"]
    fx, fy = me["farmer"]
    seeds = private.get("seeds", {}) or {}
    shed = private.get("shed", {}) or {}
    money = float(me["money"])

    board = len(tiles)
    half = board // 2
    # Spawn / shed-access corner of the always-unlocked NW quadrant.
    spawn_x, spawn_y = half - 1, half - 1

    # Farmable slots: every NW tile, ordered by distance from spawn so the
    # patrol prefers nearby tiles and wastes fewer moves.
    cluster = sorted(
        ((x, y) for x in range(half) for y in range(half)),
        key=lambda p: (abs(p[0] - spawn_x) + abs(p[1] - spawn_y), p),
    )
    cluster_set = set(cluster)

    def crop_age(t):
        return day - int(t["planted_day"])

    def is_crop(t):
        return isinstance(t, dict) and t.get("kind") == "PLANT" and t.get("crop") == CROP

    def need_harvest(t):
        return is_crop(t) and int(t.get("yield_units", 0)) > 0 and crop_age(t) >= HARVEST_AGE

    def need_water(t):
        if not is_crop(t) or t.get("watered_today"):
            return False
        a = crop_age(t)
        if a > MAX_YIELD_DAY:
            return False
        # Age 1 needs no water when the plant was watered on its planting day
        # (it survives a single dry day); every other age up to harvest does.
        if a == 1 and int(t.get("consecutive_unwatered", 0)) < 1:
            return False
        return True

    def need_dig(t):
        return isinstance(t, dict) and t.get("kind") == "WEED"

    def is_empty(t):
        return t is None

    def nearest(pred):
        best = None
        best_d = None
        for (x, y) in cluster:
            if pred(tiles[y][x]):
                d = abs(x - fx) + abs(y - fy)
                if best_d is None or d < best_d:
                    best_d = d
                    best = (x, y)
        return best

    cur = tiles[fy][fx]
    on_slot = (fx, fy) in cluster_set
    have_seed = int(seeds.get(CROP, 0)) > 0

    # --- Farmer: act on the current tile, else step toward the best target. ---
    if need_harvest(cur):
        farmer = ["HARVEST"]
    elif need_water(cur):
        farmer = ["WATER"]
    elif on_slot and need_dig(cur):
        farmer = ["DIG"]
    elif on_slot and is_empty(cur) and have_seed:
        farmer = ["PLANT", CROP]
    else:
        target = nearest(need_harvest) or nearest(need_water)
        if target is None and have_seed:
            target = nearest(is_empty)
        if target is None:
            target = nearest(need_dig)
        if target is None:
            farmer = ["PASS"]
        else:
            tx, ty = target
            if tx != fx:
                farmer = ["EAST"] if tx > fx else ["WEST"]
            elif ty != fy:
                farmer = ["SOUTH"] if ty > fy else ["NORTH"]
            else:
                farmer = ["PASS"]

    # --- Market: sell shed produce; keep a seed buffer for open slots. ---
    market = []
    stock = int(shed.get(CROP, 0))
    if stock > 0:
        market.append(["SELL", CROP, stock])

    open_slots = sum(1 for (x, y) in cluster if tiles[y][x] is None)
    want_seed = min(SEED_BUFFER, open_slots)
    have = int(seeds.get(CROP, 0))
    if have < want_seed:
        buy = want_seed - have
        if money >= SEED_COST * buy:
            market.append(["BUY_SEED", CROP, buy])

    return {"farmer": farmer, "hands": [], "market": market}
