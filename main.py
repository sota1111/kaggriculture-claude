"""Multi-worker (farmer + hired hands) routing engine for Kaggriculture.

Builds on the single-farmer multi-tile router (SOT-2259). A lone farmer makes
only one physical action per turn and cannot keep all 25 unlocked NW wheat tiles
planted, watered and harvested on time: each tile needs ~6 actions per 4-day
cycle, so 25 tiles need ~150 actions per 96 turns — the single farmer is
throughput-bound and leaves tiles idle. This agent HIREs farm hands each morning
and routes every worker (farmer + hands) in parallel over the NW quadrant, so
many more tiles are serviced per turn.

Hands cost the env's Fibonacci hire sequence (1,1,2,3,5,8… ×farmHandCostMult);
at the default mult=1, TARGET_HANDS=5 costs only 1+1+2+3+5=12/day (~360 over the
30-day game) — trivially recovered by the extra harvests. Hands and their
per-worker inventories are dissolved every end-of-day and re-spawned at the shed
corners, so hiring repeats every morning (`hour == 0`).

Routing each turn (greedy, nearest-unclaimed-task first, one worker per tile):
  HARVEST a mature tile > WATER a plant that needs it > PLANT on an empty slot
  (bounded by seeds on hand) > DIG a weed > otherwise step one tile toward the
  nearest tile that needs service. Produce accrues in each worker's inventory,
  is auto-dropped to the shed at end of day, and is sold from the shed every turn
  on the independent market channel.

Wheat timing (env constants): first_yield_day=2, max_yield_day=4, so a plant is
watered on age 0 (keep-alive) and on ages 2,3,4 (each watering inside the
[(max_yield_day+1)//2, max_yield_day] window adds a yield unit), then harvested
at age 4 with a full yield. Only self-contained Python is used so the file runs
under Kaggle's exec harness (no imports, no __file__, no cwd use).
"""

# Wheat parameters, mirrored from the competition's CROPS table.
CROP = "WHEAT"
SEED_COST = 10
FIRST_YIELD_DAY = 2
MAX_YIELD_DAY = 4
HARVEST_AGE = MAX_YIELD_DAY  # harvest once the full yield has accrued
SEED_BUFFER = 16             # seed on hand to (re)fill open slots across workers
TARGET_HANDS = 5             # farm hands hired each morning (env resets them nightly)


def agent(obs):
    player = int(obs["player"])
    me = obs["farms"][player]
    private = obs["private"]
    day = int(obs["day"])
    hour = int(obs.get("hour", 0))
    tiles = me["tiles"]
    seeds = private.get("seeds", {}) or {}
    shed = private.get("shed", {}) or {}
    money = float(me["money"])
    hands = me.get("hands", []) or []

    board = len(tiles)
    half = board // 2
    # Spawn / shed-access corner of the always-unlocked NW quadrant.
    spawn_x, spawn_y = half - 1, half - 1

    # Farmable slots: every NW tile, ordered by distance from spawn so patrols
    # prefer nearby tiles and waste fewer moves.
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

    # --- Worker roster: index 0 = farmer, 1.. = hands (each has its own inv). ---
    workers = [tuple(me["farmer"])] + [tuple(h) for h in hands]
    n = len(workers)
    unit_actions = [["PASS"] for _ in range(n)]

    have_seed = int(seeds.get(CROP, 0))
    plant_budget = have_seed        # atomic seed cap: at most this many PLANTs/turn
    claimed = set()                 # tiles reserved for acting or as a move target

    def on_slot_op(pos):
        """Immediate action for a worker standing on `pos`, if the tile needs it."""
        x, y = pos
        if (x, y) not in cluster_set:
            return None
        t = tiles[y][x]
        if need_harvest(t):
            return ["HARVEST"]
        if need_water(t):
            return ["WATER"]
        if need_dig(t):
            return ["DIG"]
        return None

    # Pass 1: workers already on a serviceable tile act in place (claim it).
    pending = []
    for i, pos in enumerate(workers):
        op = on_slot_op(pos)
        if op is not None and pos not in claimed:
            claimed.add(pos)
            unit_actions[i] = op
        else:
            pending.append(i)

    # Pass 2: workers on an empty in-cluster slot plant (respecting seed budget).
    still = []
    for i in pending:
        pos = workers[i]
        if (
            pos in cluster_set
            and is_empty(tiles[pos[1]][pos[0]])
            and pos not in claimed
            and plant_budget > 0
        ):
            claimed.add(pos)
            plant_budget -= 1
            unit_actions[i] = ["PLANT", CROP]
        else:
            still.append(i)

    # Pass 3: route remaining workers toward the nearest unclaimed task tile.
    def nearest_target(fx, fy, allow_plant):
        best = None
        best_key = None
        for (x, y) in cluster:
            if (x, y) in claimed:
                continue
            t = tiles[y][x]
            if need_harvest(t):
                rank = 0
            elif need_water(t):
                rank = 1
            elif allow_plant and is_empty(t):
                rank = 2
            elif need_dig(t):
                rank = 3
            else:
                continue
            key = (rank, abs(x - fx) + abs(y - fy))
            if best_key is None or key < best_key:
                best_key, best = key, (x, y)
        is_plant = best_key is not None and best_key[0] == 2
        return best, is_plant

    plant_reservations = 0
    for i in still:
        fx, fy = workers[i]
        allow_plant = plant_budget - plant_reservations > 0
        target, is_plant = nearest_target(fx, fy, allow_plant)
        if target is None:
            unit_actions[i] = ["PASS"]
            continue
        claimed.add(target)
        if is_plant:
            plant_reservations += 1
        tx, ty = target
        if tx != fx:
            unit_actions[i] = ["EAST"] if tx > fx else ["WEST"]
        elif ty != fy:
            unit_actions[i] = ["SOUTH"] if ty > fy else ["NORTH"]
        else:
            unit_actions[i] = ["PASS"]

    farmer = unit_actions[0]
    hands_out = unit_actions[1:]

    # --- Market: hire each morning; sell shed produce; keep a seed buffer. ---
    market = []
    if hour == 0:
        for _ in range(max(0, TARGET_HANDS - len(hands))):
            market.append(["HIRE"])

    stock = int(shed.get(CROP, 0))
    if stock > 0:
        market.append(["SELL", CROP, stock])

    open_slots = sum(1 for (x, y) in cluster if tiles[y][x] is None)
    want_seed = min(SEED_BUFFER, open_slots)
    if have_seed < want_seed:
        buy = want_seed - have_seed
        if money >= SEED_COST * buy:
            market.append(["BUY_SEED", CROP, buy])

    return {"farmer": farmer, "hands": hands_out, "market": market}
