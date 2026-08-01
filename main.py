"""High-revenue crop portfolio + glut-aware selling for Kaggriculture.

Builds on the multi-worker routing engine (SOT-2259 + SOT-2261). Those cycles
maximized *throughput* (patrol all 25 NW tiles with a farmer + 5 hired hands) but
planted a single crop — WHEAT — and dumped it all onto one market channel.

The lever here is *crop economics*. The env's market prices each product with
`price(inv) = base ± amp·f(|inv - I0|)`, `I0 = 10000`, and a per-product town
center + shop demand schedule that *drains* inventory below I0 (scarcity) for any
product nobody supplies. All-WHEAT play sells wheat at ~$20 (its `log` glut curve
is shallow, but base is only 25 and every shop already sinks wheat, so its price
sits near base) while the TOMATO / STRAWBERRY / MELON markets sit far ABOVE base
($105 / $321 / $293 with near-zero supply) because their demand goes unmet.

Measured $/tile-day (real env, self-mirror) is dominated by:
  - MELON      base 250, in **0 shops** → only the town-center demand sink (~140
                units/game) supports it, but that sink pays ~$260/unit — the
                highest value-density crop when supply is kept under the sink.
  - STRAWBERRY base 120 but scarcity ~$321, and it feeds **4 shops** (BRUNCH,
                ICE_CREAM, SMOOTHIE, FARMERS_MARKET) → the largest demand sink,
                so many units clear at a high price.
  - WHEAT      fast (first yield day 2) early-game cashflow while the slow
                high-value crops (first yield day 10-12) mature.

Over-planting any one product gluts it (melon's `sq` glut collapses its price to
the $1 floor past the sink; strawberry's `linear` glut is gentler), so the tiles
are **diversified** across the three demand sinks. A real-env allocation sweep
under a symmetric self-mirror (both farmers flood the same crops — the honest
glut test) picked MELON 10 / STRAWBERRY 8 / WHEAT 7: self-mirror money 10.9k →
35.7k, and vs the all-wheat engine champion +30k (≈41.5k vs ≈11.2k), sign-
consistent across 15 seeds.

Selling is glut-aware: every turn each product in the shed is sold highest-unit-
price-first (the visible `obs["market"]["prices"]`), so scarce high-value produce
clears before cheap wheat and no order budget is wasted dumping a floored product
ahead of a profitable one. Only self-contained Python is used so the file runs
under Kaggle's exec harness (no imports, no `__file__`, no cwd use).
"""

# Crop parameters, mirrored from the competition's CROPS table.
CROPS = {
    "WHEAT":      {"seed": 10,  "first_yield_day": 2,  "max_yield_day": 4,  "interval": 0, "max_yield": 6, "ongoing": False},
    "CARROT":     {"seed": 20,  "first_yield_day": 2,  "max_yield_day": 3,  "interval": 0, "max_yield": 4, "ongoing": False},
    "TOMATO":     {"seed": 50,  "first_yield_day": 8,  "max_yield_day": 8,  "interval": 1, "max_yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "max_yield": 4, "ongoing": True},
    "MELON":      {"seed": 80,  "first_yield_day": 10, "max_yield_day": 12, "interval": 0, "max_yield": 6, "ongoing": False},
}

# Tile allocation across demand sinks (rest of the 25 NW slots default to WHEAT).
# Chosen by a real-env self-mirror sweep; see the module docstring / measurements.
PORTFOLIO = [("MELON", 10), ("STRAWBERRY", 8)]

TARGET_HANDS = 5  # farm hands hired each morning (env resets them nightly)
SEED_BUFFER = 2   # per-crop seed headroom beyond the open target slots


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
    market = obs.get("market", {}) or {}
    prices = market.get("prices", {}) or {}

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

    # Assign a target crop to each cluster tile from PORTFOLIO (rest WHEAT).
    tile_crop = {}
    idx = 0
    for crop, count in PORTFOLIO:
        for _ in range(count):
            if idx < len(cluster):
                tile_crop[cluster[idx]] = crop
                idx += 1
    for p in cluster:
        tile_crop.setdefault(p, "WHEAT")

    def cdata(t):
        return CROPS[t["crop"]]

    def crop_age(t):
        return day - int(t["planted_day"])

    def is_plant(t):
        return isinstance(t, dict) and t.get("kind") == "PLANT" and t.get("crop") in CROPS

    def need_harvest(t):
        if not is_plant(t) or int(t.get("yield_units", 0)) <= 0:
            return False
        c = cdata(t)
        if c["ongoing"]:
            return True  # collect ongoing yield as soon as it accrues
        return crop_age(t) >= c["max_yield_day"]  # non-ongoing: harvest at full yield

    def need_water(t):
        # Water any live plant not yet watered today: watering is free, adds a
        # yield unit inside the yield window, and keeps the plant alive (a plant
        # dies after 2 consecutive dry days). Harvest is prioritized above this.
        if not is_plant(t) or t.get("watered_today"):
            return False
        c = cdata(t)
        if not c["ongoing"] and crop_age(t) > c["max_yield_day"]:
            return False  # spent non-ongoing crop: just harvest it
        return True

    def need_dig(t):
        return isinstance(t, dict) and t.get("kind") == "WEED"

    # --- Worker roster: index 0 = farmer, 1.. = hands (each has its own inv). ---
    workers = [tuple(me["farmer"])] + [tuple(h) for h in hands]
    n = len(workers)
    unit_actions = [["PASS"] for _ in range(n)]
    claimed = set()

    # Per-crop atomic seed budget: at most this many PLANTs of each crop per turn.
    plant_budget = {c: int(seeds.get(c, 0)) for c in CROPS}

    def slot_op(pos):
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
        op = slot_op(pos)
        if op is not None and pos not in claimed:
            claimed.add(pos)
            unit_actions[i] = op
        else:
            pending.append(i)

    # Pass 2: workers on an empty target slot plant its crop (respect seed budget).
    still = []
    for i in pending:
        pos = workers[i]
        crop = tile_crop.get(pos)
        if (
            pos in cluster_set
            and tiles[pos[1]][pos[0]] is None
            and pos not in claimed
            and crop and plant_budget.get(crop, 0) > 0
        ):
            claimed.add(pos)
            plant_budget[crop] -= 1
            unit_actions[i] = ["PLANT", crop]
        else:
            still.append(i)

    # Pass 3: route remaining workers toward the nearest unclaimed task tile.
    def nearest_target(fx, fy):
        best = None
        best_key = None
        best_plant = False
        for (x, y) in cluster:
            if (x, y) in claimed:
                continue
            t = tiles[y][x]
            plant_here = False
            if need_harvest(t):
                rank = 0
            elif need_water(t):
                rank = 1
            elif t is None and plant_budget.get(tile_crop.get((x, y)), 0) > 0:
                rank = 2
                plant_here = True
            elif need_dig(t):
                rank = 3
            else:
                continue
            key = (rank, abs(x - fx) + abs(y - fy))
            if best_key is None or key < best_key:
                best_key, best, best_plant = key, (x, y), plant_here
        return best, best_plant

    for i in still:
        fx, fy = workers[i]
        target, plant_here = nearest_target(fx, fy)
        if target is None:
            continue
        claimed.add(target)
        if plant_here:
            plant_budget[tile_crop.get(target)] -= 1
        tx, ty = target
        if tx != fx:
            unit_actions[i] = ["EAST"] if tx > fx else ["WEST"]
        elif ty != fy:
            unit_actions[i] = ["SOUTH"] if ty > fy else ["NORTH"]

    farmer = unit_actions[0]
    hands_out = unit_actions[1:]

    # --- Market: hire each morning; sell high-value produce first; buy seed. ---
    orders = []
    if hour == 0:
        for _ in range(max(0, TARGET_HANDS - len(hands))):
            orders.append(["HIRE"])

    # Sell every shed product, ordered by its current unit price (desc) so scarce
    # high-value produce clears before cheap wheat within the 10-order budget.
    sellable = [
        (float(prices.get(item, 0)), item, qty)
        for item, qty in shed.items()
        if qty and qty > 0 and item in prices
    ]
    sellable.sort(reverse=True)
    sells = [["SELL", item, qty] for _, item, qty in sellable]

    # Seed buys: cover the open target slots per crop plus a small buffer.
    want = {}
    for p in cluster:
        if tiles[p[1]][p[0]] is None:
            crop = tile_crop.get(p, "WHEAT")
            want[crop] = want.get(crop, 0) + 1
    buys = []
    for crop, open_slots in want.items():
        deficit = (open_slots + SEED_BUFFER) - int(seeds.get(crop, 0))
        if deficit > 0:
            cost = CROPS[crop]["seed"] * deficit
            if money >= cost:
                buys.append((CROPS[crop]["seed"], ["BUY_SEED", crop, deficit]))
    buys.sort()  # cheaper seeds first (they gate the most tiles)
    buy_orders = [b for _, b in buys]

    # Order budget: HIRE (labor) first, then high-value sells, then seed buys.
    market_orders = (orders + sells + buy_orders)[:10]

    return {"farmer": farmer, "hands": hands_out, "market": market_orders}
