"""High-revenue crop portfolio + glut-aware selling + animal husbandry for Kaggriculture.

Builds on the multi-worker routing engine (SOT-2259 + SOT-2261), the crop-economics
portfolio (SOT-2260), and glut-aware rationed selling (SOT-2298). Those cycles maximized
crop revenue: patrol all 25 NW tiles, plant the high-value MELON / STRAWBERRY demand
sinks, and meter sells so scarce produce clears above the market floor.

The lever here (SOT-2297) is **animal husbandry** — an *exclusive* demand sink the
opponent cannot free-ride. The prior land-expansion cycle (SOT-2299) was competitively
dominated: expanding into MELON / STRAWBERRY loses head-to-head because the non-expanding
champion free-rides the same shared crop sink (tragedy of the commons). Its recorded
conclusion: retry only with a sink the champion supplies **zero** of, whose revenue beats
its labor cost. `EGG` / `MILK` / `WOOL` come only from `ANIMALS` (`GOOSE→EGG`,
`COW→MILK`, `SHEEP→WOOL`); the crop-only champion never touches those markets, so their
town-center + shop demand keeps them scarce and high-priced (self-mirror end-market
diagnosis: MILK ≈ $351, WOOL ≈ $247, both scarce with zero supply). MILK (base 160,
`sqrt` scarcity) and WOOL (base 200) are the value targets; EGG (base 50, shallow `linear`
scarcity) is marginal.

SOT-2344 scaled the plan from COW2/SHEEP2 to **COW3/SHEEP3** (6 animals) by fixing the
feed logistics that made larger plans collapse: the day-0 bulk animal buy used to starve
the wheat feed reserve, so animals went unfed and escaped. The market pass now reserves
feed cash first and buys animals staged, one at a time, as crop revenue accrues.

Implementation: a few of the low-value WHEAT tiles (the high-value MELON 10 / STRAWBERRY 8
allocation is preserved) are converted to COOP / PASTURE structures near the shed. A small
set of dedicated *rancher* workers builds the structures, buys + places the animals, and
each day feeds (1 WHEAT/animal, drawn from the shed — a wheat reserve is held back from
selling), cares (a fed-day-only yield bonus), and harvests the product into the shed for
the existing glut-aware seller. The env animal constants (first-yield/interval/max_held,
wheat feed, care-on-fed-only) are mirrored exactly. Non-rancher workers and the non-animal
tiles keep the unchanged crop engine. Only `math` is imported so the file runs under
Kaggle's exec harness (no `__file__`, no cwd use).

SOT-2342 adds **fertilizer synergy**, layered on top of the ranch. Every surviving animal
emits one free `FERTILIZER` byproduct per day (env sets `fertilizer_available=True` in
`_daily_refresh_animals`); it was previously wasted. A `FERTILIZE`d plant carries
`fertilized_until_day = day + 2` (3 active days), and on a *watered* production day the env
accrues +2 yield instead of +1 (env L384 / L768-769). Only an *ongoing* crop turns that
into extra harvested units: STRAWBERRY (interval 2, max_yield 4) doubles each of its four
production events (ages 10/12/14/16), roughly doubling strawberry output. MELON is skipped —
it is non-ongoing and already reaches its max_yield 6 cap from daily watering alone (7 waters
in the age 6–12 window ≥ cap 6), so fertilizing it is pure waste. The ranchers collect the
byproduct as their *lowest* animal-chore priority (below feed/harvest/care, so animals are
never starved for it) and, with genuine slack, courier it to the nearest in-window
STRAWBERRY and spray it. Unlike the rejected land expansion (SOT-2299/SOT-2343, which lost
head-to-head to a free-riding champion because the extra sink cost land+labor+cash), this
lever wins decisively: it has *zero* capital cost (the fertilizer is free) and only marginal
rancher labor, and the doubled high-value strawberry beats the shared-sink price depression.
vs the COW3/SHEEP3 champion: ALL WIN on 20 disjoint seeds (diff_min +8714, ~+9.5k mean, cand
~60–68k vs champ ~50–58k); self-mirror honest-glut ≈ +8.9k/player symmetric absolute gain
(so not a head-to-head-only artifact). Guarded by `FERTILIZE_ENABLED`; off ⇒ behavior is
identical to the prior champion. See docs/measurements/SOT-2342.md.
"""

import math

# Crop parameters, mirrored from the competition's CROPS table.
CROPS = {
    "WHEAT":      {"seed": 10,  "first_yield_day": 2,  "max_yield_day": 4,  "interval": 0, "max_yield": 6, "ongoing": False},
    "CARROT":     {"seed": 20,  "first_yield_day": 2,  "max_yield_day": 3,  "interval": 0, "max_yield": 4, "ongoing": False},
    "TOMATO":     {"seed": 50,  "first_yield_day": 8,  "max_yield_day": 8,  "interval": 1, "max_yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "max_yield": 4, "ongoing": True},
    "MELON":      {"seed": 80,  "first_yield_day": 10, "max_yield_day": 12, "interval": 0, "max_yield": 6, "ongoing": False},
}

# Animal parameters, mirrored from the competition's ANIMALS table.
ANIMALS = {
    "GOOSE": {"cost": 300, "structure": "COOP",    "first_yield_day": 4, "interval": 1, "max_held": 4, "product": "EGG"},
    "COW":   {"cost": 400, "structure": "PASTURE", "first_yield_day": 8, "interval": 2, "max_held": 6, "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "first_yield_day": 6, "interval": 3, "max_held": 6, "product": "WOOL"},
}
_BUILD_OP = {"COOP": "BUILD_COOP", "PASTURE": "BUILD_PASTURE"}

# Tile allocation across demand sinks (rest of the open NW slots default to WHEAT).
# Chosen by a real-env self-mirror sweep; see the module docstring / measurements.
PORTFOLIO = [("MELON", 10), ("STRAWBERRY", 8)]

# Animal husbandry plan: (animal, count). Structures occupy the NW tiles nearest the
# shed so ranchers tour them cheaply. Swept on the real env; MILK/WOOL are the value
# targets, EGG (GOOSE) is marginal.
# SOT-2297 shipped COW2/SHEEP2 because larger plans collapsed (~300-420 self-mirror);
# SOT-2344 traced that collapse to the *day-0 bulk animal buy* starving the wheat feed
# reserve (animals unfed 2 days escape) — NOT to a demand limit. With the feed-cash-
# priority staged buyer below, the symmetric COW3/SHEEP3 plan (6 animals, all survive)
# is the new optimum: ALL WIN on 20 disjoint seeds vs the COW2/SHEEP2 champion
# (diff_min +1705..+2250, cand ~53-54.6k vs ~51-52k). Asymmetric 3+2/2+3 plans stay
# fragile (single-seed collapses) and 7-8 animals (4+3, 4+4, +GOOSE2) over-extend again
# or hit MILK/WOOL sink saturation (GOOSE only +263). See docs/measurements/SOT-2344.md.
ANIMAL_PLAN = [("COW", 3), ("SHEEP", 3)]
N_RANCHERS = 2               # dedicated animal-chore workers (rest patrol crops)
WHEAT_FEED_RESERVE_DAYS = 3  # shed wheat held back from selling to guarantee feed
#   (rd>=4 is a fragile cash/feed knife-edge that collapses vs champion — keep at 3)

TARGET_HANDS = 6  # farm hands hired each morning (env resets them nightly)
SEED_BUFFER = 2   # per-crop seed headroom beyond the open target slots

# --- Fertilizer synergy (SOT-2342): collect the free animal FERTILIZER byproduct
# and spray it on STRAWBERRY to double its per-event yield accrual (1->2). ---
# Each surviving animal makes 1 FERTILIZER/day (env `_daily_refresh_animals` sets
# `fertilizer_available=True`), currently wasted. A `FERTILIZE`d PLANT gets
# `fertilized_until_day=day+2` (3 days active); on a watered production day the env
# adds +2 instead of +1 yield (env L384 / L768-769). Only ongoing crops benefit at
# harvest: STRAWBERRY (interval 2, max_yield 4) doubles each of its 4 production
# events. MELON is a non-ongoing crop that already reaches its max_yield 6 cap from
# daily watering alone (window ages 6-12 = 7 waters >= cap 6), so fertilizing it is
# pure waste — excluded. Fertilizer is a FREE byproduct (no capital cost), but it is
# a SHARED market sink (like SOT-2299 land), so the vs-champion sign gate decides.
# Guarded by FERTILIZE_ENABLED: with it off the file is behavior-identical to the
# committed champion (clean revert on non-promotion).
FERTILIZE_ENABLED = True
FERT_CROP = "STRAWBERRY"          # the only crop the doubling actually reaches harvest
FERT_MIN_AGE = 9                  # start spraying just before first_yield_day (10)
FERT_MAX_AGE = 16                 # last production event age (10,12,14,16)
FERT_CARRY_CAP = 3                # max FERTILIZER a rancher hoards before delivering

# --- Sell metering (SOT-2298): ration sells to the demand drains, avoid glut. ---
# Hold back any unit whose marginal sell price would drop below SELL_FLOOR_FRAC · base
# (keeping market inventory near/under I0 = scarcity-premium territory); dump the held
# remainder on the final day so nothing rots (unsold shed = $0 at game end).
SELL_FLOOR_FRAC = 0.85
LIQUIDATE_DAY = 29  # last game day (episodeSteps 720 / turnsPerDay 24 → days 0..29)

# Market pricing table, mirrored from the env's MARKET_PARAMS (base / I0 / T and the
# below/above shape + target that set amp = target · base / f(T)). Used only to predict
# the marginal sell price so we can meter quantity; the env remains the source of truth.
_MARKET = {
    "WHEAT":      {"base":  25, "I0": 10000, "T": 400, "bf": "sqrt",   "bt": 0.80, "af": "log",    "at": 0.20},
    "CARROT":     {"base":  35, "I0": 10000, "T": 450, "bf": "log",    "bt": 0.20, "af": "sqrt",   "at": 0.70},
    "TOMATO":     {"base":  60, "I0": 10000, "T": 200, "bf": "linear", "bt": 0.40, "af": "sqrt",   "at": 0.60},
    "STRAWBERRY": {"base": 120, "I0": 10000, "T": 100, "bf": "sqrt",   "bt": 0.70, "af": "linear", "at": 1.60},
    "MELON":      {"base": 250, "I0": 10000, "T": 300, "bf": "log",    "bt": 0.20, "af": "sq",     "at": 3.60},
    "EGG":        {"base":  50, "I0": 10000, "T": 332, "bf": "linear", "bt": 0.40, "af": "log",    "at": 0.20},
    "MILK":       {"base": 160, "I0": 10000, "T": 122, "bf": "sqrt",   "bt": 0.60, "af": "linear", "at": 1.60},
    "WOOL":       {"base": 200, "I0": 10000, "T": 105, "bf": "log",    "bt": 0.20, "af": "sq",     "at": 3.20},
    "FERTILIZER": {"base": 100, "I0": 10000, "T": 200, "bf": "linear", "bt": 0.40, "af": "linear", "at": 0.40},
}


def _shape(func, x):
    x = x if x > 0 else 0.0
    if func == "linear": return x
    if func == "sq":     return x * x
    if func == "sqrt":   return math.sqrt(x)
    if func == "log":    return math.log(1.0 + x)
    return x


def _market_price(item, inv):
    """Predicted unit price at market inventory `inv` (mirrors env market_price)."""
    p = _MARKET.get(item)
    if p is None:
        return None
    base, i0, t = p["base"], p["I0"], p["T"]
    if inv < i0:
        amp = p["bt"] * base / _shape(p["bf"], t)
        price = base + amp * _shape(p["bf"], i0 - inv)
    else:
        amp = p["at"] * base / _shape(p["af"], t)
        price = base - amp * _shape(p["af"], inv - i0)
    return max(1, int(round(price)))


def _meter_sell_qty(item, qty, inv0, day):
    """Units of `item` to sell this turn: hold those whose marginal price < floor.

    Selling raises market inventory by 1 per unit, so unit j clears at price(inv0 + j).
    Stop once that marginal price drops below SELL_FLOOR_FRAC · base — the held units
    wait for the town drains to reopen headroom. On the final day, dump everything.
    """
    if qty <= 0:
        return 0
    if day >= LIQUIDATE_DAY:
        return qty
    p = _MARKET.get(item)
    if p is None:
        return qty
    threshold = SELL_FLOOR_FRAC * p["base"]
    inv = int(inv0)
    k = 0
    while k < qty and _market_price(item, inv) >= threshold:
        k += 1
        inv += 1
    return k


def agent(obs):
    player = int(obs["player"])
    me = obs["farms"][player]
    private = obs["private"]
    day = int(obs["day"])
    hour = int(obs.get("hour", 0))
    tiles = me["tiles"]
    seeds = private.get("seeds", {}) or {}
    shed = private.get("shed", {}) or {}
    inventories = private.get("inventories", []) or []
    money = float(me["money"])
    hands = me.get("hands", []) or []
    market = obs.get("market", {}) or {}
    prices = market.get("prices", {}) or {}
    inventory = market.get("inventory", {}) or {}

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

    # --- Animal tiles: the nearest cluster tiles to the shed (excluding the spawn
    # tile itself, which stays a clean pickup/logistics slot). ---
    animal_assign = {}   # (x, y) -> animal name
    ranch_candidates = [p for p in cluster if p != (spawn_x, spawn_y)]
    ai = 0
    for animal, count in ANIMAL_PLAN:
        for _ in range(count):
            if ai < len(ranch_candidates):
                animal_assign[ranch_candidates[ai]] = animal
                ai += 1
    animal_tiles = set(animal_assign)

    # Crop tiles = the rest of the cluster; assign the crop portfolio (rest WHEAT).
    crop_cluster = [p for p in cluster if p not in animal_tiles]
    crop_cluster_set = set(crop_cluster)
    tile_crop = {}
    idx = 0
    for crop, count in PORTFOLIO:
        for _ in range(count):
            if idx < len(crop_cluster):
                tile_crop[crop_cluster[idx]] = crop
                idx += 1
    for p in crop_cluster:
        tile_crop.setdefault(p, "WHEAT")

    def cdata(t):
        return CROPS[t["crop"]]

    def crop_age(t):
        return day - int(t["planted_day"])

    def is_plant(t):
        return isinstance(t, dict) and t.get("kind") == "PLANT" and t.get("crop") in CROPS

    def is_animal(t):
        return isinstance(t, dict) and "animal" in t

    def is_structure(t):
        return isinstance(t, dict) and t.get("kind") in _BUILD_OP and "animal" not in t

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
    claimed = set()      # tiles claimed this turn
    busy = set()         # worker indices already assigned an action

    def worker_inv(i):
        if 0 <= i < len(inventories) and isinstance(inventories[i], dict):
            return inventories[i]
        return {}

    # =====================================================================
    # RANCH PASS (SOT-2297): dedicate the last N_RANCHERS workers to animals.
    # Ranchers build structures, place bought animals, and each day feed / care /
    # harvest. A rancher with no reachable animal chore falls through to crops.
    # =====================================================================
    n_placed = {a: 0 for a in ANIMALS}
    for (x, y), animal in animal_assign.items():
        t = tiles[y][x]
        if is_animal(t) and t.get("animal") == animal:
            n_placed[animal] += 1

    def _step_towards(fx, fy, tx, ty):
        if tx != fx:
            return ["EAST"] if tx > fx else ["WEST"]
        if ty != fy:
            return ["SOUTH"] if ty > fy else ["NORTH"]
        return None

    def ranch_action(i):
        """Best animal chore (or a step toward it) for rancher `i`; None if idle."""
        fx, fy = workers[i]
        inv_i = worker_inv(i)
        wheat = int(inv_i.get("WHEAT", 0))
        at_shed = (fx, fy) == (spawn_x, spawn_y)

        # 1. Act in place if standing on one of my animal tiles.
        if (fx, fy) in animal_assign:
            animal = animal_assign[(fx, fy)]
            t = tiles[fy][fx]
            struct = ANIMALS[animal]["structure"]
            if t is None:
                claimed.add((fx, fy))
                return [_BUILD_OP[struct]]
            if is_structure(t) and t.get("kind") == struct and int(inv_i.get(animal, 0)) > 0:
                claimed.add((fx, fy))
                return ["PLACE", animal]
            if is_animal(t) and t.get("animal") == animal:
                if not t.get("fed_today") and wheat > 0:
                    claimed.add((fx, fy))
                    return ["FEED"]
                if int(t.get("yield_units", 0)) > 0:
                    claimed.add((fx, fy))
                    return ["HARVEST"]
                if t.get("fed_today") and not t.get("cared_today"):
                    claimed.add((fx, fy))
                    return ["CARE"]
                # Lowest animal-chore priority: pocket the free daily fertilizer
                # (only while there is unfertilized strawberry demand and carry room).
                if (FERTILIZE_ENABLED and t.get("fertilizer_available")
                        and fert_targets
                        and int(inv_i.get("FERTILIZER", 0)) < FERT_CARRY_CAP):
                    claimed.add((fx, fy))
                    return ["COLLECT_FERTILIZER"]

        # 2. Restock at the shed when I lack the item a pending chore needs.
        need_feed = any(
            is_animal(tiles[y][x]) and not tiles[y][x].get("fed_today")
            for (x, y) in animal_assign
        )
        want_animal = None
        for (x, y), animal in animal_assign.items():
            t = tiles[y][x]
            if (t is None or is_structure(t)) and int(inv_i.get(animal, 0)) == 0 \
               and int(shed.get(animal, 0)) > 0:
                want_animal = animal
                break
        if (need_feed and wheat == 0) or want_animal is not None:
            if at_shed:
                if want_animal is not None:
                    return ["PICKUP", want_animal, 1]
                take = len(animal_tiles) + 1
                if int(shed.get("WHEAT", 0)) > 0:
                    return ["PICKUP", "WHEAT", take]
                return None  # nothing to pick up yet
            mv = _step_towards(fx, fy, spawn_x, spawn_y)
            if mv:
                return mv

        # 3. Route toward the nearest actionable animal tile.
        best, best_key = None, None
        for (x, y), animal in animal_assign.items():
            if (x, y) in claimed:
                continue
            t = tiles[y][x]
            if t is None:
                rank = 3                      # build
            elif is_structure(t) and int(inv_i.get(animal, 0)) > 0:
                rank = 2                      # place
            elif is_animal(t) and t.get("animal") == animal and (
                (not t.get("fed_today") and wheat > 0)
                or int(t.get("yield_units", 0)) > 0
                or (t.get("fed_today") and not t.get("cared_today"))
            ):
                rank = 0 if not t.get("fed_today") else 1  # feed first, else harvest/care
            else:
                continue
            key = (rank, abs(x - fx) + abs(y - fy))
            if best_key is None or key < best_key:
                best_key, best = key, (x, y)
        if best is not None:
            claimed.add(best)
            return _step_towards(fx, fy, best[0], best[1])

        # 4. Deliver value: carrying fertilizer -> spray the nearest in-window
        # STRAWBERRY. Strictly below every animal survival chore above, so animals
        # are never neglected for fertilizer. Only claim a tile when acting on it
        # in place (routing must not block a crop worker from watering it).
        if FERTILIZE_ENABLED and int(inv_i.get("FERTILIZER", 0)) > 0 and fert_targets:
            if (fx, fy) in fert_targets:
                claimed.add((fx, fy))
                return ["FERTILIZE"]
            best_f, best_fk = None, None
            for (x, y) in fert_targets:
                key = abs(x - fx) + abs(y - fy)
                if best_fk is None or key < best_fk:
                    best_fk, best_f = key, (x, y)
            if best_f is not None:
                return _step_towards(fx, fy, best_f[0], best_f[1])

        # 5. Otherwise route to an animal whose fertilizer byproduct is waiting, but
        # only while strawberry demand exists and I have room to carry more.
        if (FERTILIZE_ENABLED and fert_targets
                and int(inv_i.get("FERTILIZER", 0)) < FERT_CARRY_CAP):
            best_c, best_ck = None, None
            for (x, y), animal in animal_assign.items():
                t = tiles[y][x]
                if is_animal(t) and t.get("animal") == animal and t.get("fertilizer_available"):
                    key = abs(x - fx) + abs(y - fy)
                    if best_ck is None or key < best_ck:
                        best_ck, best_c = key, (x, y)
            if best_c is not None:
                return _step_towards(fx, fy, best_c[0], best_c[1])
        return None

    # STRAWBERRY tiles that would benefit from fertilizer now: a live plant inside
    # its production age window whose fertilized window has lapsed.
    fert_targets = set()
    if FERTILIZE_ENABLED:
        for (x, y) in crop_cluster:
            t = tiles[y][x]
            if (is_plant(t) and t.get("crop") == FERT_CROP
                    and FERT_MIN_AGE <= crop_age(t) <= FERT_MAX_AGE
                    and int(t.get("fertilized_until_day", -1)) < day):
                fert_targets.add((x, y))

    ranchers = list(range(max(1, n - N_RANCHERS), n)) if n > 1 else []
    for i in ranchers:
        act = ranch_action(i)
        if act is not None:
            unit_actions[i] = act
            busy.add(i)

    # =====================================================================
    # CROP PASSES: unchanged engine over the non-ranch workers + crop tiles.
    # =====================================================================
    plant_budget = {c: int(seeds.get(c, 0)) for c in CROPS}

    def slot_op(pos):
        x, y = pos
        if (x, y) not in crop_cluster_set:
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
    for i in range(n):
        if i in busy:
            continue
        pos = workers[i]
        op = slot_op(pos)
        if op is not None and pos not in claimed:
            claimed.add(pos)
            unit_actions[i] = op
            busy.add(i)
        else:
            pending.append(i)

    # Pass 2: workers on an empty target slot plant its crop (respect seed budget).
    still = []
    for i in pending:
        pos = workers[i]
        crop = tile_crop.get(pos)
        if (
            pos in crop_cluster_set
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
        for (x, y) in crop_cluster:
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

    # =====================================================================
    # MARKET: hire; buy animals; keep a wheat feed reserve; sell; buy seed.
    # =====================================================================
    orders = []
    if hour == 0:
        for _ in range(max(0, TARGET_HANDS - len(hands))):
            orders.append(["HIRE"])

    # Buy animals with FEED-CASH PRIORITY + staging (SOT-2344). An animal that
    # misses 2 consecutive feed days escapes (env `_daily_refresh_animals`), so a
    # purchase the farm can't also feed is pure loss. The prior plan bought the whole
    # ANIMAL_PLAN in one day-0 order (`money >= cost + 500`), which for a 6-animal plan
    # sank ~$2.7k up front and left too little to buy the wheat feed reserve — every
    # animal starved and the farm collapsed to ~$500 (self-mirror). Instead: reserve the
    # cash to buy the wheat feed reserve first, then buy animals ONE AT A TIME,
    # interleaved across types, only while cash stays above (feed reserve + buffer).
    # Remaining animals are bought on later days as crop revenue accrues (staging).
    wheat_price = _market_price("WHEAT", int(inventory.get("WHEAT", 10000)) - 1) or 25
    wheat_reserve = len(animal_tiles) * WHEAT_FEED_RESERVE_DAYS
    shed_wheat = int(shed.get("WHEAT", 0))
    feed_cash = max(0, wheat_reserve - shed_wheat) * wheat_price
    CASH_BUFFER = 500

    animal_buys = []
    if animal_tiles:
        deficit = {}
        for animal, count in ANIMAL_PLAN:
            have = n_placed.get(animal, 0) + int(shed.get(animal, 0))
            have += sum(int(iv.get(animal, 0)) for iv in inventories if isinstance(iv, dict))
            deficit[animal] = max(0, count - have)
        buy_ct = {a: 0 for a in deficit}
        progress = True
        while progress:
            progress = False
            for animal, _count in ANIMAL_PLAN:
                if deficit[animal] - buy_ct[animal] <= 0:
                    continue
                cost = ANIMALS[animal]["cost"]
                if money - cost >= feed_cash + CASH_BUFFER:
                    money -= cost
                    buy_ct[animal] += 1
                    progress = True
        for animal, _count in ANIMAL_PLAN:
            if buy_ct[animal] > 0:
                animal_buys.append(["BUY_ANIMAL", animal, buy_ct[animal]])

    # Guarantee wheat feed: top the shed up to the reserve (cash reserved above so this
    # order — queued after the animal buys — still clears on the same turn).
    feed_buys = []
    if wheat_reserve > 0:
        deficit_w = wheat_reserve - shed_wheat
        if deficit_w > 0 and money >= wheat_price * deficit_w:
            feed_buys.append(["BUY_PRODUCT", "WHEAT", deficit_w])

    # Sell every shed product, highest unit price first, metered to the demand drains.
    # Hold back the wheat feed reserve so animals never starve.
    sellable = []
    for item, qty in shed.items():
        q = int(qty) if qty else 0
        if q <= 0 or item not in prices:
            continue
        if item == "WHEAT" and day < LIQUIDATE_DAY:
            q = max(0, q - wheat_reserve)
        if q <= 0:
            continue
        inv0 = inventory.get(item, 10000)
        sell_qty = _meter_sell_qty(item, q, inv0, day)
        if sell_qty > 0:
            sellable.append((float(prices.get(item, 0)), item, sell_qty))
    sellable.sort(reverse=True)
    sells = [["SELL", item, qty] for _, item, qty in sellable]

    # Seed buys: cover the open crop target slots per crop plus a small buffer.
    want = {}
    for p in crop_cluster:
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

    # Order budget: labor + animals (setup) first, then feed, high-value sells, seed.
    market_orders = (orders + animal_buys + feed_buys + sells + buy_orders)[:10]

    return {"farmer": farmer, "hands": hands_out, "market": market_orders}
