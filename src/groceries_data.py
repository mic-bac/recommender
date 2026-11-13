"""
groceries_data.py
-----------------
Loading the Groceries market-basket data — with a synthetic fallback whose
answer is known in advance.

The real dataset is a Kaggle download (see README) and is git-ignored, so a fresh
clone — or the deployed teaching app — may not have it. Rather than fail, we
generate baskets in the same format (``Member_number``, ``Date``,
``itemDescription``) and say so.

The synthetic data is built to be *interpretable*, which turns a fallback into a
teaching device:

- **Staples with realistic popularity.** Whole milk is in a large share of
  baskets, vegetables and rolls close behind — exactly like the real file.
  Staples are drawn independently of each other, so rules between them come out
  with **lift ~ 1**: frequent, but not associated. "Popular" is not "related".
- **Planted shopping missions.** A share of baskets is built around a mission —
  baking, a barbecue, breakfast, pasta night, a party, baby care — whose items
  are bought *together*. These are the associations Apriori should find, with
  lift far above 1. ``PLANTED_MISSIONS`` is the ground truth to check against.

Pure and importable (no printing, no plotting), so the lesson script and the
teaching app use the very same generator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_CSV = Path(__file__).resolve().parents[1] / "data" / "groceries" / "Groceries_dataset.csv"

# Everyday items and roughly how often each lands in a basket on its own. Drawn
# independently, so rules among them come out with lift close to 1 (slightly
# below, because an empty basket is not a transaction and is never recorded).
STAPLES = {
    "whole milk": 0.25,
    "other vegetables": 0.20,
    "rolls/buns": 0.18,
    "soda": 0.16,
    "yogurt": 0.14,
    "bottled water": 0.11,
    "root vegetables": 0.11,
    "tropical fruit": 0.10,
    "shopping bags": 0.09,
    "citrus fruit": 0.08,
    "pastry": 0.08,
    "newspapers": 0.07,
    "fruit/vegetable juice": 0.06,
    "brown bread": 0.06,
    "domestic eggs": 0.06,
    "margarine": 0.05,
    "coffee": 0.05,
    "chocolate": 0.05,
    "frozen vegetables": 0.04,
    "chicken": 0.04,
    "white bread": 0.04,
    "napkins": 0.04,
    "cream cheese": 0.04,
    "beef": 0.04,
    # Mission items are also bought on their own now and then — otherwise their
    # lift would be absurd (an item that only ever appears with its partners).
    "butter": 0.04,
    "bottled beer": 0.06,
    "sausage": 0.06,
    "salty snack": 0.03,
    "onions": 0.03,
    "hard cheese": 0.02,
    "candy": 0.02,
    "sugar": 0.02,
    "hygiene articles": 0.02,
    "pasta": 0.02,
    "jam": 0.01,
    "flour": 0.01,
    "mustard": 0.01,
    "canned tomatoes": 0.01,
    "popcorn": 0.01,
    "baking powder": 0.005,
    "baby food": 0.012,
    "baby cosmetics": 0.010,
}

# Items bought together for one purpose: (core items, chance each core item is
# in a mission basket, share of all baskets built around this mission).
PLANTED_MISSIONS = {
    "Baking": (["flour", "sugar", "baking powder", "butter"], 0.85, 0.030),
    "Barbecue": (["sausage", "bottled beer", "mustard", "salty snack"], 0.80, 0.040),
    "Breakfast": (["rolls/buns", "butter", "jam", "coffee"], 0.75, 0.040),
    "Pasta night": (["pasta", "canned tomatoes", "hard cheese", "onions"], 0.80, 0.035),
    "Party": (["soda", "salty snack", "popcorn", "candy"], 0.75, 0.030),
    "Baby care": (["baby food", "hygiene articles", "baby cosmetics"], 0.85, 0.015),
}


def make_synthetic_groceries(
    n_members: int = 3_000, n_baskets: int = 15_000, random_state: int = 42
) -> pd.DataFrame:
    """Generate baskets in the Kaggle file's format, with known associations.

    One row per (member, date, item) — the same long format as the real file, so
    the lesson's grouping into baskets (member + date) works unchanged.
    """
    rng = np.random.default_rng(random_state)
    staple_names = np.array(list(STAPLES))
    staple_p = np.array(list(STAPLES.values()))
    missions = list(PLANTED_MISSIONS.values())
    mission_share = np.array([share for _, _, share in missions])

    dates = pd.date_range("2014-01-01", "2015-12-30", freq="D")
    rows = []
    for basket_id in range(n_baskets):
        member = int(rng.integers(1000, 1000 + n_members))
        date = dates[rng.integers(len(dates))].strftime("%d-%m-%Y")
        items = set(staple_names[rng.random(len(staple_names)) < staple_p])

        # At most one mission per basket, with the probabilities above.
        draw = rng.random()
        cumulative = np.cumsum(mission_share)
        hit = np.searchsorted(cumulative, draw, side="right")
        if hit < len(missions):
            core, p_core, _ = missions[hit]
            chosen = [item for item in core if rng.random() < p_core]
            items.update(chosen if len(chosen) >= 2 else core[:2])

        if not items:  # every basket buys at least one thing
            items = {str(rng.choice(staple_names, p=staple_p / staple_p.sum()))}
        rows.extend((member, date, item) for item in sorted(items))

    return pd.DataFrame(rows, columns=["Member_number", "Date", "itemDescription"])


def load_groceries(csv_path: Path | str = DATA_CSV) -> tuple[pd.DataFrame, str]:
    """Return ``(frame, source)`` — the Kaggle file if present, else synthetic.

    ``source`` is ``"kaggle"`` or ``"synthetic"`` so the caller can be honest
    about which data the rules came from.
    """
    csv_path = Path(csv_path)
    if csv_path.exists():
        return pd.read_csv(csv_path), "kaggle"
    return make_synthetic_groceries(), "synthetic"
