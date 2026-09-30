"""EDA gates G1-G10 (plan section 3). Label statistics use train only; test is read label-free."""
from __future__ import annotations

from collections import Counter

import numpy as np

from .normalize import cty_key, normalize_name

PLACEHOLDERS = {"", "na", "n/a", "-", "--", "none", "null", "nil", "0", "."}


def _rows(split):
    out = {}
    for name, df in (("S1", split["s1"]), ("pool", split["pool"])):
        c = Counter(zip(df["entity_id"].str[:2], (cty_key(x) for x in df["country"])))
        for (src, cty), n in sorted(c.items()):
            out[f"{src}|{cty}"] = n
    return out


def gates(train, test):
    g = {}
    gold = train["gold"]
    g["G1_rows_train"] = _rows(train)
    g["G1_rows_test"] = _rows(test)
    cnt = np.array([len(v) for v in gold.values()])
    g["G2_match_count_share"] = {k: float(np.mean(np.minimum(cnt, 3) == i)) for i, k in
                                 enumerate(["0", "1", "2", "3+"])}
    g["G2_match_count_p99_max"] = [float(np.percentile(cnt, 99)) if len(cnt) else 0.0, int(cnt.max()) if len(cnt) else 0]
    all_ids = [x for v in gold.values() for x in v]
    src = Counter(x[:2] for x in all_ids)
    g["G2_source_share_of_matches"] = {k: v / max(len(all_ids), 1) for k, v in sorted(src.items())}
    multi = Counter(all_ids)
    g["G3_pool_ids_in_2plus_lists"] = int(sum(1 for v in multi.values() if v > 1))
    g["G3_exclusivity_holds"] = g["G3_pool_ids_in_2plus_lists"] == 0
    g["G4_pool_share_matched"] = len(multi) / max(len(train["pool"]), 1)
    c1 = dict(zip(train["s1"]["entity_id"], (cty_key(x) for x in train["s1"]["country"])))
    cp = dict(zip(train["pool"]["entity_id"], (cty_key(x) for x in train["pool"]["country"])))
    same = [c1.get(s) == cp.get(x) for s, v in gold.items() for x in v]
    g["G5_same_country_rate"] = float(np.mean(same)) if same else 1.0
    g["G6_s1_with_2plus_from_same_source"] = float(np.mean([
        max(Counter(x[:2] for x in v).values()) >= 2 for v in gold.values() if v])) if any(gold.values()) else 0.0
    names = [" ".join(normalize_name(n)["core"]) for n in train["s1"]["business_name"]]
    nc = Counter(zip(names, (cty_key(x) for x in train["s1"]["country"])))
    g["G8_s1_names_shared_by_2plus"] = int(sum(v for v in nc.values() if v > 1))
    pn = Counter(" ".join(normalize_name(n)["core"]) for n in train["pool"]["business_name"])
    g["G8_top_pool_names"] = pn.most_common(10)
    drift = {}
    for split_name, split in (("train", train), ("test", test)):
        for key, df in (("S1", split["s1"]), ("pool", split["pool"])):
            ct = np.array([cty_key(x) for x in df["country"]], dtype=object)
            for cty in sorted(set(ct)):
                m = ct == cty
                drift[f"{split_name}|{key}|{cty}"] = {
                    "n": int(m.sum()),
                    "name_len_mean": float(df["business_name"][m].str.len().mean()),
                    "addr_len_mean": float(df["business_address"][m].str.len().mean()),
                    "addr_empty_rate": float((df["business_address"][m].str.strip() == "").mean())}
    g["G9_drift"] = drift
    ph = {}
    for split_name, split in (("train", train), ("test", test)):
        for key, df in (("S1", split["s1"]), ("pool", split["pool"])):
            for col in ("business_name", "business_address", "country"):
                v = df[col].str.strip().str.lower()
                ph[f"{split_name}|{key}|{col}"] = int(v.isin(PLACEHOLDERS).sum())
    g["G10_placeholder_counts"] = ph
    return g
