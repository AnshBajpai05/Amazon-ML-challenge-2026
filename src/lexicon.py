"""Data-driven lexicon for normalization (built from the provided data only - no external resources).

indic   native-script word -> English word, learned from TRAIN gold pairs whose pool record is written in an Indic
        script ('इंटरनेशनल' -> 'international', 'மீடியா' -> 'media'). Phonetic transliteration alone gives
        'intaraneshanal' / 'gret' / 'midiya', which neither blocking nor the string features can match.
        Names are aligned word by word (same word count); address words are matched to the S1 address word whose
        spelling is closest to their transliteration.
vocab   word counts of every Source-1 name (train + test S1 files, no labels), used to split concatenated website /
        handle names ('olaniqpetcare.com' -> 'olaniq pet care').

    python -m src.lexicon      # writes src/lexicon.json (normalize.py loads it at import)
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.csv as pcsv

LEX_PATH = Path(__file__).with_name("lexicon.json")
INDIC_RE = "[ऀ-෿]"
_WORD = re.compile(r"[^\s,;:|()\[\]/.\-]+")


def _read(path, cols):
    opts = pcsv.ParseOptions(delimiter="\t", quote_char=False, newlines_in_values=False)
    ropts = pcsv.ReadOptions(block_size=1 << 26)
    conv = pcsv.ConvertOptions(include_columns=cols, strings_can_be_null=False)
    return pcsv.read_csv(path, read_options=ropts, parse_options=opts, convert_options=conv)


def _latin_words(s):
    from .normalize import fold
    return re.findall(r"[a-z0-9]+", fold(s))


def build(data_dir, min_count=2, min_share=0.6):
    from rapidfuzz.distance import JaroWinkler

    from .normalize import fold, has_indic
    d = Path(data_dir)
    t0 = time.time()
    # ---- vocabulary of S1 names (train + test S1 files; unlabeled text)
    vocab = Counter()
    for split in ("train", "test"):
        t = _read(d / split / f"{split}_source1.tsv", ["business_name"])
        for name in t.column(0).to_pylist():
            vocab.update(w for w in _latin_words(name or "") if len(w) >= 2 and not w.isdigit())
    print(f"vocab: {len(vocab):,} words from S1 names ({time.time() - t0:.0f}s)", flush=True)

    # ---- Indic word -> English word from train gold pairs
    s1 = _read(d / "train" / "train_source1.tsv", ["entity_id", "business_name", "business_address"]).to_pandas()
    s1 = s1.set_index("entity_id")
    pools = []
    for k in (2, 3):
        t = _read(d / "train" / f"train_source{k}.tsv", ["entity_id", "business_name", "business_address"])
        m = pc.or_(pc.match_substring_regex(t.column("business_name"), INDIC_RE),
                   pc.match_substring_regex(t.column("business_address"), INDIC_RE))
        pools.append(t.filter(m).to_pandas())
    pool = pd.concat(pools, ignore_index=True).set_index("entity_id")
    gt = pd.read_csv(d / "train" / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    owner = {}
    for s, ms in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        for m in ms.split(","):
            m = m.strip()
            if m in pool.index:
                owner[m] = s
    print(f"indic pool records: {len(pool):,}, with a gold S1: {len(owner):,} ({time.time() - t0:.0f}s)", flush=True)
    name_pairs, addr_pairs = Counter(), Counter()
    for pid, sid in owner.items():
        pn, pa = pool.at[pid, "business_name"] or "", pool.at[pid, "business_address"] or ""
        sn, sa = s1.at[sid, "business_name"] or "", s1.at[sid, "business_address"] or ""
        if has_indic(pn):
            nw = [w for w in _WORD.findall(pn) if has_indic(w)]
            lw = _latin_words(sn)
            allw = _WORD.findall(pn)
            if len(allw) == len(lw) and nw:                     # word-by-word translation of the name
                for w, e in zip(allw, lw):
                    if has_indic(w) and not e.isdigit():
                        name_pairs[(w, e)] += 1
        if has_indic(pa):
            sw = [w for w in _latin_words(sa) if not w.isdigit() and len(w) >= 3]
            for w in _WORD.findall(pa):
                if not has_indic(w) or not sw:
                    continue
                tr = re.sub(r"[^a-z]", "", fold(w))
                if len(tr) < 3:
                    continue
                best = max(sw, key=lambda e: JaroWinkler.similarity(tr, e))
                if JaroWinkler.similarity(tr, best) >= 0.8:
                    addr_pairs[(w, best)] += 1

    def resolve(pairs):
        by = defaultdict(Counter)
        for (w, e), n in pairs.items():
            by[w][e] += n
        out = {}
        for w, c in by.items():
            e, n = c.most_common(1)[0]
            if n >= min_count and n / sum(c.values()) >= min_share:
                out[w] = e
        return out
    indic = resolve(addr_pairs)
    indic.update(resolve(name_pairs))                           # names win on conflicts
    print(f"indic lexicon: {len(indic):,} words (names {len(resolve(name_pairs)):,}, address "
          f"{len(resolve(addr_pairs)):,}) ({time.time() - t0:.0f}s)", flush=True)
    vocab = dict(vocab)            # every S1 word, even once-seen brands: a known word is never split
    LEX_PATH.write_text(json.dumps({"indic": indic, "vocab": vocab}, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {LEX_PATH} ({LEX_PATH.stat().st_size / 2**20:.1f} MB): vocab {len(vocab):,}", flush=True)
    return indic, vocab


def main(argv=None):
    import sys
    for st in (sys.stdout, sys.stderr):                       # native-script samples on any console
        try:
            st.reconfigure(errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[3] / "student_resource" / "dataset"))
    a = ap.parse_args(argv)
    indic, vocab = build(a.data)
    rng = np.random.RandomState(0)
    keys = list(indic)
    for w in rng.choice(len(keys), min(25, len(keys)), replace=False):
        print(f"  {keys[w]} -> {indic[keys[w]]}")


if __name__ == "__main__":
    main()
