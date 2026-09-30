"""Country-agnostic record normalization (plan section 4, appendix A.4).

Standard library + rapidfuzz only: no gazetteers, address parsers or external data.
Every lexicon below is generic linguistic knowledge (abbreviations, legal forms,
landmark cues, stopwords, state abbreviations) and is listed in the documentation. The one learned lexicon,
src/lexicon.json (src/lexicon.py), is built from the provided data only: native-script words mapped to the
English words they translate (from train gold pairs) and the vocabulary of Source-1 names.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from rapidfuzz.distance import JaroWinkler

from .compat import effective_cpus

_LIG = str.maketrans({"œ": "oe", "æ": "ae", "ß": "ss", "ø": "o", "ł": "l", "đ": "d", "ı": "i", "þ": "th"})
_APOS = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'", "ʼ": "'"})


# ---- Indic scripts -> Latin (hand-written phonetic table; no external data) --------------------------
# Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada and Malayalam share the ISCII
# layout: the same sound sits at the same offset inside each 128-code-point block, so one table keyed by
# offset covers all nine. Consonants carry an inherent 'a' that a virama or a vowel sign replaces; the
# word-final inherent 'a' is dropped (schwa deletion), so प्राइवेट -> "praivet" (skeleton prvt = private).
_INDIC_BASES = (0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00)
# Long vowels are written short (aa -> a) and pha as f: business names are mostly English words written in an
# Indic script, and their Latin spellings use single vowels ("praaivet" -> "praivet", "phuud" -> "fud").
_IV = {0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri", 0x0C: "li", 0x0D: "e",
       0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "li"}
_IC = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng", 0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh",
       0x1E: "ny", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d",
       0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "f", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y",
       0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "zh", 0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s",
       0x39: "h", 0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y"}
_IM = {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri", 0x45: "e", 0x46: "e",
       0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au", 0x62: "li", 0x63: "li"}
_IX = {0x01: "n", 0x02: "n", 0x03: "h", 0x70: "n", 0x4E: "t", 0x7A: "n", 0x7B: "n", 0x7C: "r", 0x7D: "l",
       0x7E: "l", 0x7F: "k"}                                   # anusvara, visarga, tippi, khanda-ta, chillus
_OVERRIDE = {(0x0D00, 0x31): "t",                              # Malayalam rra: English t (limiRRaD = limited)
             (0x0B80, 0x1A): "s"}                              # Tamil ca: English s in loanwords (seven)
_VIRAMA = 0x4D


def _indic_block(cp):
    for b in _INDIC_BASES:
        if b <= cp < b + 0x80:
            return b, cp - b
    return None, None


def translit_indic(s: str) -> str:
    """Latin transliteration of any Indic-script characters in s; other characters pass through."""
    if not s or max(s) < "ऀ":
        return s
    out = []
    pending = False            # the last consonant still carries its inherent 'a'
    cluster = False            # ... and it closed a conjunct (virama + consonant), e.g. the ya of Aditya
    after_virama = False
    last = ""

    def word_end():            # schwa deletion, except conjunct-final y/r/v: Aditya, Surya, Chandra
        if pending and cluster and last in ("y", "r", "v"):
            out.append("a")

    for ch in s:
        cp = ord(ch)
        base, off = _indic_block(cp) if 0x0900 <= cp < 0x0D80 else (None, None)
        if off is None:
            if cp in (0x200C, 0x200D):                        # zero-width (non-)joiner
                continue
            if pending:
                if ch.isalpha():
                    out.append("a")
                else:
                    word_end()
            pending = cluster = after_virama = False
            out.append(ch)
            continue
        if off in _IC:
            if pending:
                out.append("a")
            last = _OVERRIDE.get((base, off), _IC[off])
            out.append(last)
            cluster, pending, after_virama = after_virama, True, False
        elif off in _IM:
            out.append(_IM[off])
            pending = after_virama = False
        elif off == _VIRAMA:
            pending, after_virama = False, True
        elif off in _IV:
            if pending:
                out.append("a")
            out.append(_IV[off])
            pending = after_virama = False
        elif off in _IX:
            if pending and off in (0x01, 0x02, 0x70):
                out.append("a")
            out.append(_IX[off])
            pending = after_virama = False
        elif 0x66 <= off <= 0x6F:                              # Indic digits
            word_end()
            out.append(str(off - 0x66))
            pending = after_virama = False
    word_end()
    return "".join(out)


def _load_lexicon():
    p = Path(__file__).with_name("lexicon.json")
    if not p.exists():
        return {}, {}, 1, "none"
    raw = p.read_bytes()
    d = json.loads(raw.decode("utf-8"))
    vocab = d.get("vocab", {})
    return d.get("indic", {}), vocab, max(sum(vocab.values()), 1), hashlib.sha1(raw).hexdigest()[:12]


LEX_INDIC, VOCAB, VOCAB_TOTAL, LEX_HASH = _load_lexicon()
_NATIVE_WORD = re.compile(r"[^\s,;:|()\[\]/.\-]+")


def fold(s: str) -> str:
    """Indic -> Latin (learned word translations first, phonetic transliteration for the rest), NFKC ->
    lowercase -> ligatures -> strip accents."""
    s = s or ""
    if LEX_INDIC and has_indic(s):
        s = _NATIVE_WORD.sub(lambda m: LEX_INDIC.get(m.group(), m.group()), s)
    s = translit_indic(s)
    s = unicodedata.normalize("NFKC", s).lower().translate(_LIG).translate(_APOS)
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


# Canonical SHORT forms: 'street' and 'saint' both become 'st', so an ambiguous 'St'
# matches either reading without guessing which one was meant.
CANON = {
    "street": "st", "str": "st", "saint": "st", "sainte": "ste", "suite": "ste",
    "road": "rd", "avenue": "av", "ave": "av", "boulevard": "bd", "blvd": "bd", "bld": "bd", "boul": "bd",
    "drive": "dr", "doctor": "dr", "lane": "ln", "court": "ct", "place": "pl", "square": "sq",
    "highway": "hwy", "parkway": "pkwy", "expressway": "expy", "freeway": "fwy", "terrace": "ter",
    "circle": "cir", "crescent": "cres", "junction": "jct", "trail": "trl", "turnpike": "tpke",
    "building": "bldg", "bldng": "bldg", "floor": "fl", "flr": "fl", "tower": "twr", "towers": "twr",
    "apartment": "apt", "apartments": "apt", "apts": "apt", "appartement": "apt", "complex": "cplx",
    "plaza": "plz", "heights": "hts", "opposite": "opp", "nr": "near",
    "north": "n", "south": "s", "east": "e", "west": "w", "nord": "n", "sud": "s", "est": "e", "ouest": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "mount": "mt", "fort": "ft", "centre": "ctr", "center": "ctr", "market": "mkt", "nagar": "ngr",
    "colony": "col", "sector": "sec", "station": "stn", "hospital": "hosp", "district": "dist",
    "bazaar": "bzr", "bazar": "bzr", "chowk": "chk", "main": "mn",
    "chemin": "ch", "impasse": "imp", "faubourg": "fbg", "route": "rte", "batiment": "bat", "allee": "all",
    "residence": "res", "quartier": "qtr",
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
    "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10", "premier": "1", "premiere": "1",
    "international": "intl", "manufacturing": "mfg", "associates": "assoc", "association": "assoc",
    "brothers": "bros", "services": "svc", "service": "svc", "svcs": "svc", "management": "mgmt",
    "technologies": "tech", "technology": "tech", "industries": "ind", "industry": "ind",
    "enterprises": "ent", "enterprise": "ent", "department": "dept", "government": "govt",
    "solutions": "soln", "solution": "soln", "systems": "sys", "system": "sys", "products": "prod",
    "laboratories": "labs", "laboratory": "labs", "lab": "labs", "pharmaceuticals": "pharma",
    "engineering": "engg", "engineers": "engrs", "consultants": "consult", "consulting": "consult",
    "restaurant": "rest", "restaurants": "rest", "clinic": "clin", "medical": "med",
    "shree": "sri", "shri": "sri", "sree": "sri", "shreee": "sri",
    "private": "pvt", "limited": "ltd", "corporation": "corp", "incorporated": "inc",
    "company": "co", "compagnie": "cie", "etablissements": "ets", "etablissement": "ets", "societe": "ste",
}
DROP = {"the", "and", "et", "und", "of", "no", "number", "num", "le", "la", "les", "de", "du", "des",
        "au", "aux", "en"}
LEGAL = sorted([("pvt", "ltd"), ("pty", "ltd"), ("pte", "ltd"), ("sdn", "bhd"), ("ltd",), ("pvt",),
                ("llc",), ("inc",), ("corp",), ("co",), ("llp",), ("lp",), ("plc",), ("pllc",), ("opc",),
                ("pc",), ("ltd", "co"), ("co", "ltd"), ("corp", "ltd"),
                ("sarl",), ("sas",), ("sasu",), ("sa",), ("eurl",), ("snc",), ("sci",), ("scop",),
                ("selarl",), ("selas",), ("gie",), ("cie",), ("ets",), ("ste",), ("gmbh",), ("ag",), ("bv",),
                ("nv",), ("srl",), ("spa",), ("bhd",)], key=len, reverse=True)
LEADING_LEGAL = {"sarl", "sas", "sasu", "eurl", "ste", "ets", "societe", "sci", "snc", "selarl", "scop"}
DBA = re.compile(r"\b(?:d\W{0,2}b\W{0,2}a|doing business as|t\s*/\s*a|trading as|"
                 r"a\W{0,2}k\W{0,2}a|f\W{0,2}k\W{0,2}a|formerly|operating as)\b\.?")
LANDMARK = re.compile(r"\b(?:near|nr|opp|opposite|behind|beside|besides|next to|adjacent to|adj to|"
                      r"in front of|infront of|above|below|facing|close to|landmark|"
                      r"pres de|a cote de|en face de|face a|derriere|proche de)\b")
_ORD = re.compile(r"\b(\d+)(?:st|nd|rd|th|er|ere|eme|e)\b")   # 1st / 3rd / 1er / 5eme -> number
_PIN = re.compile(r"(?<!\w)(\d{3})[ -](\d{3})(?!\w)")          # 411 001 -> 411001 (Indian PIN format)
# digit-for-letter typos inside words (hospita1ity, ran0ix, f1int)
_LEET = re.compile(r"(?<=[A-Za-z])[0134578](?=[A-Za-z])")
_LEET_MAP = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"}
# website / handle forms of a name (olaniqpetcare.com, www.x.in, @brand, #brand)
_WEB = re.compile(r"(?i)\bwww\.|\.(?:com|net|org|in|co|biz|info|fr|us|io)\b|(?:^|(?<=\s))[@#]+")
# zero-padded house numbers (00218 1st St, No-0278, Gali No-007): at the start or after a number cue
_ZPAD_START = re.compile(r"^(\W*)0+(?=\d)")
_ZPAD_AFTER = re.compile(r"(\b(?:no|nos|hno|plot|door|shop|gali|flat|unit|ste|suite|apt|bldg|sr|survey|dno)\b"
                         r"\W{0,3}|#+\W{0,2})0+(?=\d)")
# state names -> the postal abbreviations the sources also use (folded, lowercase)
STATE_ABBR = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
    "uttar pradesh": "up", "madhya pradesh": "mp", "andhra pradesh": "ap", "himachal pradesh": "hp",
    "arunachal pradesh": "arp", "tamil nadu": "tn", "west bengal": "wb", "maharashtra": "mh", "karnataka": "ka",
    "gujarat": "gj", "rajasthan": "rj", "kerala": "kl", "telangana": "tg", "haryana": "hr", "punjab": "pb",
    "bihar": "br", "odisha": "od", "orissa": "od", "jharkhand": "jh", "chhattisgarh": "cg", "chattisgarh": "cg",
    "uttarakhand": "uk", "uttaranchal": "uk"}
_STATE_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, STATE_ABBR), key=len, reverse=True)) + r")\b")


def has_indic(s: str) -> bool:
    return bool(s) and max(s) >= "ऀ" and any("ऀ" <= ch < "඀" for ch in s)


@lru_cache(maxsize=1 << 18)
def segment(tok: str):
    """Split a concatenated name token into Source-1-name vocabulary words (olaniqpetcare -> olaniq pet care).
    Minimum-cost split (word cost = -log frequency + a per-word penalty); None unless >= 2 known words."""
    n = len(tok)
    if not VOCAB or n < 6:
        return None
    inf = float("inf")
    best, back = [0.0] + [inf] * n, [0] * (n + 1)
    lt = math.log(VOCAB_TOTAL)
    for i in range(2, n + 1):
        for j in range(max(0, i - 20), i - 1):                 # words of >= 2 letters
            c = VOCAB.get(tok[j:i])
            if c and best[j] < inf:
                v = best[j] + lt - math.log(c) + 3.0
                if v < best[i]:
                    best[i], back[i] = v, j
    if best[n] == inf:
        return None
    out, i = [], n
    while i > 0:
        out.append(tok[back[i]:i])
        i = back[i]
    out = out[::-1]
    # every piece >= 3 letters except a leading pair of initials (tk federation); truncated domains
    # (frontierm -> front ie rm) stay whole instead
    if len(out) < 2 or any(len(w) < 3 for w in out[1:]):
        return None
    return out


def _split_concat(words, only):
    """Split the website / handle tokens in `only` that are not themselves a known Source-1 word."""
    out = []
    for t in words:
        if t in only and len(t) >= 6 and t.isalpha() and t not in VOCAB:
            parts = segment(t) or (segment(t[:-3]) if t.endswith("com") and len(t) > 9 else None)
            if parts:
                out.extend(parts)
                continue
        out.append(t)
    return out


_DOMAIN = re.compile(r"(?i)([^\s|()\[\]@#]+)\.(?:com|net|org|in|co|biz|info|fr|us|io)\b|(?:^|(?<=\s))[@#]+(\S+)")


def clean_name(raw: str):
    """Digit-for-letter typos repaired and website / handle markers removed. Returns (text, website / handle
    tokens) - only those tokens may be split into words later (plain one-word brands never are)."""
    s = _LEET.sub(lambda m: _LEET_MAP[m.group()], raw or "")
    doms = frozenset(re.sub(r"[^a-z0-9]+", "", fold((a or b).split(".")[-1] if a else b))
                     for a, b in _DOMAIN.findall(s))
    return (_WEB.sub(" ", s) if doms else s), doms


def clean_address(raw: str) -> str:
    """Folded address with Indian PINs joined, zero-padded house numbers unpadded and state names abbreviated."""
    f = _PIN.sub(r"\1\2", fold(raw))
    f = _ZPAD_AFTER.sub(r"\1", _ZPAD_START.sub(r"\1", f))
    return _STATE_RE.sub(lambda m: STATE_ABBR[m.group(1)], f)


def _tokens(s: str, translit=None, seg=frozenset()):
    """translit: the text was written in an Indic script, so its tokens are phonetic spellings of (mostly
    English) words: they are canonicalised by phonetic skeleton too ('praivet' -> pvt, 'sarvisis' -> svc).
    seg: website / handle tokens of the text; each is split into words with the name vocabulary."""
    if translit is None:
        translit = has_indic(s)
    s = fold(s)
    s = re.sub(r"(?<=\w)'s\b", "", s)                          # joe's -> joe
    s = re.sub(r"\b(?:[ldjmntcs]|qu)'(?=\w)", "", s)           # l'atelier -> atelier
    s = s.replace("'", "").replace("&", " and ").replace("+", " and ")
    s = re.sub(r"\b((?:[a-z]\.){2,})", lambda m: m.group(1).replace(".", ""), s)  # m.g. -> mg
    s = _ORD.sub(r"\1", s)
    s = re.sub(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)", " ", s)   # 12bis -> 12 bis
    s = re.sub(r"[^a-z0-9]+", " ", s)
    words = _split_concat(s.split(), seg) if seg else s.split()
    out = []
    for t in words:
        c = CANON.get(t)
        if c is None and translit and not t.isdigit():
            c = TR_CANON.get(t) or SKEL_CANON.get(skeleton(t))
        c = c or t
        if c not in DROP:
            out.extend(c.split())
    return out


def split_legal(toks):
    legal, changed = [], True
    while changed and len(toks) > 1:
        changed = False
        for lf in LEGAL:
            if len(toks) > len(lf) and tuple(toks[-len(lf):]) == lf:
                legal.append(" ".join(lf))
                toks = toks[:-len(lf)]
                changed = True
                break
    while len(toks) > 2 and toks[0] in LEADING_LEGAL:          # keep >= 2 core tokens
        legal.append(toks[0])
        toks = toks[1:]
    return toks, legal


def normalize_name(raw: str):
    """core = legal-form-free tokens of every DBA/parenthesised piece; variants = the pieces."""
    raw, seg = clean_name(raw)
    tr = has_indic(raw)
    pieces = [p for p in re.split(r"[|()\[\]]", DBA.sub(" | ", fold(raw))) if p.strip()]
    core, legal, variants = [], [], []
    for p in pieces:
        t, lg = split_legal(_tokens(p, tr, seg))
        legal += lg
        if t:
            variants.append(t)
            core += [x for x in t if x not in core]
    return {"core": core, "legal": sorted(set(legal)), "variants": variants if len(variants) > 1 else []}


def normalize_address(raw: str):
    f = clean_address(raw)
    core, landmark = [], []
    for seg in re.split(r"[,;\n]", f):
        m = LANDMARK.search(seg)
        if m:
            core.append(seg[:m.start()])
            landmark.append(seg[m.end():])
        else:
            core.append(seg)
    tr = has_indic(raw)
    core_toks = _tokens(" ".join(core), tr)
    postal = [n for n in re.findall(r"\d+", f) if len(n) >= 5]
    nums = [t.lstrip("0") or "0" for t in core_toks if t.isdigit() and len(t) < 5]
    return {"core": core_toks, "landmark": _tokens(" ".join(landmark), tr),
            "numbers": list(dict.fromkeys(nums)), "postal": list(dict.fromkeys(postal))}


def is_abbrev(a: str, b: str) -> bool:
    """rd~road, pvt~private, bd~boulevard, fbg~faubourg; rejects st~south, rd~ridge."""
    if len(a) < 2 or len(a) >= len(b) or a[0] != b[0] or a.isdigit() or b.isdigit():
        return False
    it = iter(b)
    if not all(ch in it for ch in a):                          # subsequence test
        return False
    return b.startswith(a) or a[-1] == b[-1] or len(a) >= 3


_SOFT_C = re.compile(r"c(?=[eiy])")
_SOFT_G = re.compile(r"g(?=[eiy])")
_H_AFTER = re.compile(r"([cdfgjklmnprstv])h")
_DOUBLE = re.compile(r"(.)\1+")
_VOWELS = re.compile(r"[aeiou]")


@lru_cache(maxsize=1 << 20)
def skeleton(tok: str) -> str:
    """Phonetic consonant key: lakshmi/laxmi -> lksm, aggarwal/agrawal -> agrvl, shree/sri -> sr,
    services/sarvisis -> srvs, private/praivet/praibhet -> prvt, software/sofatver -> sftvr."""
    t = re.sub(r"[^a-z]", "", tok)
    if not t:
        return ""
    t = _SOFT_G.sub("j", _SOFT_C.sub("s", t))                   # city -> sity, technologies -> teknolojies
    for x, y in (("ph", "f"), ("ck", "k"), ("sh", "s"), ("q", "k"), ("x", "ks"), ("z", "s"), ("w", "v"),
                 ("b", "v"), ("y", "i")):
        t = t.replace(x, y)
    t = _H_AFTER.sub(r"\1", t).replace("c", "k")
    t = _DOUBLE.sub(r"\1", t)
    return _DOUBLE.sub(r"\1", t[0] + _VOWELS.sub("", t[1:]))


# Transliterated (Indic-script) tokens are canonicalised through their skeleton, so any phonetic spelling of
# a CANON word maps to the same short form. Ambiguous skeletons (two CANON values) are left out.
TR_CANON = {"pra": "pvt", "li": "ltd", "limitet": "ltd", "elelpi": "llp", "aiti": "it", "kampani": "co"}


def _build_skel_canon():
    m, bad = {}, set()
    for k, v in list(CANON.items()) + [("private", "pvt"), ("limited", "ltd")]:
        sk = skeleton(k)
        if len(k) < 5 or len(sk) < 3:
            continue
        if sk in m and m[sk] != v:
            bad.add(sk)
        m[sk] = v
    for sk in bad:
        m.pop(sk, None)
    return m


SKEL_CANON = _build_skel_canon()


@lru_cache(maxsize=1 << 21)
def tok_sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if is_abbrev(a, b) or is_abbrev(b, a):
        return 0.9
    if len(a) == 1 or len(b) == 1:                             # initials: 'r' ~ 'rue'
        return 0.5 if a[0] == b[0] and not (a.isdigit() or b.isdigit()) else 0.0
    if a.isdigit() or b.isdigit():
        return 0.0
    if min(len(a), len(b)) >= 4:
        jw = JaroWinkler.similarity(a, b)
        if jw >= 0.9:
            return jw
        if len(a) >= 5 and skeleton(a) == skeleton(b):
            return 0.85
    return 0.0


def soft_align(A, B, idf, default_idf):
    """Greedy one-to-one, IDF-weighted, abbreviation/typo-aware token alignment.
    Returns (soft_dice, cov_a, cov_b, max_unmatched_idf_a, max_unmatched_idf_b,
             shared_idf_max, shared_idf_min, idf_sum_a, idf_sum_b)."""
    if not A or not B:
        wa = sum(idf.get(t, default_idf) for t in A)
        wb = sum(idf.get(t, default_idf) for t in B)
        return (0.0, 0.0, 0.0, max([idf.get(t, default_idf) for t in A], default=0.0),
                max([idf.get(t, default_idf) for t in B], default=0.0), 0.0, 0.0, wa, wb)
    wA = [idf.get(t, default_idf) for t in A]
    wB = [idf.get(t, default_idf) for t in B]
    cand = []
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            s = tok_sim(a, b)
            if s > 0:
                cand.append((s * (wA[i] + wB[j]) / 2, s, i, j))
    cand.sort(reverse=True)
    ua, ub, num_a, num_b, shared = set(), set(), 0.0, 0.0, []
    for _, s, i, j in cand:
        if i in ua or j in ub:
            continue
        ua.add(i)
        ub.add(j)
        num_a += s * wA[i]
        num_b += s * wB[j]
        if s >= 0.85:
            shared.append(min(wA[i], wB[j]))
    WA, WB = sum(wA) or 1e-9, sum(wB) or 1e-9
    return ((num_a + num_b) / (WA + WB), num_a / WA, num_b / WB,
            max([wA[i] for i in range(len(A)) if i not in ua], default=0.0),
            max([wB[j] for j in range(len(B)) if j not in ub], default=0.0),
            max(shared, default=0.0), min(shared, default=0.0), WA, WB)


def acronym_match(A, B) -> bool:
    """State Bank of India ~ SBI (DROP already removed 'of')."""
    ia = "".join(t[0] for t in A if not t.isdigit())
    ib = "".join(t[0] for t in B if not t.isdigit())
    return bool((len(A) >= 2 and len(ia) >= 2 and (ia in B or ia == "".join(B))) or
                (len(B) >= 2 and len(ib) >= 2 and (ib in A or ib == "".join(A))))


def cty_key(c: str) -> str:
    k = " ".join(fold(c).split())
    return k or "__na__"


NORM_COLS = ["n_core", "n_full", "n_legal", "n_vars", "n_skel", "n_nums",
             "a_core", "a_full", "a_lm", "a_nums", "a_postal"]


def normalize_record(name: str, addr: str):
    nn = normalize_name(name)
    na = normalize_address(addr)
    core = nn["core"]
    skel = [skeleton(t) for t in core]
    cn, seg = clean_name(name)
    return (" ".join(core),
            " ".join(_tokens(cn, None, seg)),
            "|".join(nn["legal"]),
            "|".join(" ".join(v) for v in nn["variants"]),
            " ".join(s for s in skel if s),
            " ".join(t for t in core if t.isdigit()),
            " ".join(na["core"]),
            " ".join(_tokens(clean_address(addr))),
            " ".join(na["landmark"]),
            " ".join(na["numbers"]),
            " ".join(na["postal"]))


def _norm_chunk(names, addrs):
    return [normalize_record(n, a) for n, a in zip(names, addrs)]


def normalize_frame(df: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    """One row per record: id, source, partition key and the normalized fields in NORM_COLS."""
    names = df["business_name"].tolist()
    addrs = df["business_address"].tolist()
    n = len(df)
    workers = effective_cpus() if n_jobs in (-1, None) else n_jobs
    if n > 20000 and workers > 1:
        step = max(5000, n // (workers * 4) + 1)
        parts = Parallel(n_jobs=workers)(delayed(_norm_chunk)(names[i:i + step], addrs[i:i + step])
                                         for i in range(0, n, step))
        rows = [r for p in parts for r in p]
    else:
        rows = _norm_chunk(names, addrs)
    out = pd.DataFrame(rows, columns=NORM_COLS)
    out.insert(0, "id", df["entity_id"].to_numpy())
    out.insert(1, "src", df["entity_id"].str[:2].to_numpy())
    out.insert(2, "cty", [cty_key(c) for c in df["country"]])
    out["raw_name"] = df["business_name"].to_numpy()
    out["raw_addr"] = df["business_address"].to_numpy()
    return out
