"""Synthetic dataset with the challenge's schema and noise patterns — for smoke tests only.

Never used for training the submitted model. Train covers US + India; test adds France.
    python -m src.synth --out /tmp/synth/dataset --entities 2500
Writes dataset/{train,test}/... plus dataset/_synthetic_test_ground_truth.tsv for local scoring.
"""
from __future__ import annotations

import argparse
import random
import unicodedata
from pathlib import Path

US_A = ["Sunrise", "Golden", "Blue Sky", "Pacific", "Liberty", "Eagle", "Summit", "Riverside", "Evergreen", "Maple",
        "Pioneer", "Harbor", "Silver", "Redwood", "Lakeside", "Northstar", "Cornerstone", "Heritage", "Premier",
        "Johnson", "Smith", "Miller", "Garcia", "Nguyen", "Patel", "Brown", "Davis", "Wilson", "Anderson", "Taylor",
        "Moore", "Joe's", "Bella", "Coastal", "Midwest", "Frontier", "Keystone", "Granite", "Horizon", "Apex"]
US_B = ["Dental Clinic", "Auto Repair", "Pizza", "Bakery", "Consulting", "Plumbing", "Law Office", "Insurance Agency",
        "Pharmacy", "Hardware", "Coffee House", "Fitness Center", "Realty", "Construction", "Landscaping", "Printing",
        "Pet Care", "Florist", "Cleaning Services", "Electric", "Eye Clinic", "Medical Center", "Tax Services",
        "Steakhouse", "Motors", "Technologies", "Manufacturing", "Associates", "Brothers Roofing", "Laboratories"]
US_LEGAL = ["Inc", "Inc.", "LLC", "Corp", "Corporation", "Co", "Company", "Ltd", "", "", ""]
US_ST = ["Main", "Oak", "Maple", "Washington", "Lake", "Park", "Hill", "Cedar", "Elm", "Pine", "Sunset", "Market",
         "Broadway", "Church", "Mill", "First", "Second", "Third", "Jefferson", "Lincoln", "Madison", "Highland"]
US_TYPE = [("Street", "St"), ("Avenue", "Ave"), ("Road", "Rd"), ("Boulevard", "Blvd"), ("Drive", "Dr"), ("Lane", "Ln"),
           ("Court", "Ct"), ("Highway", "Hwy")]
US_CITY = [("Springfield", "IL", "62701"), ("Austin", "TX", "73301"), ("Denver", "CO", "80202"),
           ("Portland", "OR", "97201"), ("Columbus", "OH", "43004"), ("Raleigh", "NC", "27601"),
           ("Madison", "WI", "53703"), ("Tampa", "FL", "33602"), ("Phoenix", "AZ", "85001"),
           ("Boise", "ID", "83702"), ("Albany", "NY", "12207"), ("Salem", "MA", "01970")]

IN_A = ["Shree Ganesh", "Lakshmi", "Sai", "Balaji", "Krishna", "Om", "Jai Hind", "Annapurna", "Mahalakshmi", "Durga",
        "Hanuman", "Gayatri", "Aggarwal", "Sharma", "Gupta", "Reddy", "Iyer", "Patel", "Mehta", "Singh", "Kumar",
        "Verma", "Mohammed", "Shri Ram", "Vishnu", "Saraswati", "New India", "Royal", "Janta", "Bharat"]
IN_B = ["Traders", "Textiles", "Sweets", "Medical Stores", "Electricals", "Hardware", "Jewellers", "Enterprises",
        "Motors", "Steel Works", "Agencies", "Kirana Store", "Tent House", "Tours and Travels", "Engineering Works",
        "Pharma", "Constructions", "Opticals", "Furnitures", "Mobiles", "Industries", "Brothers", "Associates",
        "Dairy", "Hospital", "Book Depot", "Automobiles", "Cloth House"]
IN_LEGAL = ["Pvt Ltd", "Pvt. Ltd.", "Private Limited", "LLP", "Ltd", "", "", ""]
IN_AREA = ["M.G. Road", "Station Road", "Gandhi Nagar", "Laxmi Nagar", "Sector 15", "Civil Lines", "Nehru Chowk",
           "Main Bazar", "Old City", "Jawahar Colony", "Ring Road", "Mall Road", "Rajendra Nagar", "Sadar Bazar"]
IN_LM = ["Near SBI ATM", "Opp. Bus Stand", "Behind Hanuman Mandir", "Near Railway Station", "Next to Post Office",
         "Opposite City Hospital", "Near Clock Tower"]
IN_CITY = [("Pune", "411001"), ("Mumbai", "400001"), ("Delhi", "110001"), ("Jaipur", "302001"),
           ("Lucknow", "226001"), ("Bengaluru", "560001"), ("Chennai", "600001"), ("Hyderabad", "500001"),
           ("Indore", "452001"), ("Nagpur", "440001")]
TRANSLIT = {"Lakshmi": ["Laxmi", "Lakshmee"], "Mahalakshmi": ["Mahalaxmi"], "Shree": ["Sri", "Shri"],
            "Shri": ["Shree", "Sri"], "Aggarwal": ["Agarwal", "Agrawal"], "Mohammed": ["Mohd", "Muhammad"],
            "Ganesh": ["Ganesha", "Ganapati"], "Krishna": ["Krishan", "Krsna"], "Saraswati": ["Sarswati"],
            "Jewellers": ["Jewelers"], "Furnitures": ["Furniture"], "Balaji": ["Balajee"], "Durga": ["Durgaa"]}

FR_A = ["Boulangerie", "Pharmacie", "Café", "Garage", "Librairie", "Fromagerie", "Boucherie", "Brasserie", "Salon",
        "Cabinet", "Atelier", "Épicerie", "Fleuriste", "Pâtisserie", "Hôtel", "Restaurant", "Crêperie", "Cave"]
FR_B = ["de l'Étoile", "du Centre", "Saint-Michel", "des Arts", "Martin", "Dubois", "Lefèvre", "du Port",
        "de la Gare", "Bernard", "Moreau", "Laurent", "Sainte-Anne", "du Marché", "des Lilas", "Beaulieu"]
FR_LEGAL = ["SARL", "SAS", "EURL", "SA", "", "", ""]
FR_TYPE = [("Rue", "R."), ("Avenue", "Av."), ("Boulevard", "Bd"), ("Place", "Pl."), ("Chemin", "Ch."),
           ("Impasse", "Imp."), ("Allée", "All."), ("Route", "Rte")]
FR_ST = ["de la Paix", "Victor Hugo", "Saint-Germain", "Jean Jaurès", "des Lilas", "du Général de Gaulle", "Pasteur",
         "de la République", "Voltaire", "Gambetta", "Sainte-Catherine", "du Faubourg Saint-Antoine"]
FR_CITY = [("Paris", "75005"), ("Lyon", "69002"), ("Marseille", "13001"), ("Toulouse", "31000"), ("Nice", "06000"),
           ("Nantes", "44000"), ("Bordeaux", "33000"), ("Lille", "59000"), ("Strasbourg", "67000")]
FR_LM = ["en face de la Poste", "près de la Mairie", "à côté de l'église"]

ABBR = {"Street": "St", "Avenue": "Ave", "Road": "Rd", "Boulevard": "Blvd", "Drive": "Dr", "Lane": "Ln",
        "Private": "Pvt", "Limited": "Ltd", "Company": "Co", "Corporation": "Corp", "Services": "Svcs",
        "Saint": "St", "Brothers": "Bros", "Associates": "Assoc", "Technologies": "Tech", "Manufacturing": "Mfg",
        "Laboratories": "Labs", "Rue": "R.", "Place": "Pl.", "Center": "Ctr", "Medical": "Med", "Highway": "Hwy",
        "Enterprises": "Ent.", "Industries": "Inds"}


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


class Gen:
    def __init__(self, seed):
        self.r = random.Random(seed)

    def typo(self, s):
        if len(s) < 4:
            return s
        i = self.r.randrange(1, len(s) - 1)
        op = self.r.random()
        if op < 0.3:
            return s[:i] + s[i + 1] + s[i] + s[i + 2:]
        if op < 0.6:
            return s[:i] + s[i + 1:]
        if op < 0.8:
            return s[:i] + s[i] + s[i:]
        return s[:i] + self.r.choice("aeiourstln") + s[i + 1:]

    def words(self, s, fn, p):
        return " ".join(fn(w) if self.r.random() < p else w for w in s.split())

    def abbrev(self, s, p=0.6):
        return self.words(s, lambda w: ABBR.get(w, w), p)

    def expand(self, s, p=0.6):
        inv = {v: k for k, v in ABBR.items()}
        return self.words(s, lambda w: inv.get(w, w), p)

    def translit(self, s, p=0.7):
        return self.words(s, lambda w: self.r.choice(TRANSLIT[w]) if w in TRANSLIT else w, p)

    def case(self, s):
        x = self.r.random()
        return s.upper() if x < 0.15 else (s.lower() if x < 0.25 else s)


def make_entity(g, cty, chain_name=None):
    r = g.r
    if cty == "US":
        core = chain_name or f"{r.choice(US_A)} {r.choice(US_B)}"
        legal = r.choice(US_LEGAL)
        name = f"{core} {legal}".strip()
        st, (tf, _) = r.choice(US_ST), r.choice(US_TYPE)
        city, state, z = r.choice(US_CITY)
        unit = f", Suite {r.randint(1, 400)}" if r.random() < 0.2 else ""
        addr = {"num": str(r.randint(1, 9999)), "street": f"{st} {tf}", "unit": unit, "city": city,
                "state": state, "postal": z, "lm": ""}
    elif cty == "India":
        core = chain_name or f"{r.choice(IN_A)} {r.choice(IN_B)}"
        name = f"{core} {r.choice(IN_LEGAL)}".strip()
        city, pin = r.choice(IN_CITY)
        addr = {"num": f"Shop No. {r.randint(1, 250)}", "street": r.choice(IN_AREA), "unit": "", "city": city,
                "state": "", "postal": pin, "lm": r.choice(IN_LM) if r.random() < 0.5 else ""}
    else:
        core = chain_name or f"{r.choice(FR_A)} {r.choice(FR_B)}"
        lg = r.choice(FR_LEGAL)
        name = (f"{lg} {core}" if r.random() < 0.5 else f"{core} {lg}").strip()
        (tf, _), st = r.choice(FR_TYPE), r.choice(FR_ST)
        city, cp = r.choice(FR_CITY)
        addr = {"num": str(r.randint(1, 180)), "street": f"{tf} {st}", "unit": "", "city": city, "state": "",
                "postal": cp, "lm": r.choice(FR_LM) if r.random() < 0.2 else ""}
    dba = None
    if r.random() < 0.08:
        dba = f"{r.choice(US_A if cty != 'France' else FR_B)} {r.choice(US_B if cty != 'France' else FR_A)}"
    return {"cty": cty, "core": core, "name": name, "addr": addr, "dba": dba}


def render(g, e, noise):
    r = g.r
    name = e["name"]
    if e["dba"] and r.random() < 0.5:
        name = f"{name} dba {e['dba']}" if r.random() < 0.5 else e["dba"]
    a = dict(e["addr"])
    if noise:
        if r.random() < 0.5:
            name = g.abbrev(name) if r.random() < 0.6 else g.expand(name)
        if r.random() < 0.3:                                  # legal suffix inconsistencies
            toks = name.split()
            if toks and toks[-1].strip(".").lower() in {"inc", "llc", "corp", "co", "ltd", "llp", "sarl", "sas", "sa"}:
                name = " ".join(toks[:-1])
            else:
                name = name + " " + r.choice(["Inc", "LLC", "Pvt Ltd", "Ltd"] if e["cty"] != "France" else ["SARL", "SAS"])
        if e["cty"] == "India":
            name = g.translit(name, 0.6)
        if r.random() < 0.25:
            name = g.typo(name)
        if r.random() < 0.1:
            t = name.split()
            if len(t) >= 3:
                i = r.randrange(len(t) - 1)
                t[i], t[i + 1] = t[i + 1], t[i]
                name = " ".join(t)
        if r.random() < 0.15:
            name = name.replace(" and ", " & ") if " and " in name else name.replace("&", "and")
        if e["cty"] == "France" and r.random() < 0.5:
            name = strip_accents(name)
        name = g.case(name)
        if r.random() < 0.4:
            a["street"] = g.abbrev(a["street"]) if r.random() < 0.7 else g.expand(a["street"])
        if r.random() < 0.3:
            a["postal"] = ""
        if r.random() < 0.2:
            a["state"] = ""
        if r.random() < 0.15:
            a["city"] = ""
        if r.random() < 0.15:
            a["unit"] = ""
        if e["cty"] == "India":
            if r.random() < 0.4:
                a["lm"] = r.choice(IN_LM) if not a["lm"] else ("" if r.random() < 0.5 else a["lm"])
            if r.random() < 0.3:
                a["num"] = a["num"].replace("Shop No. ", r.choice(["#", "No.", "Shop ", ""]))
            if a["postal"] and r.random() < 0.2:
                a["postal"] = a["postal"][:3] + " " + a["postal"][3:]
            a["street"] = g.translit(a["street"], 0.5)
        if r.random() < 0.08:
            a["num"] = g.typo(a["num"]) if len(a["num"]) > 3 else str(int(r.randint(1, 999)))
        if e["cty"] == "France" and r.random() < 0.5:
            a["street"] = strip_accents(a["street"])
        if r.random() < 0.1:
            a["street"] = g.typo(a["street"])
    parts = [f"{a['num']} {a['street']}".strip() + a["unit"]]
    if a["lm"]:
        parts.insert(1 if r.random() < 0.5 else 0, a["lm"])
    tail = " ".join(x for x in (a["city"], a["state"], a["postal"]) if x)
    if e["cty"] == "France":
        tail = " ".join(x for x in (a["postal"], a["city"]) if x)
    if tail:
        parts.append(tail)
    if noise and r.random() < 0.1:
        r.shuffle(parts)
    addr = ", ".join(parts)
    if noise and r.random() < 0.02:
        addr = ""
    return name, addr


def make_split(g, countries, n_ent, split, id_ctr):
    r = g.r
    ents = []
    for cty in countries:
        chains = [make_entity(g, cty)["core"] for _ in range(max(3, n_ent // 60))]
        for _ in range(n_ent):
            ents.append(make_entity(g, cty, r.choice(chains) if r.random() < 0.12 else None))
    rows = {1: [], 2: [], 3: []}
    gold = []

    def nid(src):
        id_ctr[src] += 1
        return f"S{src}-{id_ctr[src]:07d}"

    for e in ents:
        in_s1 = r.random() < 0.75
        k = r.choices([0, 1, 2, 3, 4, 5], [0.33, 0.35, 0.18, 0.08, 0.04, 0.02])[0]
        if not in_s1:
            k = r.choice([1, 1, 2])                           # pool-only distractor business
        recs = []
        for _ in range(k):
            src = r.choice([2, 3])
            i = nid(src)
            n, a = render(g, e, noise=True)
            rows[src].append((i, n, a, e["cty"]))
            recs.append(i)
        if in_s1:
            i = nid(1)
            n, a = render(g, e, noise=r.random() < 0.3)
            rows[1].append((i, n, a, e["cty"]))
            gold.append((i, recs))
    for k in rows:
        r.shuffle(rows[k])
    return rows, gold


def write(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(x.replace("\t", " ") for x in row) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--entities", type=int, default=2500, help="businesses per country per split")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    g = Gen(a.seed)
    out = Path(a.out)
    ctr = {1: 0, 2: 0, 3: 0}
    hdr = ["entity_id", "business_name", "business_address", "country"]
    for split, countries, n in (("train", ["US", "India"], a.entities), ("test", ["US", "India", "France"],
                                                                        max(a.entities // 2, 50))):
        rows, gold = make_split(g, countries, n, split, ctr)
        for s in (1, 2, 3):
            write(out / split / f"{split}_source{s}.tsv", hdr, rows[s])
        gpath = out / split / f"{split}_ground_truth.tsv" if split == "train" else out / "_synthetic_test_ground_truth.tsv"
        write(gpath, ["source1_entity_id", "matched_entity_ids"], [(s, ",".join(m)) for s, m in gold])
        print(split, {s: len(rows[s]) for s in rows}, "gold rows", len(gold))


if __name__ == "__main__":
    main()
