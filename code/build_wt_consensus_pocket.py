#!/usr/bin/env python3
"""
Build a WT-consensus pocket residue list from AF3 seed-* CIFs.

For every WT structure under <root>/seed-*/*.cif, find residues with any atom
within --pocket-cutoff of the ligand. Tally per-position occupancy (chain,
resseq, icode — resname is dropped so the consensus ports to mutants where the
residue name differs at the mutated positions). Residues occupied in at least
--frequency fraction of structures form the consensus pocket.

Output:
  <out-prefix>.tsv   columns: chain, resseq, icode, wt_resname, occupancy,
                              n_observed, n_total, in_consensus
  <out-prefix>.json  same content as a list of dicts; convenient for the
                     downstream SASA-with-fixed-pocket analyzer

Example (run from the deposit root; these are the parameters used for the
deposited results/wt_consensus_pocket_ETV6_ETS_DLPA.json):
  python code/build_wt_consensus_pocket.py \\
      af3_outputs/ETV6_WT_ETS_DLPA \\
      --ligand LIG_B \\
      --pocket-cutoff 4.5 \\
      --frequency 0.8 \\
      --out-prefix results/wt_consensus_pocket_ETV6_ETS_DLPA
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import is_aa


def atom_dist(a, b):
    ax, ay, az = a.get_coord()
    bx, by, bz = b.get_coord()
    dx, dy, dz = ax - bx, ay - by, az - bz
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def find_cif_files(root):
    root = Path(root)
    cifs = []
    for seed_dir in root.glob("seed-*"):
        if seed_dir.is_dir():
            cifs.extend(seed_dir.rglob("*.cif"))
    return sorted(cifs)


def get_structure_components(cif_path, ligand_name, chain_filter=None):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(cif_path.stem, str(cif_path))
    ligand_residues = []
    protein_residues = []
    for model in structure:
        for chain in model:
            if chain_filter and chain.id != chain_filter:
                continue
            for residue in chain:
                resname = residue.get_resname().strip()
                hetflag = residue.get_id()[0]
                if resname == ligand_name:
                    ligand_residues.append(residue)
                    continue
                if hetflag.startswith("W"):
                    continue
                if is_aa(residue, standard=False):
                    protein_residues.append(residue)
    return structure, ligand_residues, protein_residues


def find_pocket_residues(ligand_residues, protein_residues, pocket_cutoff):
    pocket = {}
    for pres in protein_residues:
        hit = False
        for patom in pres.get_atoms():
            for lig in ligand_residues:
                for latom in lig.get_atoms():
                    if atom_dist(patom, latom) <= pocket_cutoff:
                        chain = pres.get_parent().id
                        resseq, icode = pres.get_id()[1], pres.get_id()[2]
                        icode = "" if icode in (None, " ") else str(icode)
                        pocket[(chain, int(resseq), icode, pres.get_resname().strip())] = pres
                        hit = True
                        break
                if hit:
                    break
            if hit:
                break
    return pocket


def position_key(residue):
    chain = residue.get_parent().id
    resseq, icode = residue.get_id()[1], residue.get_id()[2]
    icode = "" if icode in (None, " ") else str(icode)
    return (chain, int(resseq), icode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="WT run dir containing seed-* subdirs")
    ap.add_argument("--ligand", required=True, help="Ligand residue name in CIF, e.g. LIG")
    ap.add_argument("--chain", default=None, help="Optional protein chain filter")
    ap.add_argument("--pocket-cutoff", type=float, default=4.5)
    ap.add_argument(
        "--frequency",
        type=float,
        default=0.8,
        help="Minimum occupancy fraction to enter consensus (default 0.8)",
    )
    ap.add_argument(
        "--out-prefix",
        default="wt_consensus_pocket",
        help="Output file prefix (writes .tsv and .json)",
    )
    args = ap.parse_args()

    cifs = find_cif_files(args.root)
    if not cifs:
        print(f"No CIF files under {args.root}/seed-*/")
        return

    occupancy = Counter()
    resnames_seen = defaultdict(Counter)
    n_total = 0

    for cif in cifs:
        _, lig_res, prot_res = get_structure_components(
            cif, args.ligand, chain_filter=args.chain
        )
        if not lig_res or not prot_res:
            continue
        pocket = find_pocket_residues(lig_res, prot_res, args.pocket_cutoff)
        n_total += 1
        for res in pocket.values():
            pk = position_key(res)
            occupancy[pk] += 1
            resnames_seen[pk][res.get_resname().strip()] += 1

    if n_total == 0:
        print("No structures yielded both ligand and protein residues.")
        return

    rows = []
    for pk, n_obs in sorted(occupancy.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        chain, resseq, icode = pk
        wt_resname = resnames_seen[pk].most_common(1)[0][0]
        occ = n_obs / n_total
        rows.append({
            "chain": chain,
            "resseq": resseq,
            "icode": icode,
            "wt_resname": wt_resname,
            "occupancy": occ,
            "n_observed": n_obs,
            "n_total": n_total,
            "in_consensus": occ >= args.frequency,
        })

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    tsv_path = str(out_prefix) + ".tsv"
    with open(tsv_path, "w") as fh:
        fh.write("chain\tresseq\ticode\twt_resname\toccupancy\tn_observed\tn_total\tin_consensus\n")
        for r in rows:
            fh.write(
                f"{r['chain']}\t{r['resseq']}\t{r['icode']}\t{r['wt_resname']}\t"
                f"{r['occupancy']:.3f}\t{r['n_observed']}\t{r['n_total']}\t"
                f"{int(r['in_consensus'])}\n"
            )

    json_path = str(out_prefix) + ".json"
    with open(json_path, "w") as fh:
        json.dump({
            "root": str(args.root),
            "ligand": args.ligand,
            "chain_filter": args.chain,
            "pocket_cutoff": args.pocket_cutoff,
            "frequency_threshold": args.frequency,
            "n_structures": n_total,
            "residues": rows,
        }, fh, indent=2)

    n_in = sum(1 for r in rows if r["in_consensus"])
    print(f"Scanned {n_total} structures from {args.root}")
    print(f"Consensus pocket: {n_in}/{len(rows)} residues at occupancy ≥ {args.frequency}")
    print(f"Wrote {tsv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
