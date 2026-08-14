#!/usr/bin/env python3
"""
SASA analysis using a FIXED WT-consensus pocket residue list, comparable
across genotypes (WT, Y344F, V345L, Y344F+V345L, ...).

Loads <consensus-json> (output of build_wt_consensus_pocket.py) and selects
residues where in_consensus=True. Matching uses (chain, resseq, icode) only —
resname is ignored so mutated positions (Y344→F, V345→L) are picked up at the
same sequence position in mutant CIFs.

Per CIF, computes:
  - ligand_sasa_complex, ligand_sasa_isolated, ligand_exposed_pct
  - pocket_sasa_with_ligand, pocket_sasa_no_ligand, delta_pocket_sasa
  - n_pocket_residues_found  (sanity check: should equal n_consensus)

Outputs:
  <out-prefix>_per_structure.tsv
  <out-prefix>_per_genotype.tsv     mean ± SD per genotype
  <out-prefix>_summary.png          2x2 boxplots across genotypes

Example (run from the deposit root):
  python code/pocket_sasa_consensus.py \\
      --consensus results/wt_consensus_pocket_ETV6_ETS_DLPA.json \\
      --runs WT=af3_outputs/ETV6_WT_ETS_DLPA \\
             Y344F=af3_outputs/ETV6_Y344F_DLPA \\
             V345L=af3_outputs/ETV6_V345L_DLPA \\
             Y344F_V345L=af3_outputs/ETV6_Y344F_V345L_DLPA \\
      --ligand LIG_B \\
      --out-prefix results/pocket_sasa_consensus_DLPA
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import freesasa
import matplotlib.pyplot as plt
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import is_aa


# ---------- structure parsing ----------

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
    ligand_residues, protein_residues = [], []
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
    return ligand_residues, protein_residues


def position_key(residue):
    chain = residue.get_parent().id
    resseq, icode = residue.get_id()[1], residue.get_id()[2]
    icode = "" if icode in (None, " ") else str(icode)
    return (chain, int(resseq), icode)


def select_pocket(protein_residues, consensus_positions):
    found = []
    for res in protein_residues:
        if position_key(res) in consensus_positions:
            found.append(res)
    return found


def mean_residue_plddt(residues):
    """AF3 stores per-residue pLDDT in the atomic B-factor column.
    Take one value per residue (prefer CA, else first atom)."""
    if not residues:
        return 0.0
    vals = []
    for res in residues:
        ca = next((a for a in res.get_atoms() if a.get_name().strip() == "CA"), None)
        atom = ca if ca is not None else next(iter(res.get_atoms()), None)
        if atom is not None:
            vals.append(float(atom.get_bfactor()))
    return sum(vals) / len(vals) if vals else 0.0


def mean_atom_plddt(residues):
    """Mean B-factor over all atoms — useful for ligand (no CA equivalent)."""
    vals = [float(a.get_bfactor()) for res in residues for a in res.get_atoms()]
    return sum(vals) / len(vals) if vals else 0.0


# ---------- freesasa helpers ----------

def atom_records(residues):
    records = []
    for residue in residues:
        chain = residue.get_parent().id
        resname = residue.get_resname().strip()
        resseq = residue.get_id()[1]
        for atom in residue.get_atoms():
            c = atom.get_coord()
            records.append((atom.get_name().strip(), resname, resseq, chain,
                            float(c[0]), float(c[1]), float(c[2])))
    return records


def build_fs(records):
    fs = freesasa.Structure()
    for name, rn, rs, ch, x, y, z in records:
        fs.addAtom(name, rn, str(rs), ch, x, y, z)
    return fs


def total_sasa(records):
    if not records:
        return 0.0
    return freesasa.calc(build_fs(records)).totalArea()


def split_sasa(first_records, second_records):
    """SASA of first group when complexed with second group; returns first-group area."""
    all_records = first_records + second_records
    if not first_records:
        return 0.0
    result = freesasa.calc(build_fs(all_records))
    return sum(result.atomArea(i) for i in range(len(first_records)))


# ---------- per-structure analysis ----------

def analyze_structure(cif_path, ligand_name, consensus_positions, chain_filter=None):
    lig_res, prot_res = get_structure_components(cif_path, ligand_name, chain_filter)
    if not lig_res or not prot_res:
        return {"file": str(cif_path), "status": "missing_ligand_or_protein"}

    pocket_res = select_pocket(prot_res, consensus_positions)
    if not pocket_res:
        return {"file": str(cif_path), "status": "no_consensus_residues_found"}

    lig_rec = atom_records(lig_res)
    prot_rec = atom_records(prot_res)
    pocket_rec = atom_records(pocket_res)

    lig_complex = split_sasa(lig_rec, prot_rec)
    lig_isolated = total_sasa(lig_rec)
    exposed_pct = 100.0 * lig_complex / lig_isolated if lig_isolated > 0 else 0.0

    pocket_with_lig = split_sasa(pocket_rec, lig_rec)
    pocket_no_lig = total_sasa(pocket_rec)
    delta_pocket = pocket_no_lig - pocket_with_lig
    cavity_covered_pct = 100.0 * delta_pocket / pocket_no_lig if pocket_no_lig > 0 else 0.0

    pocket_plddt = mean_residue_plddt(pocket_res)
    ligand_plddt = mean_atom_plddt(lig_res)

    return {
        "file": str(cif_path),
        "status": "ok",
        "n_pocket_residues_found": len(pocket_res),
        "ligand_sasa_complex": lig_complex,
        "ligand_sasa_isolated": lig_isolated,
        "ligand_exposed_pct": exposed_pct,
        "pocket_sasa_with_ligand": pocket_with_lig,
        "pocket_sasa_no_ligand": pocket_no_lig,
        "delta_pocket_sasa": delta_pocket,
        "cavity_covered_pct": cavity_covered_pct,
        "mean_pocket_plddt": pocket_plddt,
        "mean_ligand_plddt": ligand_plddt,
    }


# ---------- driver ----------

def load_consensus(path):
    data = json.load(open(path))
    positions = set()
    for r in data["residues"]:
        if r["in_consensus"]:
            positions.add((r["chain"], r["resseq"], r["icode"]))
    return positions, data


def parse_runs(spec):
    """Each entry is name=dir or name=dir@consensus_json (overrides global)."""
    out = []
    for s in spec:
        if "=" not in s:
            raise SystemExit(f"--runs entry must be name=dir[@consensus_json], got: {s}")
        name, rest = s.split("=", 1)
        if "@" in rest:
            d, j = rest.split("@", 1)
            out.append((name, d, j))
        else:
            out.append((name, rest, None))
    return out


def write_per_structure_tsv(rows, path):
    cols = ["genotype", "file", "status", "n_pocket_residues_found",
            "ligand_sasa_complex", "ligand_sasa_isolated", "ligand_exposed_pct",
            "pocket_sasa_with_ligand", "pocket_sasa_no_ligand", "delta_pocket_sasa",
            "cavity_covered_pct", "mean_pocket_plddt", "mean_ligand_plddt"]
    with open(path, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")


def write_per_genotype_tsv(rows, path):
    metrics = ["ligand_exposed_pct", "cavity_covered_pct",
               "pocket_sasa_with_ligand", "pocket_sasa_no_ligand",
               "delta_pocket_sasa", "mean_pocket_plddt", "mean_ligand_plddt"]
    genotypes = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        genotypes.setdefault(r["genotype"], []).append(r)

    with open(path, "w") as fh:
        header = ["genotype", "n"]
        for m in metrics:
            header += [m + "_mean", m + "_sd"]
        fh.write("\t".join(header) + "\n")
        for g, rs in genotypes.items():
            line = [g, str(len(rs))]
            for m in metrics:
                vals = [r[m] for r in rs]
                mean = statistics.mean(vals)
                sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
                line += [f"{mean:.3f}", f"{sd:.3f}"]
            fh.write("\t".join(line) + "\n")


def boxplot_summary(rows, genotype_order, out_png):
    metrics = [
        ("ligand_exposed_pct", "Ligand exposed (%)"),
        ("delta_pocket_sasa", "Δ pocket SASA (Å²)"),
        ("mean_pocket_plddt", "Mean pocket pLDDT"),
        ("pocket_sasa_with_ligand", "Pocket SASA with ligand (Å²)"),
        ("pocket_sasa_no_ligand", "Pocket SASA without ligand (Å²)"),
        ("mean_ligand_plddt", "Mean ligand pLDDT"),
    ]

    by_geno = {g: {m: [] for m, _ in metrics} for g in genotype_order}
    for r in rows:
        if r.get("status") != "ok":
            continue
        g = r["genotype"]
        if g not in by_geno:
            continue
        for m, _ in metrics:
            by_geno[g][m].append(r[m])

    n = len(genotype_order)
    positions = [1 + i * 0.6 for i in range(n)]
    # Width scales with number of conditions so boxes stay readable.
    # 3-column subplot layout, ~1.1 inch per condition per panel.
    fig_width = min(16, max(7.5, 3.0 + n * 1.3))
    fig, axes = plt.subplots(2, 3, figsize=(fig_width, 7), dpi=200, facecolor="white")
    for ax, (m, label) in zip(axes.flat, metrics):
        ax.set_facecolor("white")
        data = [by_geno[g][m] for g in genotype_order]
        ax.boxplot(
            data,
            labels=genotype_order,
            positions=positions,
            widths=0.4,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.2},
        )
        for i, (xpos, vals) in enumerate(zip(positions, data)):
            x_jitter = [xpos + (j - len(vals) / 2) * 0.012 for j in range(len(vals))]
            ax.plot(x_jitter, vals, "o",
                    markersize=2.5, color="#888888", markeredgewidth=0, alpha=0.7)
        ax.set_xlim(positions[0] - 0.4, positions[-1] + 0.4)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha("right")
        if "plddt" in m:
            ax.axhline(70, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
            ax.axhline(90, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
    fig.suptitle("Consensus-pocket SASA + pLDDT across genotypes", y=1.00)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    # Always emit a PDF alongside the PNG
    out_pdf = str(out_png)
    if out_pdf.lower().endswith(".png"):
        out_pdf = out_pdf[:-4] + ".pdf"
    else:
        out_pdf = out_pdf + ".pdf"
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--consensus", default=None,
                    help="Default consensus JSON; per-run override via name=dir@json")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="genotype=dir entries, e.g. WT=output/ETV6_WT_ETS_DLPA")
    ap.add_argument("--ligand", required=True, help="Ligand residue name")
    ap.add_argument("--chain", default=None, help="Optional protein chain filter")
    ap.add_argument("--out-prefix", default="pocket_sasa_consensus")
    args = ap.parse_args()

    default_positions = None
    if args.consensus:
        default_positions, _ = load_consensus(args.consensus)
        print(f"Default consensus: {len(default_positions)} positions from {args.consensus}")

    runs = parse_runs(args.runs)
    genotype_order = [g for g, _, _ in runs]

    all_rows = []
    for genotype, run_dir, override_json in runs:
        if override_json:
            positions, _ = load_consensus(override_json)
            print(f"[{genotype}] uses {len(positions)} positions from {override_json}")
        elif default_positions is not None:
            positions = default_positions
        else:
            raise SystemExit(f"[{genotype}] no consensus JSON (set --consensus or use name=dir@json)")

        cifs = find_cif_files(run_dir)
        print(f"[{genotype}] {len(cifs)} CIF files under {run_dir}")
        for cif in cifs:
            r = analyze_structure(cif, args.ligand, positions, args.chain)
            r["genotype"] = genotype
            all_rows.append(r)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    per_struct = str(out_prefix) + "_per_structure.tsv"
    per_geno = str(out_prefix) + "_per_genotype.tsv"
    summary_png = str(out_prefix) + "_summary.png"

    write_per_structure_tsv(all_rows, per_struct)
    write_per_genotype_tsv(all_rows, per_geno)
    boxplot_summary(all_rows, genotype_order, summary_png)

    n_ok = sum(1 for r in all_rows if r.get("status") == "ok")
    print(f"Processed {n_ok}/{len(all_rows)} structures successfully.")
    print(f"Wrote {per_struct}")
    print(f"Wrote {per_geno}")
    print(f"Wrote {summary_png}")
    summary_pdf = summary_png[:-4] + ".pdf" if summary_png.lower().endswith(".png") else summary_png + ".pdf"
    print(f"Wrote {summary_pdf}")


if __name__ == "__main__":
    main()
