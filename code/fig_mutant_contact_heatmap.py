#!/usr/bin/env python3
"""
fig_mutant_contact_heatmap.py

Per-residue PA contact frequency heatmap comparing ETV6 ETS WT vs
Y344F, V345L, and Y344F/V345L mutants (all with DLPA).

Example (run from the deposit root):
  python code/fig_mutant_contact_heatmap.py \\
      --af3-input af3_inputs/run_ETV6_WT_ETS_DLPA.json \\
      --consensus results/wt_consensus_pocket_ETV6_ETS_DLPA.json \\
      --structures-dir af3_outputs \\
      --output-dir results
"""

import argparse
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

try:
    import gemmi
except ImportError:
    print("ERROR: gemmi not installed.")
    sys.exit(1)


# Deposit-relative defaults; all four are overridable on the command line.
BASE_DIR = Path(".")
OUTPUT_DIR = BASE_DIR / "af3_outputs"
FIG_DIR = BASE_DIR / "results"

CONTACT_CUTOFF = 4.5

CONSTRUCT_FASTA_JSON = BASE_DIR / "af3_inputs" / "run_ETV6_WT_ETS_DLPA.json"
CONSENSUS_JSON = BASE_DIR / "results" / "wt_consensus_pocket_ETV6_ETS_DLPA.json"

# ETV6 ETS construct starts at full-length residue 338 → local + 337 = full-length
CONSTRUCT_OFFSET = 337

POCKET_COLOR = "#f5a623"  # orange, matches ChimeraX pocket viz
AA31 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def load_construct_sequence():
    """Read ETV6 construct sequence from the AF3 input JSON."""
    if not CONSTRUCT_FASTA_JSON.exists():
        raise SystemExit(
            f"ERROR: AF3 input JSON not found: {CONSTRUCT_FASTA_JSON}\n"
            f"       Pass --af3-input with the path to the ETV6 WT AF3 input JSON."
        )
    data = json.loads(CONSTRUCT_FASTA_JSON.read_text())
    for s in data.get("sequences", []):
        if "protein" in s:
            return s["protein"]["sequence"]
    raise RuntimeError("No protein sequence found in construct JSON")


def load_consensus_pocket():
    """Return set of consensus pocket residue positions (local resseq)."""
    if not CONSENSUS_JSON.exists():
        raise SystemExit(
            f"ERROR: consensus pocket JSON not found: {CONSENSUS_JSON}\n"
            f"       Pass --consensus with the path to "
            f"wt_consensus_pocket_ETV6_ETS_DLPA.json."
        )
    data = json.loads(CONSENSUS_JSON.read_text())
    return {r["resseq"] for r in data["residues"] if r["in_consensus"]}

CONDITIONS = [
    ("ETV6_WT_ETS_DLPA",        "WT"),
    ("ETV6_Y344F_DLPA",         "Y344F"),
    ("ETV6_V345L_DLPA",         "V345L"),
    ("ETV6_Y344F_V345L_DLPA",   "Y344F/V345L"),
]


def get_ligand_chain_name(model):
    chains = list(model)
    return min(chains, key=lambda c: len(list(c.first_conformer()))).name


def get_protein_chains(model, exclude_chain):
    return [c.name for c in model if c.name != exclude_chain]


def extract_ligand_atoms(model, ligand_chain_name):
    chain = model.find_chain(ligand_chain_name)
    coords = []
    for residue in chain:
        for atom in residue:
            coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
    return np.array(coords)


def compute_contact_per_residue(model, protein_chains, ligand_coords, cutoff):
    contacts = []
    res_idx = 0
    for chain_name in protein_chains:
        chain = model.find_chain(chain_name)
        for residue in chain.first_conformer():
            res_idx += 1
            in_contact = False
            for atom in residue:
                rc = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                if np.min(np.linalg.norm(ligand_coords - rc, axis=1)) <= cutoff:
                    in_contact = True
                    break
            contacts.append((res_idx, in_contact))
    return contacts


def get_model_path(run_dir, run_name, seed, sample):
    return run_dir / f"seed-{seed}_sample-{sample}" / (
        f"{run_name}_seed-{seed}_sample-{sample}_model.cif"
    )


def compute_contact_frequencies(run_name):
    run_dir = OUTPUT_DIR / run_name
    if not run_dir.exists():
        return None, None
    all_contacts = []
    residue_indices = None
    for seed in range(1, 6):
        for sample in range(5):
            mp = get_model_path(run_dir, run_name, seed, sample)
            if not mp.exists():
                continue
            try:
                doc = gemmi.cif.read(str(mp))
                st = gemmi.make_structure_from_block(doc[0])
                model = st[0]
                lc = get_ligand_chain_name(model)
                pc = get_protein_chains(model, lc)
                lig = extract_ligand_atoms(model, lc)
                contacts = compute_contact_per_residue(
                    model, pc, lig, CONTACT_CUTOFF
                )
                if residue_indices is None:
                    residue_indices = np.array([c[0] for c in contacts])
                all_contacts.append([c[1] for c in contacts])
            except Exception as e:
                print(f"  ERROR processing {mp}: {e}")
                continue
    if not all_contacts:
        return None, None
    mat = np.array(all_contacts, dtype=float)
    return residue_indices, np.mean(mat, axis=0)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--af3-input", default=str(CONSTRUCT_FASTA_JSON),
                    help="AF3 input JSON supplying the ETV6 construct sequence "
                         "(default: %(default)s)")
    ap.add_argument("--consensus", default=str(CONSENSUS_JSON),
                    help="Consensus pocket JSON from build_wt_consensus_pocket.py "
                         "(default: %(default)s)")
    ap.add_argument("--structures-dir", default=str(OUTPUT_DIR),
                    help="Directory holding one subdirectory per AF3 run "
                         "(default: %(default)s)")
    ap.add_argument("--output-dir", default=str(FIG_DIR),
                    help="Directory for the output PDF/PNG (default: %(default)s)")
    return ap.parse_args()


def main():
    global CONSTRUCT_FASTA_JSON, CONSENSUS_JSON, OUTPUT_DIR, FIG_DIR
    args = parse_args()
    CONSTRUCT_FASTA_JSON = Path(args.af3_input)
    CONSENSUS_JSON = Path(args.consensus)
    OUTPUT_DIR = Path(args.structures_dir)
    FIG_DIR = Path(args.output_dir)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not OUTPUT_DIR.exists():
        print(f"WARNING: structures directory not found: {OUTPUT_DIR}\n"
              f"         Expected one subdirectory per AF3 run "
              f"(see af3_outputs/README.md).")

    print(f"Computing contact frequencies (cutoff = {CONTACT_CUTOFF} Å) "
          f"for {len(CONDITIONS)} conditions...\n")

    rows = []
    labels = []
    residue_indices = None

    for run_name, short in CONDITIONS:
        print(f"Processing: {run_name} ({short})")
        ri, freq = compute_contact_frequencies(run_name)
        if freq is None:
            print("  SKIPPED (no data)\n")
            continue

        if residue_indices is None:
            residue_indices = ri

        if len(freq) != len(residue_indices):
            n = min(len(freq), len(residue_indices))
            freq = freq[:n]
            residue_indices = residue_indices[:n]
            rows = [r[:n] for r in rows]

        rows.append(freq)
        labels.append(short)
        print(f"  {int(np.sum(freq > 0))} contacted, "
              f"{int(np.sum(freq > 0.5))} consensus (>50%)\n")

    if not rows:
        print("ERROR: No data to plot.")
        sys.exit(1)

    matrix = np.array(rows)

    construct_seq = load_construct_sequence()
    consensus_set = load_consensus_pocket()

    fig, ax = plt.subplots(figsize=(13, 4), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="YlOrRd",
        vmin=0, vmax=1,
        interpolation="nearest",
        extent=(
            residue_indices[0] - 0.5, residue_indices[-1] + 0.5,
            len(labels) - 0.5, -0.5,
        ),
    )

    # Light vertical shading at consensus pocket columns
    for p in consensus_set:
        if residue_indices[0] <= p <= residue_indices[-1]:
            ax.axvspan(p - 0.5, p + 0.5, color=POCKET_COLOR, alpha=0.12, zorder=0)

    # Mark triad residues Y344 (local 7) and V345 (local 8).
    for res_num, res_label, dx in [(7, "Y344", -6.0), (8, "V345", 6.0)]:
        ax.annotate(
            res_label,
            xy=(res_num, -0.5),
            xytext=(res_num + dx, -1.6),
            ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#A32D2D",
            arrowprops=dict(
                arrowstyle="-|>", color="#A32D2D",
                lw=1.0, shrinkA=0, shrinkB=2,
            ),
            annotation_clip=False,
        )

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)

    # Bottom residue labels: only the 16 consensus pocket + Y344 mutation site.
    # Labels use full-length ETV6 numbering (local + 337).
    mutation_positions = {7, 8}  # Y344 and V345 in local numbering
    to_label = sorted(consensus_set | mutation_positions)
    label_texts = [
        f"{construct_seq[p - 1]}{p + CONSTRUCT_OFFSET}" for p in to_label
    ]

    ax.set_xticks(to_label)
    ax.set_xticklabels(label_texts, fontsize=8, rotation=90, ha="center")
    for tick, pos in zip(ax.get_xticklabels(), to_label):
        if pos in mutation_positions:
            tick.set_color("#A32D2D")
            tick.set_fontweight("bold")
        elif pos in consensus_set:
            tick.set_color(POCKET_COLOR)
            tick.set_fontweight("bold")

    ax.set_xlabel("Pocket residue (full-length ETV6 numbering)", fontsize=9)
    ax.set_title(
        f"ETV6 ETS + DLPA contact frequency across triad mutants "
        f"(cutoff = {CONTACT_CUTOFF} Å, N = 25 per condition)",
        fontsize=11, pad=10,
    )

    cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
    cbar.set_label("Contact frequency", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.tick_params(labelsize=9)

    # Legend strip explaining annotations
    legend_handles = [
        mpatches.Patch(color=POCKET_COLOR, alpha=0.5,
                       label="Consensus pocket residue (16)"),
        mpatches.Patch(color="#A32D2D", alpha=0.6,
                       label="Mutation site (Y344, V345)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              bbox_to_anchor=(1.0, 1.30), fontsize=8, frameon=False, ncol=2)

    plt.tight_layout()

    out_pdf = FIG_DIR / "fig_mutant_contact_heatmap.pdf"
    out_png = FIG_DIR / "fig_mutant_contact_heatmap.png"
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, bbox_inches="tight", facecolor="white", dpi=300)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")
    plt.close()


if __name__ == "__main__":
    main()
