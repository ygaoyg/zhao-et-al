#!/usr/bin/env python3
"""
Map the WT-consensus pocket residue positions (defined in ETV6 local numbering)
to homologous positions in other ETS-family paralogs.

Inputs:
  - ETV6 consensus pocket JSON (from build_wt_consensus_pocket.py)
  - matched_ETS_windows.fasta with one ETS-window sequence per paralog

Approach:
  Pairwise-align (global, BLOSUM62) ETV6's construct sequence to each paralog's
  construct sequence. For each ETV6 pocket position (1-based local resseq), find
  the aligned position in the paralog — i.e. the paralog's local resseq for the
  homologous residue. Positions that align to a gap are flagged as "no equivalent."

Outputs:
  <out-dir>/<paralog>_consensus_pocket.json   per paralog
  <out-dir>/mapping_table.tsv                 ETV6_pos -> {paralog: (pos, aa, gap?)}

Example (run from the deposit root):
  python code/map_pocket_to_paralogs.py \\
      --etv6-consensus results/wt_consensus_pocket_ETV6_ETS_DLPA.json \\
      --fasta results/matched_ETS_windows.fasta \\
      --etv6-id ETV6_P41212_res338-442 \\
      --paralogs ETV7_Q9Y603_res223-326 ELF1_P32519_res210-312 \\
                 ELF3_P78545_res275-370 ERF_P50548_res29-106 \\
                 ELK1_P19419_res7-90 GABPA_Q06546_res322-400 \\
      --out-dir results/paralog_consensus
"""

import argparse
import json
from pathlib import Path

from Bio import Align, SeqIO
from Bio.Align import substitution_matrices


def read_fasta(path):
    return {r.id: str(r.seq) for r in SeqIO.parse(path, "fasta")}


def make_aligner():
    aln = Align.PairwiseAligner()
    aln.mode = "global"
    aln.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aln.open_gap_score = -10
    aln.extend_gap_score = -0.5
    return aln


def align_pair(aligner, ref_seq, query_seq):
    alignment = aligner.align(ref_seq, query_seq)[0]
    # Bio.Align.Alignment exposes .aligned (a pair of ranges) and an indexed
    # representation we can iterate column-wise.
    a_str = str(alignment).splitlines()
    # Lines: [ref-with-gaps, match-bars, query-with-gaps] (sometimes with names)
    # Use alignment[0], alignment[1] which return the gapped strings directly.
    ref_gapped = alignment[0]
    qry_gapped = alignment[1]
    return ref_gapped, qry_gapped


def build_pos_map(ref_gapped, qry_gapped):
    """Return list of (ref_pos_1based or None, qry_pos_1based or None, ref_aa, qry_aa) per column."""
    ref_pos = 0
    qry_pos = 0
    cols = []
    for a, b in zip(ref_gapped, qry_gapped):
        rp = qp = None
        if a != "-":
            ref_pos += 1
            rp = ref_pos
        if b != "-":
            qry_pos += 1
            qp = qry_pos
        cols.append((rp, qp, a, b))
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etv6-consensus", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--etv6-id", required=True, help="FASTA header for ETV6 entry")
    ap.add_argument("--paralogs", nargs="+", required=True,
                    help="FASTA headers for paralog entries to map")
    ap.add_argument("--out-dir", default="paralog_consensus")
    args = ap.parse_args()

    seqs = read_fasta(args.fasta)
    if args.etv6_id not in seqs:
        raise SystemExit(f"{args.etv6_id} not found in {args.fasta}")
    etv6_seq = seqs[args.etv6_id]

    consensus = json.load(open(args.etv6_consensus))
    etv6_pocket_positions = [
        (r["chain"], r["resseq"], r["icode"], r["wt_resname"])
        for r in consensus["residues"]
        if r["in_consensus"]
    ]

    aligner = make_aligner()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table_rows = []
    header = ["etv6_resseq", "etv6_aa"]
    for p in args.paralogs:
        header += [f"{p}_resseq", f"{p}_aa"]
    table_rows.append(header)

    paralog_maps = {}  # paralog_id -> list of {chain, resseq, ...}
    for p in args.paralogs:
        if p not in seqs:
            raise SystemExit(f"{p} not found in {args.fasta}")
        ref_gap, qry_gap = align_pair(aligner, etv6_seq, seqs[p])
        cols = build_pos_map(ref_gap, qry_gap)
        # Build lookup: etv6_resseq -> (paralog_resseq, aa)
        etv6_to_par = {}
        for rp, qp, ra, qa in cols:
            if rp is not None:
                etv6_to_par[rp] = (qp, qa)
        paralog_maps[p] = etv6_to_par

    # build mapping table rows
    for chain, resseq, icode, wt_resname in etv6_pocket_positions:
        row = [str(resseq), wt_resname]
        for p in args.paralogs:
            qp, qa = paralog_maps[p].get(resseq, (None, "-"))
            row += [str(qp) if qp is not None else "-", qa]
        table_rows.append(row)

    # write mapping table
    mapping_tsv = out_dir / "mapping_table.tsv"
    with open(mapping_tsv, "w") as fh:
        for row in table_rows:
            fh.write("\t".join(row) + "\n")

    # write one consensus JSON per paralog
    for p in args.paralogs:
        residues = []
        for chain, resseq, icode, wt_resname in etv6_pocket_positions:
            qp, qa = paralog_maps[p].get(resseq, (None, "-"))
            if qp is None or qa == "-":
                continue  # paralog has a gap here — drop the position
            residues.append({
                "chain": chain,
                "resseq": qp,
                "icode": icode,
                "wt_resname": _aa_three(qa),
                "occupancy": 1.0,
                "n_observed": 0,
                "n_total": 0,
                "in_consensus": True,
                "etv6_equivalent_resseq": resseq,
                "etv6_equivalent_resname": wt_resname,
            })
        out_json = out_dir / f"{p}_consensus_pocket.json"
        with open(out_json, "w") as fh:
            json.dump({
                "source": f"mapped from {args.etv6_consensus}",
                "paralog": p,
                "n_residues": len(residues),
                "residues": residues,
            }, fh, indent=2)

    print(f"Wrote {mapping_tsv}")
    for p in args.paralogs:
        n = sum(1 for r in json.load(open(out_dir / f"{p}_consensus_pocket.json"))["residues"]
                if r["in_consensus"])
        print(f"Wrote {out_dir}/{p}_consensus_pocket.json  ({n}/{len(etv6_pocket_positions)} positions mapped)")


_AA3 = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
        "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
        "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
        "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}


def _aa_three(one):
    return _AA3.get(one.upper(), "UNK")


if __name__ == "__main__":
    main()
