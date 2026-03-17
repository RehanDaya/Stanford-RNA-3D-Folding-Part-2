Stanford RNA 3D Folding – Part 2
================================

This project implements models and tooling for the **Stanford RNA 3D Folding Part 2** competition. The goal is to predict the **3D structure of RNA molecules from sequence alone**, advancing methods that can match or exceed human expert performance and help close the gap with protein-structure tools like AlphaFold.

The competition focuses on more challenging RNA targets (including ones without structural templates) and uses **TM-score** as the evaluation metric (0.0–1.0, higher is better), computed via US-align. For each RNA sequence, you must output **five candidate 3D structures**, and scoring is based on the **best-of-5 TM-score** averaged across all targets (for some targets, multiple experimental structures exist and the best TM-score among them is used).

### Dataset 

https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/data

The dataset provides:
- **`[train/validation/test]_sequences.csv`**: RNA target sequences, IDs, temporal cutoffs, stoichiometry, full chain FASTA sequences, and ligand information.
- **`[train/validation]_labels.csv`**: Experimental C1' atom coordinates (x, y, z) for each residue (possibly multiple conformations), plus chain and copy identifiers.
- **`sample_submission.csv`**: Example of the required submission format with C1' coordinates for five predicted structures per residue.
- **`MSA/`**: Multiple sequence alignments for each train/validation target, with headers encoding chain and copy information.
- **`PDB_RNA/`**: RNA-related structural data from the Protein Data Bank (`{PDB_id}.cif`, nucleic acid FASTA, and release dates).
- **`extra/`**: Helper scripts (e.g., `parse_fasta_py.py`) and `rna_metadata.csv` describing all RNA and RNA/DNA hybrid structures (see `extra/README.md`).

The train/validation split uses **cluster-based temporal partitioning** (MMseqs2 at 30% identity) to minimize homology between train and evaluation data. Additional filters ensure sufficient RNA content, structural completeness, and removal of redundant or low-quality entries.
