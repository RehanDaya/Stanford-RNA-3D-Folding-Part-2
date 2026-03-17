Overview
If you were handed only a melody and asked to compose a full symphony, you’d need to rely on patterns, structure, and musical intuition to build the rest. In the same way, predicting Ribonucleic acid (RNA)’s 3D structure means using its sequence to determine how it folds into the shapes that define its function.

In this part 2 of the original Stanford RNA 3D Folding competition, help uncover how RNA molecules fold and function at the molecular level by developing machine learning models that predict the 3D structure of RNA molecules using only their sequences.

Description
RNA is essential to life’s core functions, but predicting its 3D structure remains difficult. Unlike proteins, where models like AlphaFold have made major progress, RNA modeling is still held back by limited data and the complexity of RNA folding.

This is the second Stanford RNA 3D Folding Challenge. The first marked a major milestone, where fully automated models matched human experts for the first time. Now, you’ll face even more complex targets, including ones with no structural templates, and a new evaluation metric designed to reward greater accuracy.

Your work could solve a key challenge in molecular biology. Better RNA structure prediction could unlock new treatments, accelerate research, and deepen our understanding of how life works. This competition runs on a roughly two-month timeline, aiming to surface new breakthroughs ahead of the 17th Critical Assessment of Structure Prediction (CASP17) in April 2026.

This competition is made possible through a worldwide collaborative effort including the organizers, experimental RNA structural biologists, NVIDIA Healthcare, the AI@HHMI initiative of the Howard Hughes Medical Institute, and Stanford University School of Medicine.

Evaluation
Submissions are scored using TM-score ("template modeling" score), which goes from 0.0 to 1.0 (higher is better):


where:

Lref is the number of residues solved in the experimental reference structure ("ground truth").

Lalign is the number of aligned residues.

di is the distance between the ith pair of aligned residues, in Angstroms.

d0 is a distance scaling factor in Angstroms, defined as:


for Lref ≥ 30; and d0 = 0.3, 0.4, 0.5, 0.6, or 0.7 for Lref <12, 12-15, 16-19, 20-23, or 24-29, respectively.

The rotation and translation of predicted structures to align with experimental reference structures are carried out by US-align. To reward high accuracy modeling, the alignment will reward only residues in the prediction that align with the reference residues with the same numbering.

For each target RNA sequence, you will submit 5 predictions and your final score will be the average of best-of-5 TM-scores of all targets. For a few targets, multiple slightly different structures have been captured experimentally; your predictions' scores will be based on the best TM-score compared to each of these reference structures.

The scoring metric is publicly available at this link.

Submission File
For each sequence in the test set, you can predict five structures. Your notebook should look for a file test_sequences.csv and output submission.csv. This file should contain x, y, z coordinates of the C1' atom in each residue across your predicted structures 1 to 5:

ID,resname,resid,x_1,y_1,z_1,... x_5,y_5,z_5
R1107_1,G,1,-7.561,9.392,9.361,... -7.301,9.023,8.932
R1107_2,G,1,-8.02,11.014,14.606,... -7.953,10.02,12.127
etc.
You must submit five sets of coordinates.

Additional Resources
What's the state-of-the-art in RNA 3D structure prediction?

2024 CASP16 challenge – 2025 paper from assessors
https://onlinelibrary.wiley.com/doi/10.1002/prot.70072

Part 1 of the Stanford RNA 3D Folding challenge

Outcome of Part 1, Kaggle Post
https://www.kaggle.com/competitions/stanford-rna-3d-folding/discussion/609187

"Template-based RNA structure prediction advanced through a blind code competition", Preprint summarizing Part 1
https://www.biorxiv.org/content/10.64898/2025.12.30.696949v1.full

How to think about RNA structure
A perspective from domain experts
https://www.pnas.org/doi/10.1073/pnas.2112677119

Dataset Description
In this competition you will predict five 3D structures for each RNA sequence.

Files
[train/validation/test]_sequences.csv - the target sequences of the RNA molecules.

target_id - (string) An arbitrary identifier. In train_sequences.csv, this is the id of the entry in the Protein Data Bank.
sequence - (string) The RNA sequence of all chains in the target, concatenated together according to stoichiometry
temporal_cutoff - (string) The date in yyyy-mm-dd format that the sequence was or will be published.
description - (string) Details of the origins of the sequence. For PDB entries, this is the entry title.
stoichiometry - (string) the chains used for the target. These take the form of {chain:number}, where chain corresponds to the author-defined chain in all_sequences, joined with a semicolon delimiter (;).
all_sequences - (string) FASTA-formatted sequences of all molecular chains present in the experimentally solved structure. May include multiple copies of the target RNA (look for the word "Chains" in the header) and/or partners like other RNAs or proteins or DNA. You don't need to make predictions for all these molecules, just the ones specified in stoichiometry which are concatenated in sequence. Can be parsed into a dictionary with extra/parse_fasta_py.py.
ligand_ids - (string) three-letter names in PDB chemical component dictionary of any small molecule ligands solved in the experimental structure, joined with a semicolon delimiter (;). You don't need to make predictions for these molecules.
ligand_SMILES - (string) SMILES strings giving chemical structures of any small molecule ligands solved in the experimental structure, joined with a semicolon delimiter (;).
[train/validation]_labels.csv - experimental structures.

ID - (string) that identifies the target_id and residue number, separated by _. Note: residue numbers use one-based indexing.
resname - (character) The RNA nucleotide ( A, C, G, or U) for the residue.
resid - (integer) residue number.
x_1,y_1,z_1,x_2,y_2,z_2,… - (float) Coordinates (in Angstroms) of the C1' atom for each experimental RNA structure. There is typically one structure for the RNA sequence, and train_labels.csv curates one structure for each training sequence. However, in some targets the experimental method has captured more than one conformation, and each will be used as a potential reference for scoring your predictions.
chain - (string) residue's chain ID. For the target there is one chain assigned to each unique sequence, potentially derived from author-assigned chain in PDB entry. Note: Multiple chains of the molecule can share the same chain if they have the same sequence.
copy - (integer) which chain copy (1,2, …) the residue is in. Greater than 1 if there are multiple copies of the same sequence in the structure.
sample_submission.csv

Same format as train_labels.csv but with five sets of coordinates for each of your five predicted structures (x_1,y_1,z_1,x_2,y_2,z_2,…x_5,y_5,z_5).
You must submit five sets of coordinates.
Note that x,y,z are clipped between -999.999 and 9999.999 before scoring, due to use of a legacy PDB format that has maximal 8 characters for coordinates.
chain and copy do not have to be provided.
MSA/

contains multiple sequence alignments in FASTA format for each target in train_sequences.csv and in validation_sequences.csv. Files are named {target_id}.MSA.fasta. During evaluation with hidden test sequences, your notebook will have access to these MSA files for the hidden test_sequences.csv.
For multi-chain targets, each homolog found for a given chain sequence is presented in a separate row with placeholders for other chain sequences provided as gaps (-).
The header for each homolog encodes the source of the sequence. A tag chain={chain} is appended. If multiple copies of the chain are present in the target, a tag copies={copy} is also appended. Tags are separated by | delimiter.
PDB_RNA/ contains 3D structural information available in the Protein Data Bank with

{PDB_id}.cif files for each RNA-containing entry
pdb_seqres_NA.fasta - sequences of all nucleic acid chains in the PDB in FASTA format.
pdb_release_dates_NA.csv - Entry ID and Release dates of the RNA-containing PDB entries in csv format.
extra/

parse_fasta_py.py - helper script with function parse_fasta(), which can take the all_sequences field of {train/test/validation}_sequences.csv and produce a dictionary of chain:sequence.
rna_metadata.csv - the data extracted from all RNA and RNA/DNA hybrid structures up to December 17,2025
README.md - description of the data in the rna_metadata.csv
Additional notes
train/validation split: The datasets were partitioned using a cluster-based temporal split to minimize homology between training and evaluation data. All chain sequences were clustered using MMseqs2 at a 30% identity level, and each cluster was assigned a temporal_cutoff based on its oldest member. Any cluster containing at least one entry released before May 29, 2025, was assigned to the training set. Only clusters where all members were released after May 29, 2025,the final submission date of the last Stanford RNA 3D Folding competition and up to December 17, 2025) were included in the validation set. This ensures that no structure in the validation set shares more than 30% sequence identity with any structure released prior to the training cutoff.

The sequences in the validation_sequences.csv (which is the same as test_sequences.csv publicly provided here) were further filtered to have composition of at least 40% RNA and unique sequences (up to sequence identity 90%).

The train_sequences.csv has not been filtered for sequence redundancy but contains only PDB entries that pass selection criteria. train_sequences.csv should have no overlap with validation_sequences.csv. Note that train_sequences.csv has been filtered, so it does not have all RNA targets in the PDB to date or all the cif files in PDB_RNA/.

The additional file extra/rna_metadata.csv contains data extracted from all RNA and RNA/DNA hybrid structures up to December 17,2025. The metadata description is included in extra/README.md. These metadata were used to filter the structures that were included in the {train,test,validation}_sequences.csv using the following criteria:

canonical ACGU residues or modified residues that can be mapped to canonical using either PDB chemical component dictionary or NAKB mapping
no undefined (N) residues and no T (for hybrid NA)
no more than 25% of residues that were modified / non-canonical
at least 50% residues reported in the sequence were modeled/observed
total adjusted structuredness (see extra/README.md) of all RNA chains is at least 20%
If any RNA chain in the PDB file didn't meet those criteria, the whole entry has been removed. Additionally the remaining entries were filtered to have at least 10 nt in all chains combined, and entries without resolved C1' atoms (for example P traces) were rejected. Targets in {test,validation}_sequences.csv were further filtered to contain only entries with at least 40% RNA composition and the redundancy was removed by selecting single entry based on MMseqs2 clustering at 90% identity level. The processing pipeline is available here: https://github.com/JaneliaSciComp/jrc-rna-structure-pipeline
