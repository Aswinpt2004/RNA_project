import torch
from torch.utils.data import Dataset
from structure_utils import dot_bracket_to_matrix
from encoding_utils import one_hot_encode_sequence
from Bio import SeqIO
import os

class RNADataset(Dataset):
    def __init__(self, fasta_path, dbn_dir, max_len=400, max_samples=1000):
        self.pairs = []
        records = list(SeqIO.parse(fasta_path, "fasta"))
        for record in records:
            rna_id = record.id
            seq = str(record.seq)
            dbn_path = os.path.join(dbn_dir, f"{rna_id}.dbn")
            if not os.path.isfile(dbn_path):
                continue
            with open(dbn_path) as f:
                for line in f:
                    line = line.strip()
                    if set(line).issubset(set("().[]{}<>")):
                        struct = line
                        break
                else:
                    continue
            if len(seq) == len(struct) and len(seq) <= max_len:
                self.pairs.append((seq, struct))
                if len(self.pairs) >= max_samples:
                    break

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        seq, struct = self.pairs[idx]
        enc = one_hot_encode_sequence(seq)
        label = dot_bracket_to_matrix(struct)
        return torch.tensor(enc, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def pad_collate(batch):
    sequences, matrices = zip(*batch)
    lengths = [s.shape[0] for s in sequences]
    max_len = max(lengths)
    padded_seqs = torch.zeros(len(batch), max_len, 4)
    padded_labels = torch.zeros(len(batch), max_len, max_len)
    for i, (seq, mat) in enumerate(zip(sequences, matrices)):
        L = seq.shape[0]
        padded_seqs[i, :L] = seq
        padded_labels[i, :L, :L] = mat
    return padded_seqs, padded_labels, lengths
