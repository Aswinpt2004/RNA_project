from data_loader import RNADataset
from structure_utils import dot_bracket_to_matrix
from encoding_utils import one_hot_encode_sequence
import os

def main():
    fasta_path = "bpRNA_1m_90.fasta"
    dbn_folder = "bpRNA_1m_90_DBNFILES"

    print("📁 Current directory:", os.getcwd())
    print("📂 Files in directory:", os.listdir())

    dataset = RNADataset(fasta_path, dbn_folder, max_samples=5)

    for i, (seq, struct) in enumerate(dataset.pairs[:5]):
        print(f"\nSample {i+1}:")
        print("Sequence:", seq)
        print("Structure:", struct)
        matrix = dot_bracket_to_matrix(struct)
        print("Base-pair matrix:")
        for row in matrix:
            print(row)
        encoded = one_hot_encode_sequence(seq)
        print("One-hot encoding shape:", encoded.shape)

if __name__ == "__main__":
    main()
