import torch
import matplotlib.pyplot as plt
from model import RNABasePairPredictor
from structure_utils import evaluate_base_pair_prediction
from encoding_utils import one_hot_encode_sequence
from data_loader import RNADataset
import numpy as np
import os

def visualize_matrix(matrix, title="Predicted Base Pair Probabilities", cmap="viridis", idx=0):
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap=cmap, origin='lower')
    plt.title(f"Sample {idx + 1} — {title}")
    plt.xlabel("Position j")
    plt.ylabel("Position i")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

def prob_matrix_to_pseudoknot_dot_bracket(prob_matrix, threshold=0.5):
    L = prob_matrix.shape[0]
    paired = [-1] * L
    pairs = []

    for i in range(L):
        for j in range(i+1, L):
            if prob_matrix[i, j] > threshold:
                pairs.append((i, j, prob_matrix[i, j]))

    pairs.sort(key=lambda x: x[2], reverse=True)

    brackets = ['(', ')', '[', ']', '{', '}', '<', '>']
    bracket_pairs = [(0,1), (2,3), (4,5), (6,7)]
    used_pairs = set()
    bracket_usage = [-1] * L

    structure = ['.'] * L

    def is_crossing(i1, j1, i2, j2):
        return (i1 < i2 < j1 < j2) or (i2 < i1 < j2 < j1)

    for i, j, prob in pairs:
        if paired[i] != -1 or paired[j] != -1:
            continue
        assigned = False
        for b_idx, (open_b, close_b) in enumerate(bracket_pairs):
            conflict = False
            for (pi, pj) in used_pairs:
                if is_crossing(i, j, pi, pj) and b_idx == bracket_usage[pi]:
                    conflict = True
                    break
            if not conflict:
                structure[i] = brackets[open_b]
                structure[j] = brackets[close_b]
                paired[i], paired[j] = j, i
                used_pairs.add((i,j))
                bracket_usage[i] = b_idx
                bracket_usage[j] = b_idx
                assigned = True
                break
        if not assigned:
            structure[i] = '('
            structure[j] = ')'
            paired[i], paired[j] = j, i

    return ''.join(structure)

def export_ct_file(seq, dot_bracket, filepath):
    L = len(seq)
    pair_map = [-1] * L

    bracket_pairs = {'(': ')', '[': ']', '{': '}', '<': '>'}
    stacks = {k: [] for k in bracket_pairs.keys()}

    for i, ch in enumerate(dot_bracket):
        if ch in bracket_pairs:
            stacks[ch].append(i)
        else:
            for open_b, close_b in bracket_pairs.items():
                if ch == close_b:
                    if stacks[open_b]:
                        j = stacks[open_b].pop()
                        pair_map[i] = j + 1
                        pair_map[j] = i + 1

    with open(filepath, 'w') as f:
        f.write(f"{L} RNA secondary structure\n")
        for i in range(L):
            prev_nt = i if i > 0 else 0
            next_nt = i + 2 if i < L - 1 else 0
            paired_nt = pair_map[i] if pair_map[i] != -1 else 0
            f.write(f"{i+1} {seq[i]} {prev_nt} {next_nt} {paired_nt} {i+1}\n")

def evaluate_model(checkpoint_path, test_fasta, test_dbn_dir, max_len=400, num_samples=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNABasePairPredictor().to(device)
    print("📦 Loading model weights...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    print("📥 Loading test data...")
    dataset = RNADataset(test_fasta, test_dbn_dir, max_len=max_len, max_samples=num_samples)

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    results = {th: {"precision": 0, "recall": 0, "f1": 0, "accuracy": 0} for th in thresholds}

    export_dir = "predictions_ct"
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    for idx in range(len(dataset)):
        seq_tensor, label_tensor = dataset[idx]
        input_tensor = seq_tensor.unsqueeze(0).to(device)
        label_matrix = label_tensor.to(device)
        seq_str = ''.join(['A', 'U', 'G', 'C'][np.argmax(seq_tensor.numpy(), axis=1)[i]] for i in range(seq_tensor.shape[0]))

        with torch.no_grad():
            output = model(input_tensor, [seq_tensor.shape[0]])
            output = output[0].cpu()

        for th in thresholds:
            metrics = evaluate_base_pair_prediction(label_matrix.cpu(), output, threshold=th)
            for k in results[th]:
                results[th][k] += metrics[k]

        # Visualize predicted heatmap for threshold 0.7 only (or any you prefer)
        if idx < 5:  # visualize only first few samples
            visualize_matrix(output.numpy(), idx=idx)

        # Convert to pseudoknot dot-bracket with threshold 0.7 for export and printing
        pred_structure = prob_matrix_to_pseudoknot_dot_bracket(output.numpy(), threshold=0.7)
        print(f"\n🔠 Predicted pseudoknot-aware structure (dot-bracket) for Sample {idx + 1}:\n{pred_structure}\n")

        # Export to .ct file for visualization tools
        ct_path = os.path.join(export_dir, f"sample_{idx+1}.ct")
        export_ct_file(seq_str, pred_structure, ct_path)
        print(f"💾 Exported .ct file for Sample {idx + 1} to: {ct_path}")

    # Average metrics and print
    for th in thresholds:
        for k in results[th]:
            results[th][k] /= len(dataset)
        print(f"Threshold {th:.1f} — Precision: {results[th]['precision']:.4f}, Recall: {results[th]['recall']:.4f}, "
              f"F1: {results[th]['f1']:.4f}, Accuracy: {results[th]['accuracy']:.4f}")

if __name__ == "__main__":
    evaluate_model("best_model.pt", "bpRNA_1m_90.fasta", "bpRNA_1m_90_DBNFILES")
