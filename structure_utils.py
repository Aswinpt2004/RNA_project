import numpy as np

def dot_bracket_to_matrix(dot_bracket):
    L = len(dot_bracket)
    matrix = [[0] * L for _ in range(L)]
    pairs = {'(': ')', '[': ']', '{': '}', '<': '>'}
    stacks = {k: [] for k in pairs}
    for i, sym in enumerate(dot_bracket):
        if sym in stacks:
            stacks[sym].append(i)
        else:
            for op, cl in pairs.items():
                if sym == cl and stacks[op]:
                    j = stacks[op].pop()
                    matrix[i][j] = matrix[j][i] = 1
                    break
    return matrix

def evaluate_base_pair_prediction(true_matrix, pred_matrix, threshold=0.5):
    pred_bin = (pred_matrix > threshold).int().cpu().numpy()
    true_bin = np.array(true_matrix)
    TP = np.sum((pred_bin == 1) & (true_bin == 1))
    FP = np.sum((pred_bin == 1) & (true_bin == 0))
    FN = np.sum((pred_bin == 0) & (true_bin == 1))
    TN = np.sum((pred_bin == 0) & (true_bin == 0))
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}
