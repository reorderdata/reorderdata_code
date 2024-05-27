import numpy as np


def load_large_array(file_prefix, dtype=np.float16):
    chunks = []
    i = 0
    while True:
        filename = f"{file_prefix}_{i}.npz"
        try:
            chunk = np.load(filename)["matrices"].astype(dtype)
            chunks.append(chunk)
            print(f"Loaded chunk {i} from {filename}")
            i += 1
        except FileNotFoundError:
            break
    return np.concatenate(chunks, axis=0)


def extract_scores(pred_comb, pattern_type):
    if pattern_type == "block":
        return pred_comb[:, 0]
    elif pattern_type == "star":
        return pred_comb[:, 1]
    elif pattern_type == "offblock":
        return pred_comb[:, 2]
    elif pattern_type == "band":
        return pred_comb[:, 3]
