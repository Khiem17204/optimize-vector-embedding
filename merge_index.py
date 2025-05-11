import os
import faiss
import pickle
import numpy as np

# === Config ===
BASE_DIR = "datasets/fever"
PARTS = [i for i in range(50)] 
print(PARTS)
OUTPUT_DIR = os.path.join(BASE_DIR, "index")



# === Load vecs from part0 and part1 ===
for part_id in PARTS:
    part_dir = os.path.join(BASE_DIR, f"index_part_{part_id}")
    vecs_path = os.path.join(part_dir, "vecs")
    
    if not os.path.exists(vecs_path):
        print(f"[!] vecs file missing in part {part_id}, skipping.")
        continue

    print(f"[+] Loading vectors from: {vecs_path}")
    with open(vecs_path, "rb") as f:
        vecs = pickle.load(f)

    all_vecs.append(vecs)

# === Concatenate vectors ===
print(f"[+] Concatenating {len(all_vecs)} parts...")
merged_vecs = np.vstack(all_vecs)
dim = merged_vecs.shape[1]
print(f"[✓] Total vectors: {merged_vecs.shape[0]} | Dimension: {dim}")

# === Build FAISS index ===
print("[+] Creating FAISS Flat index...")
index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
index.add(merged_vecs)

# === Save merged index ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
index_path = os.path.join(OUTPUT_DIR, "index")
vecs_path = os.path.join(OUTPUT_DIR, "vecs")

faiss.write_index(index, index_path)
with open(vecs_path, "wb") as f:
    pickle.dump(merged_vecs, f)

print(f"[✓] Merged test index saved to: {OUTPUT_DIR}")
