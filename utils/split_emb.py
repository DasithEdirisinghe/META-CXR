import os
import pickle
from pathlib import Path

# Path to the original large pickle file
input_path = "pretraining/embs/2025_05_20_blip2_pretrain_stage1_emb_embeddings_iu_xray_test.pkl"

# Output directory for individual pickles
output_dir = "pretraining/embs/split_iu_xray_test"
os.makedirs(output_dir, exist_ok=True)

# Load the full dictionary (key = DICOM ID, value = embedding array)
with open(input_path, "rb") as f:
    blip_embeddings = pickle.load(f)

# Save each entry separately
for dicom_id, embedding in blip_embeddings.items():
    output_file = os.path.join(output_dir, f"{dicom_id}.pkl")
    with open(output_file, "wb") as out_f:
        pickle.dump(embedding, out_f)

print(f"✅ Saved {len(blip_embeddings)} individual embedding files to: {output_dir}")
