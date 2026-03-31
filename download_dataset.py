from datasets import load_dataset

dataset = load_dataset(
    "poloclub/diffusiondb",
    name="2m_first_10k",
    split="train[:8000]"
)

print(f"Total rows: {len(dataset)}")
print(f"Columns: {dataset.column_names}")

dataset.save_to_disk("./diffusiondb_8000")
print("Saved!")