from datasets import Dataset, load_dataset, concatenate_datasets

with open('hf_token.txt', 'r') as f:
    hf_token = f.read().strip()

wmt_ds = load_dataset("wmt/wmt14", name = "de-en", split="train", token=hf_token).shuffle(seed=42)
opus_ds = load_dataset("Helsinki-NLP/opus-100", name = "de-en", split="train", token=hf_token).shuffle(seed=42)

# select 1000 random samples for BLEU evaluation, wmt mostly contains professional translations.
bleu_ds = wmt_ds.select([i + 1000000 for i in list(range(1000))])
bleu_ds.to_parquet("combined-de-en-1000k/bleu.parquet")

wmt_subset = wmt_ds.select(range(750000))
opus_subset = opus_ds.select(range(250000))

print(len(opus_subset))
print(len(wmt_subset))
combined_ds = concatenate_datasets([wmt_subset, opus_subset])
combined_ds.to_parquet("combined-de-en-1000k/train.parquet")

ds_sub = load_dataset("parquet", data_files="combined-de-en-1000k/train.parquet")
print(len(ds_sub['train']))

