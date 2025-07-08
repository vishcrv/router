import pandas as pd

df = pd.read_json("hf://datasets/databricks/databricks-dolly-15k/databricks-dolly-15k.jsonl", lines=True)

df["prompt"] = df.apply(
    lambda row: row["instruction"].strip() + ("\n" + row["context"].strip() if row["context"] else ""),
    axis=1
)

# Filter: keep prompts that are "question-like"
query_keywords = ["what", "who", "when", "where", "how", "why", "explain", "describe", "write", "create", "generate"]
df_filtered = df[df["instruction"].str.lower().str.startswith(tuple(query_keywords))]

# Reset index
df_filtered = df_filtered.reset_index(drop=True)

# View sample
print(df_filtered[["instruction", "context", "prompt"]].head())

# Optionally save
df_filtered[["prompt"]].to_json("filtered_prompts.json", lines=True, orient="records")

