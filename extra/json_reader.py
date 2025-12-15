import json
import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter

# Load JSON results
json_path = "/usr/local/data/elise/pig_data/pig2/Registration/Known_Trans/intra1/output_python_cma_group_allcases/grid_search_results.json"
with open(json_path, "r") as f:
    all_results = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(all_results)

# Ensure numeric columns are numbers (flatten lists if needed)
def flatten_param(x):
    if isinstance(x, list):
        return float(np.mean(x))  # use mean if list
    return float(x)

param_cols = ['sigma0','popsize','parents','base_stds','lower','upper']
for col in param_cols:
    df[col] = df[col].apply(flatten_param)

# Extract TRE per case
cases = ['L1', 'L2', 'L3', 'L4']
for case in cases:
    df[case] = df['tre'].apply(lambda x: x.get(case, np.nan) if x else np.nan)

# Mean TRE across cases
df['mtre'] = df[cases].mean(axis=1)

# Top 20 configurations
topN = 20
df_topN = df.sort_values(by='mtre').head(topN)

# Print top configurations
print(f"Top {topN} CMA configurations by mean TRE:\n")
print(df_topN[['mtre'] + param_cols].round(3).to_string(index=False))

print()

df = pd.DataFrame(all_results)

# Top 20 by mean TRE across L1-L4
cases = ['L1', 'L2', 'L3', 'L4']
for case in cases:
    df[case] = df['tre'].apply(lambda x: x.get(case, None) if x else None)
df['mtre'] = df[cases].apply(lambda row: pd.Series(row).dropna().mean(), axis=1)

topN = 20
df_topN = df.sort_values(by='mtre').head(topN)

# Convert list columns to tuples for counting
list_cols = ['lower','upper','base_stds']
for col in list_cols:
    df_topN[col + '_tuple'] = df_topN[col].apply(lambda x: tuple(x) if isinstance(x, list) else tuple([x]))

# Count occurrences
for col in list_cols:
    counts = Counter(df_topN[col + '_tuple'])
    most_common, freq = counts.most_common(1)[0]
    print(f"Most common {col} in top {topN} runs (appeared {freq} times): {most_common}")
