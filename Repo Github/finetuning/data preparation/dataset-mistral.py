import pandas as pd
import json

csv_file = 'dataset-metrics-fin.csv'
try:
    df = pd.read_csv(csv_file, header=None, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(csv_file, header=None, encoding='latin1')

data = df.to_dict(orient='records')



df_formatted = [
    {
        "messages": [
            {"role": "user", "content": item[0]},
            {"role": "assistant", "content": item[1]},
        ]
    }
    for item in df.iloc
]


jsonl_file = 'dataset-metrics-fin.jsonl'
with open(jsonl_file, 'w') as f:
    for line in df_formatted:
        json.dump(line, f)
        f.write("\n")

print(f"Conversion complete. JSONL file saved as '{jsonl_file}'.")