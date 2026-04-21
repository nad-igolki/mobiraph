import pandas as pd
files = ["/Users/nad/mobiraph/data/n12_all_sequences_kmer/4.csv", "/Users/nad/mobiraph/data/n12_all_sequences_kmer/5.csv"]

general_df = pd.DataFrame()
for file in files:
    df = pd.read_csv(file)
    if general_df.empty:
        general_df = df
    else:
        general_df = pd.merge(general_df, df, on='name', how='left')

general_df.to_csv("/Users/nad/mobiraph/data/n12_all_sequences_kmer/4_5.csv", index=False)