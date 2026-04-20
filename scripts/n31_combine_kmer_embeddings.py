import pandas as pd
files = ["/Users/nad/hse/semester08/mobiraph/data/n17_repbase_kmer/4.csv", "/Users/nad/hse/semester08/mobiraph/data/n17_repbase_kmer/5.csv"]

general_df = pd.DataFrame()
for file in files:
    df = pd.read_csv(file)
    if general_df.empty:
        general_df = df
    else:
        general_df = pd.merge(general_df, df, on='name', how='left')

general_df.to_csv("/Users/nad/hse/semester08/mobiraph/data/n17_repbase_kmer/4_5.csv", index=False)