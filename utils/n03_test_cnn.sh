python -m scripts.n19_test_kmer_cnn
--embeddings-path "/Users/nad/mobiraph/data/n12_all_sequences_kmer/7.csv"
--metadata-path "/Users/nad/mobiraph/data/n13_repbase_processed/hierarchy_sequences_02_ltr_correction.json"
--test-ids "/Users/nad/mobiraph/data/n13_repbase_processed/id_test.txt"
--model-path "/Users/nad/mobiraph/data/n20_kmer_models/DNA transposon/20_epochs/cnn_model.keras"
--label-encoder-path "/Users/nad/mobiraph/data/n20_kmer_models/DNA transposon/20_epochs/label_encoder.pkl"
--plots-dir "/Users/nad/mobiraph/figures/kmer_cnn/DNA\ transposon/"
--hierarchy-root "DNA transposon"


python -m scripts.n19_test_kmer_cnn \
--embeddings-path "/Users/nad/mobiraph/data/n12_all_sequences_kmer/7.csv" \
--metadata-path "/Users/nad/mobiraph/data/n13_repbase_processed/hierarchy_sequences_02_ltr_correction.json" \
--test-ids "/Users/nad/mobiraph/data/n13_repbase_processed/id_test.txt" \
--model-path "/Users/nad/mobiraph/data/n20_kmer_models/DNA transposon/30_epochs/cnn_model.keras" \
--label-encoder-path "/Users/nad/mobiraph/data/n20_kmer_models/DNA transposon/30_epochs/label_encoder.pkl" \
--plots-dir "/Users/nad/mobiraph/figures/kmer_cnn/DNA transposon/30_epochs" \
--hierarchy-root "DNA transposon"
