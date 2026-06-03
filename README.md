# Mobiraph

The earlier implementation of this project can be found in a separate repository:

https://github.com/nad-igolki/mobiraph_2025

# How to Use with Docker

## Build

```bash
docker buildx build --platform linux/amd64 -t mobiraph .
````

## Train

This assumes that the training data is already available inside the image at `/app/data`.

```bash
docker run --rm mobiraph train \
  --fasta_file /app/data/train.fasta \
  --metadata_file /app/data/metadata.json \
  --checkpoint_dir /app/checkpoints
```

### Metadata format
A JSON file containing a hierarchical structure of sequence classes.

It is represented as a nested dictionary tree, where each key is a class name, for example `"Non-LTR Retrotransposon"`, and each value is an object with two main fields:

* `"sequences"` — a dictionary or list of sequence indices belonging to this class and its entire subtree;
* `"subs"` — a dictionary of subclasses, each following the same structure and recursively forming a category tree.


## Test

This assumes that the testing data is already available inside the image at `/app/data`.

```bash
docker run --rm mobiraph test \
  --fasta_file /app/data/test.fasta \
  --models_path /app/models \
  --output_file /app/output/predictions.csv \
  --checkpoint_dir /app/checkpoints
```


## Сontributions

- **Nadezhda Igolkina** : Developer
- **Ilya Karpov**: Supervisor, [GitHub](https://github.com/karpovilia)
- **Anna Igolkina**: Supervisor, [GitHub](https://github.com/iganna)

## License
MIT