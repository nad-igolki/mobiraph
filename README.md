# Mobiraph
Construct a graph on the mobilome and annotate it by families of mobile elements.
**Mobiraph** is a future extension on the [Pannagram](https://github.com/iganna/pannagram) package.

[Some other open-source code is located in a separate branch](https://github.com/nad-igolki/mobiraph/tree/dev)

## Features

- Construct the graph on DNA sequences of the mobilome
- Annotate the graph with ML graph-base methods

## Installation

To install all necessary dependencies from a YAML file, use the following command:

```bash
conda env create -f mobigraph.yaml
conda activate mobigraph
```

### Project Directory Structure:

1. **`funcs/`** — Python files (.py) with the most commonly used functions and utilities.

2. **`load_data/`** — Jupyter notebooks (.ipynb) for data loading, processing, and saving.

3. **`models/`** — Python files (.py) and Jupyter notebooks (.ipynb) for model training and validation.

4. **`notebooks/`** — Other Jupyter notebooks (.ipynb), such as:
   - `get_embeddings.ipynb`: Creating embeddings.
   - `characteristics.ipynb`: Calculating graph characteristics.
   - `visualization.ipynb`: Visualizing the graph and its components.

`research_paper.pdf`: Research Paper Overview.

## Сontributions

- **Nadezhda Igolkina** : Developer
- **Ilya Karpov**: Supervisor
- **Anna Igolkina**: Supervisor, [GitHub](https://github.com/iganna)

## License
MIT
