import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import leidenalg
import igraph as ig
from collections import defaultdict
from collections import Counter
import numpy as np
from node2vec import Node2Vec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def train_graph_randomforestclassifier(G, node_family_dict):
    # Определение класса для каждой компоненты
    remaining_components = list(nx.connected_components(G))
    component_classes = []
    component_nodes = []

    for component in remaining_components:
        families = [node_family_dict[node] for node in component if
                    node in node_family_dict and not pd.isna(node_family_dict[node])]
        if families:
            # Определение класса компоненты
            most_common_class = Counter(families).most_common(1)[0][0]
            component_classes.append(most_common_class)
            component_nodes.append(component)

    node2vec = Node2Vec(G, dimensions=100, walk_length=80, num_walks=80, workers=10)

    # Обучение модели Node2Vec
    model = node2vec.fit(window=10, min_count=1, batch_words=200)

    # Получение эмбеддингов для узлов
    node_embeddings = {str(node): model.wv[str(node)] for node in G.nodes if str(node) in model.wv}

    # Создание набора данных на уровне узлов
    X = []
    y = []

    for component, component_label in zip(component_nodes, component_classes):
        for node in component:
            if str(node) in node_embeddings:
                X.append(node_embeddings[str(node)])  # Добавляем эмбеддинг узла
                y.append(component_label)  # Метка компоненты для узла

    # Преобразование в массивы numpy
    X = np.array(X)
    y = np.array(y)

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Обучение классификатора
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Прогнозирование и оценка
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    return clf
