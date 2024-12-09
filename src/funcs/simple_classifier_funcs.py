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


def get_embeddings(G, dimensions=50):
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=80, num_walks=80, workers=10)
    model = node2vec.fit(window=10, min_count=1, batch_words=200)
    # Получение эмбеддингов для узлов
    node_embeddings = {str(node): model.wv[str(node)] for node in G.nodes if str(node) in model.wv}
    return node_embeddings

def train_graph_randomforestclassifier2(G, node_embeddings, node_embeddings2, node_family_dict):
    # Определение класса для каждой компоненты
    remaining_components = list(nx.connected_components(G))
    component_classes = []
    component_nodes = []

    for component in remaining_components:
        if len(component) < 4:
            continue
        families = [node_family_dict[node] for node in component if
                    node in node_family_dict and not pd.isna(node_family_dict[node])]
        if families:
            # Определение класса компоненты
            most_common_class = Counter(families).most_common(1)[0][0]
            component_classes.append(most_common_class)
            component_nodes.append(component)
    print(len(component_nodes))


    nodes_train, nodes_test, classes_train, classes_test = train_test_split(component_nodes, component_classes, test_size=0.3, random_state=42)
    # Создание набора данных на уровне узлов
    X_train, X_test = [], []
    y_train, y_test = [], []

    for component, component_label in zip(nodes_train, classes_train):
        embeddings_list = None
        cnt = 0
        for node in component:
            # print('node:', type(node_embeddings[str(node)]))
            if str(node) in node_embeddings:
                # print(len(np.array(node_embeddings[str(node)]).flatten()))
                if embeddings_list is None:
                    embeddings_list = np.array(node_embeddings[str(node)]).flatten()
                else:
                    embeddings_list += np.array(node_embeddings[str(node)]).flatten()
                cnt += 1
        embeddings_list /= cnt
        X_train.append(embeddings_list)  # Добавляем эмбеддинги узла
        y_train.append(component_label)  # Метка компоненты

    for component, component_label in zip(nodes_test, classes_test):
        embeddings_list = None
        cnt = 0
        for node in component:
            if str(node) in node_embeddings:
                if embeddings_list is None:
                    embeddings_list = np.array(node_embeddings2[str(node)]).flatten()
                else:
                    embeddings_list += np.array(node_embeddings2[str(node)]).flatten()
                cnt += 1
            else:
                continue
        embeddings_list /= cnt
        X_test.append(embeddings_list)  # Добавляем эмбеддинг узла
        y_test.append(component_label)  # Метка компоненты

    print(len(y_train), y_train.count('Helitron'), y_train.count('LINE'))

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    # Обучение классификатора
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Результат
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    data = pd.DataFrame({'Array1': y_test, 'Array2': y_pred})

    # Создание таблицы сопряжённости
    contingency_table = pd.crosstab(data['Array1'], data['Array2'])

    print(contingency_table)

    return clf


def train_graph_randomforestclassifier(node_embeddings, node_family_dict):
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

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    return clf