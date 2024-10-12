from typing import Any, Optional, List
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, 
                             accuracy_score)
import numpy as np
import pandas as pd
from graphviz import Digraph


class DecisionTreeID3(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth: int):
        self.max_depth = max_depth
        self.tree = None
        self.feature_idx = None
        self.children = []
        self.class_label = None
        self.feature_names = None
        self.X = None

    def calculate_entropy(self, labels: np.ndarray[Any, Any]) -> float:
        _, counts = np.unique(labels, return_counts=True)
        proba = counts / counts.sum()
        return -np.sum(proba * np.log2(proba))

    def calculate_gain(
        self, X: np.ndarray, y: np.ndarray, feature_idx: int
    ) -> float:
        total_length = len(y)
        feature_values = X[:, feature_idx]
        weighted_entropy = 0.0

        for val in np.unique(feature_values):
            subset_y = y[feature_values == val]
            subset_length = len(subset_y)
            subset_entropy = self.calculate_entropy(subset_y)
            weighted_entropy += subset_entropy * subset_length

        gain = (1 / total_length) * weighted_entropy
        return gain

    def select_feature(self, X: np.ndarray, y: np.ndarray) -> int:
        gains = [self.calculate_gain(X, y, i) for i in range(X.shape[1])]
        return np.argmin(gains)

    def build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int,
        feature_indices: List[int],
        feature_names: List[str],
    ) -> "DecisionTreeID3":
        if (
            len(np.unique(y)) == 1
            or depth >= self.max_depth
            or len(feature_indices) == 0
        ):
            leaf = DecisionTreeID3(max_depth=self.max_depth)
            leaf.class_label = np.unique(y)[0]
            return leaf

        feature_idx = self.select_feature(X[:, feature_indices], y)
        selected_feature_idx = feature_indices[feature_idx]

        node = DecisionTreeID3(max_depth=self.max_depth)
        node.feature_idx = selected_feature_idx

        remaining_features = feature_indices.copy()
        remaining_features.pop(feature_idx)

        for val in np.unique(X[:, selected_feature_idx]):
            subset_mask = X[:, selected_feature_idx] == val
            subset_X = X[subset_mask]
            subset_y = y[subset_mask]

            child_node = self.build_tree(
                subset_X, subset_y, depth + 1, remaining_features, feature_names
            )
            node.children.append((val, child_node))

        return node

    def fit(
        self, X: np.ndarray, y: np.ndarray, feature_names: List[str]
    ) -> "DecisionTreeID3":
        self.feature_names = feature_names
        self.X = X
        self.y = y
        self.tree = self.build_tree(
            X, y, 0, list(range(X.shape[1])), feature_names
        )

        return self

    def predict(self, X: np.ndarray) -> List[Any]:
        predictions = []
        for instance in X:
            node = self.tree
            while node.class_label is None:
                feature_val = instance[node.feature_idx]
                found_child = False
                for val, child in node.children:
                    if val == feature_val:
                        node = child
                        found_child = True
                        break
                if not found_child:
                    class_counts = np.unique(self.y[self.X[:, node.feature_idx] == feature_val], return_counts=True)
                    node.class_label = class_counts[0][np.argmax(class_counts[1])]
                    break
            predictions.append(node.class_label)
        return predictions

    def visualize_tree(
        self,
        node: Optional["DecisionTreeID3"] = None,
        graph: Optional[Digraph] = None,
        X=None,
        y=None,
        parent_condition=None,
    ) -> Digraph:
        if graph is None:
            graph = Digraph()
            graph.attr(size="100,100")
        if node is None:
            node = self.tree
            X = self.X
            y = self.y

        if node.feature_idx is not None:
            node_label = f"{self.feature_names[node.feature_idx]}"

            node_label += f"\nTotal Count: {len(y)}"

            class_counts = np.unique(y, return_counts=True)
            for class_val, count in zip(class_counts[0], class_counts[1]):
                node_label += f"\n   Class {class_val}: {count}"
            for val, child in node.children:
                current_condition = (
                    f"{self.feature_names[node.feature_idx]} == '{val}'"
                )
                if parent_condition:
                    current_condition = (
                        f"{parent_condition} and {current_condition}"
                    )

                subset_mask = X[:, node.feature_idx] == val
                subset_X = X[subset_mask]
                subset_y = y[subset_mask]

                node_label += (
                    f"\nEntropy {val}: {self.calculate_entropy(subset_y):.3f}"
                )

                graph.edge(str(id(node)), str(id(child)), label=str(val))

                self.visualize_tree(
                    child,
                    graph,
                    subset_X.copy(),
                    subset_y.copy(),
                    current_condition,
                )

            node_label += (
                f"\nIG: {self.calculate_gain(X, y, node.feature_idx):.3f}"
            )
        else:
            node_label = f"Class: {node.class_label}"

        graph.node(str(id(node)), label=node_label, shape='box')  

        return graph


if __name__ == "__main__":
        # Пример использования
    df = pd.read_excel("./P4/Credits.xlsx")

    X = df.drop(columns="risk", axis=1).values
    y = df["risk"].values
    feature_names = df.drop(columns="risk", axis=1).columns.tolist()

    cls = DecisionTreeID3(max_depth=256)

    cls.fit(X, y, feature_names)

    # pred = cls.predict(X_test)

    dot = cls.visualize_tree()
    dot.render("./P4/decision_tree", format="png", cleanup=True)

    pred = cls.predict(
        np.array([["Х", "Н", "Н", "0-15"], ["Х", "Н", "Н", "15-35"]])
    )
    print(pred)
