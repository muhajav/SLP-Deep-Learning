import numpy as np
import pandas as pd


# 1. Load the 80 underlying samples out of the CSV export of the sheet
def load_samples(csv_path: str, n_samples: int = 80):
    """The sheet repeats the SAME 80 samples every epoch, so we only
    need to read them once (out of the epoch-1 block)."""
    df = pd.read_csv(csv_path, header=None)

    data_start_row = 4          # first data row (0-indexed) in the CSV
    x_cols = [2, 3, 4, 5]        # X1, X2, X3, X4
    y_col = 6                    # TARGET

    block = df.loc[data_start_row: data_start_row + n_samples - 1]
    X = block[x_cols].astype(float).to_numpy()
    y = block[y_col].astype(float).to_numpy()
    return X, y


# 2. The perceptron itself, matching the sheet's exact update rule
class SingleLayerPerceptron:
    def __init__(self, n_features: int, learning_rate: float = 0.1,
                 init_weight: float = 0.5):
        self.lr = learning_rate
        self.bias = init_weight
        self.weights = np.full(n_features, init_weight, dtype=float)

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def _forward(self, x):
        z = self.bias + np.dot(self.weights, x)
        return z, self._sigmoid(z)

    def train_one_epoch(self, X, y):
        """One pass over the data, updating weights sample-by-sample
        (online gradient descent), exactly like the spreadsheet."""
        predictions = np.empty(len(y), dtype=int)

        for i, (x, target) in enumerate(zip(X, y)):
            z, output = self._forward(x)
            predictions[i] = 1 if output >= 0.5 else 0

            error = output - target                     # e
            deriv = output * (1 - output)                # g'(z)
            delta = 2 * error * deriv                    # d(SSE)/dz

            d_bias = delta                                # bias "input" = 1
            d_weights = delta * x

            self.bias -= self.lr * d_bias
            self.weights -= self.lr * d_weights

        return predictions

    def predict(self, X):
        return np.array([1 if self._forward(x)[1] >= 0.5 else 0 for x in X])


# 3. Confusion matrix + Precision / Recall / F1
def evaluate(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)

    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "precision": precision, "recall": recall, "f1": f1}


# 4. Run it: 5 epochs, weights carried over between epochs
def main(csv_path="slp.csv", n_epochs=5, learning_rate=0.1):
    X, y = load_samples(csv_path)
    model = SingleLayerPerceptron(n_features=X.shape[1],
                                   learning_rate=learning_rate)

    print(f"Loaded {len(y)} samples "
          f"({int(np.sum(y == 0))} Setosa / {int(np.sum(y == 1))} Versicolor)\n")

    history = []
    for epoch in range(1, n_epochs + 1):
        preds = model.train_one_epoch(X, y)
        metrics = evaluate(y, preds)
        history.append(metrics["f1"])

        print(f"Epoch {epoch}: "
              f"bias={model.bias:.4f}  "
              f"weights={np.round(model.weights, 4)}  "
              f"TP={metrics['TP']} FP={metrics['FP']} "
              f"TN={metrics['TN']} FN={metrics['FN']}  "
              f"Precision={metrics['precision']:.4f}  "
              f"Recall={metrics['recall']:.4f}  "
              f"F1={metrics['f1']:.4f}")

    print("\nF1-score per epoch:", [round(f, 4) for f in history])
    print("(Spreadsheet reference: [0.6667, 0.9367, 0.95, 0.975, 0.975])")

    return model, history


if __name__ == "__main__":
    main()