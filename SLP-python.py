"""
Single Layer Perceptron - Iris binary classification (Setosa vs Versicolour)

Faithful re-implementation of the Google Sheets SLP workbook:
  - 4 inputs + bias, all weights initialised to 0.5
  - sigmoid activation, threshold 0.5
  - loss = Sum Square Error, averaged over the split (MSE)
  - online / stochastic gradient descent: weights updated after every sample
  - learning rate 0.1, 5 epochs, weights carried over between epochs

Split follows the lecture slides: 80 training samples (first 40 of each class),
20 validation samples (last 10 of each class).
"""

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# 1. Data
# --------------------------------------------------------------------------
# The training rows live in the CSV export of the sheet. The sheet repeats the
# SAME 80 samples every epoch, so we only read them once (the epoch-1 block).
def load_training_samples(csv_path: str, n_samples: int = 80):
    df = pd.read_csv(csv_path, header=None)

    data_start_row = 4           # first data row (0-indexed) in the CSV
    x_cols = [2, 3, 4, 5]        # X1, X2, X3, X4
    y_col = 6                    # TARGET

    block = df.loc[data_start_row: data_start_row + n_samples - 1]
    X = block[x_cols].astype(float).to_numpy()
    y = block[y_col].astype(float).to_numpy()
    return X, y


# The 20 held-out samples (Iris rows 41-50 and 91-100). X4 is held at 0.2 to
# stay consistent with the training block in the sheet.
VALIDATION = [
    (5.0, 3.5, 1.3, 0.2, 0), (4.5, 2.3, 1.3, 0.2, 0), (4.4, 3.2, 1.3, 0.2, 0),
    (5.0, 3.5, 1.6, 0.2, 0), (5.1, 3.8, 1.9, 0.2, 0), (4.8, 3.0, 1.4, 0.2, 0),
    (5.1, 3.8, 1.6, 0.2, 0), (4.6, 3.2, 1.4, 0.2, 0), (5.3, 3.7, 1.5, 0.2, 0),
    (5.0, 3.3, 1.4, 0.2, 0), (5.5, 2.6, 4.4, 0.2, 1), (6.1, 3.0, 4.6, 0.2, 1),
    (5.8, 2.6, 4.0, 0.2, 1), (5.0, 2.3, 3.3, 0.2, 1), (5.6, 2.7, 4.2, 0.2, 1),
    (5.7, 3.0, 4.2, 0.2, 1), (5.7, 2.9, 4.2, 0.2, 1), (6.2, 2.9, 4.3, 0.2, 1),
    (5.1, 2.5, 3.0, 0.2, 1), (5.7, 2.8, 4.1, 0.2, 1),
]


def load_validation_samples():
    arr = np.array(VALIDATION, dtype=float)
    return arr[:, :4], arr[:, 4]


# --------------------------------------------------------------------------
# 2. The perceptron, matching the sheet's exact update rule
# --------------------------------------------------------------------------
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
        (online gradient descent), exactly like the spreadsheet. Returns the
        predictions and squared errors recorded *as the epoch ran*, which is
        what the sheet's per-epoch metric cells summarise."""
        predictions = np.empty(len(y), dtype=int)
        sq_errors = np.empty(len(y), dtype=float)

        for i, (x, target) in enumerate(zip(X, y)):
            z, output = self._forward(x)
            predictions[i] = 1 if output >= 0.5 else 0
            sq_errors[i] = (output - target) ** 2

            error = output - target                      # e
            deriv = output * (1 - output)                # g'(z)
            delta = 2 * error * deriv                    # d(SSE)/dz

            self.bias -= self.lr * delta                 # bias "input" = 1
            self.weights -= self.lr * delta * x

        return predictions, sq_errors

    def evaluate_split(self, X, y):
        """Forward pass only - used for the validation split."""
        outputs = self._sigmoid(self.bias + X @ self.weights)
        predictions = (outputs >= 0.5).astype(int)
        return predictions, (outputs - y) ** 2


# --------------------------------------------------------------------------
# 3. Metrics
# --------------------------------------------------------------------------
def metrics(y_true, y_pred, sq_errors):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)

    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "accuracy": (tp + tn) / len(y_true),
            "loss": float(np.mean(sq_errors)),      # MSE
            "precision": precision, "recall": recall, "f1": f1}


# --------------------------------------------------------------------------
# 4. Run it
# --------------------------------------------------------------------------
def main(csv_path="SLP - Muhammad Javier.xlsx - SLP.csv",
         n_epochs=5, learning_rate=0.1):
    X_train, y_train = load_training_samples(csv_path)
    X_val, y_val = load_validation_samples()

    model = SingleLayerPerceptron(n_features=X_train.shape[1],
                                  learning_rate=learning_rate)

    print(f"Training   : {len(y_train)} samples "
          f"({int(np.sum(y_train == 0))} Setosa / "
          f"{int(np.sum(y_train == 1))} Versicolour)")
    print(f"Validation : {len(y_val)} samples "
          f"({int(np.sum(y_val == 0))} Setosa / "
          f"{int(np.sum(y_val == 1))} Versicolour)\n")

    history = []
    for epoch in range(1, n_epochs + 1):
        tr_pred, tr_sq = model.train_one_epoch(X_train, y_train)
        va_pred, va_sq = model.evaluate_split(X_val, y_val)

        tr = metrics(y_train, tr_pred, tr_sq)
        va = metrics(y_val, va_pred, va_sq)

        history.append({
            "epoch": epoch,
            "train_accuracy": tr["accuracy"], "val_accuracy": va["accuracy"],
            "train_loss": tr["loss"],         "val_loss": va["loss"],
            "train_f1": tr["f1"],             "val_f1": va["f1"],
        })

        print(f"Epoch {epoch}: "
              f"acc(train)={tr['accuracy']:.4f}  acc(val)={va['accuracy']:.4f}  "
              f"loss(train)={tr['loss']:.6f}  loss(val)={va['loss']:.6f}  "
              f"F1(train)={tr['f1']:.4f}")

    df = pd.DataFrame(history)
    print("\n" + df.round(6).to_string(index=False))
    print("\nFinal bias   :", round(model.bias, 6))
    print("Final weights:", np.round(model.weights, 6))
    print("\nGoogle Sheets reference")
    print("  training loss : [0.449439, 0.051185, 0.033209, 0.023076, 0.017130]")
    print("  training F1   : [0.666667, 0.936709, 0.950000, 0.975000, 0.975000]")

    return model, df


if __name__ == "__main__":
    main()
