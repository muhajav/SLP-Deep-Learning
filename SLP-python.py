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

Per the validation worksheet: "in validation, bias and teta are obtained from
training for each epoch" - so each epoch trains on the 80 training rows, then
evaluates the 20 validation rows with the weights that epoch ended on. The
validation pass never updates the weights.

Both splits are read straight from the CSV exports of the workbook, so the
numbers here track whatever the spreadsheet actually contains.
"""

import numpy as np
import pandas as pd

TRAIN_CSV = "SLP - Muhammad Javier.xlsx - SLP.csv"
VAL_CSV = "validation-SLP.csv"

X_COLS = [2, 3, 4, 5]        # X1, X2, X3, X4
Y_COL = 6                    # TARGET


# --------------------------------------------------------------------------
# 1. Data
# --------------------------------------------------------------------------
def _read_block(csv_path, first_row, n_samples):
    """Pull one contiguous block of samples out of a sheet export."""
    df = pd.read_csv(csv_path, header=None)
    block = df.loc[first_row: first_row + n_samples - 1]
    X = block[X_COLS].astype(float).to_numpy()
    y = block[Y_COL].astype(float).to_numpy()
    return X, y


def load_training_samples(csv_path=TRAIN_CSV, n_samples=80):
    """The training sheet repeats the SAME 80 samples every epoch, so we only
    need to read them once (out of the epoch-1 block, which starts on row 5)."""
    return _read_block(csv_path, 4, n_samples)


def load_validation_samples(csv_path=VAL_CSV, n_samples=20):
    """Likewise for the validation sheet, whose epoch-1 block starts on row 7."""
    return _read_block(csv_path, 6, n_samples)


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

    def train_one_epoch(self, X, y):
        """One pass over the data, updating weights sample-by-sample
        (online gradient descent), exactly like the spreadsheet. Returns the
        predictions and squared errors recorded *as the epoch ran*, which is
        what the sheet's per-epoch metric cells summarise."""
        predictions = np.empty(len(y), dtype=int)
        sq_errors = np.empty(len(y), dtype=float)

        for i, (x, target) in enumerate(zip(X, y)):
            z = self.bias + np.dot(self.weights, x)
            output = self._sigmoid(z)
            predictions[i] = 1 if output >= 0.5 else 0
            sq_errors[i] = (output - target) ** 2

            error = output - target                      # e
            deriv = output * (1 - output)                # g'(z)
            delta = 2 * error * deriv                    # d(SSE)/dz

            self.bias -= self.lr * delta                 # bias "input" = 1
            self.weights -= self.lr * delta * x

        return predictions, sq_errors

    def evaluate_split(self, X, y):
        """Forward pass only, using the weights training just left behind.
        Used for the validation split - no weight updates here."""
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
def main(train_csv=TRAIN_CSV, val_csv=VAL_CSV, n_epochs=5, learning_rate=0.1):
    X_train, y_train = load_training_samples(train_csv)
    X_val, y_val = load_validation_samples(val_csv)

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
    print("  training loss : [0.450203, 0.037529, 0.024418, 0.017389, 0.012764]")
    print("  training F1   : [0.672414, 0.950000, 0.975000, 0.975000, 0.987342]")
    print("  val. loss     : [0.329163, 0.247463, 0.176014, 0.119424, 0.081583]")

    return model, df


if __name__ == "__main__":
    main()
