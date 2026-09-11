"""
Single Layer Perceptron - Iris binary classification (Setosa vs Versicolour)

Python counterpart to the SLP Google Sheets workbook, which keeps training and
validation on two separate worksheets:

    SLP-training-new.csv    80 samples, weights updated after every sample
    SLP-validation-new.csv  20 held-out samples, forward pass only

Model, matching the spreadsheet exactly:
  - 4 inputs + bias, every weight initialised to 0.5
  - sigmoid activation, threshold 0.5
  - loss = Sum Square Error, averaged over the split (MSE)
  - online / stochastic gradient descent, learning rate 0.1
  - 5 epochs, weights carried over between epochs

As the validation worksheet notes, "in validation, bias and teta are obtained
from training for each epoch": each epoch trains on the 80 training rows, then
scores the 20 validation rows with the weights that epoch ended on. The
validation pass never updates the weights.

Both splits are read from the CSV exports, so these numbers track whatever the
workbook actually contains.
"""

import numpy as np
import pandas as pd

TRAIN_CSV = "SLP-training-new.csv"
VAL_CSV = "SLP-validation-new.csv"

X_COLS = [2, 3, 4, 5]        # X1, X2, X3, X4
Y_COL = 6                    # TARGET

# First data row of each epoch block, 0-indexed, as the two sheets are laid out.
TRAIN_EPOCH_ROW = {1: 4, 2: 84, 3: 184, 4: 284, 5: 384}
VAL_EPOCH_ROW = {1: 6, 2: 26, 3: 46, 4: 146, 5: 246}

N_TRAIN, N_VAL = 80, 20


# --------------------------------------------------------------------------
# 1. Data
# --------------------------------------------------------------------------
def _read_block(csv_path, first_row, n_samples):
    """Pull one contiguous block of samples out of a worksheet export."""
    df = pd.read_csv(csv_path, header=None)
    block = df.loc[first_row: first_row + n_samples - 1]
    X = block[X_COLS].astype(float).to_numpy()
    y = block[Y_COL].astype(float).to_numpy()
    return X, y


def load_training_samples(csv_path=TRAIN_CSV):
    """Every epoch block repeats the same 80 samples, so read epoch 1 only."""
    return _read_block(csv_path, TRAIN_EPOCH_ROW[1], N_TRAIN)


def load_validation_samples(csv_path=VAL_CSV):
    """Likewise the 20 held-out samples, from the validation sheet."""
    return _read_block(csv_path, VAL_EPOCH_ROW[1], N_VAL)


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
        """One pass over the training sheet, updating weights sample-by-sample.
        Returns the predictions and squared errors recorded *as the epoch ran*,
        which is what the sheet's per-epoch metric cells summarise."""
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
        """Forward pass only, on the weights training just left behind.
        This is the validation sheet - no weight updates here."""
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
    print("\nFinal bias   :", round(model.bias, 10))
    print("Final weights:", np.round(model.weights, 10))

    return model, df


if __name__ == "__main__":
    main()
