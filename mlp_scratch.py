"""
Multi-Layer Perceptron (MLP) from Scratch using NumPy
======================================================
Task 7.1: Build a 1-hidden-layer neural network without any ML frameworks.

Network Architecture:
    Input Layer  -> Hidden Layer (sigmoid) -> Output Layer (sigmoid)
    Shape: [n_features] -> [n_hidden] -> [n_output]

Math notation used throughout:
    W1, b1  : weights and bias for input -> hidden layer
    W2, b2  : weights and bias for hidden -> output layer
    Z1, A1  : pre-activation and post-activation of hidden layer
    Z2, A2  : pre-activation and post-activation of output layer
    m       : number of training samples
    L       : cross-entropy loss
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =============================================================================
# STEP 1: Dataset Creation
# =============================================================================

def make_xor_dataset(n_samples=200, noise=0.1, random_state=42):
    """
    Generate an XOR classification dataset.

    XOR logic:
        (0,0) -> 0,  (0,1) -> 1,  (1,0) -> 1,  (1,1) -> 0

    We extend to continuous inputs in [0,1]^2 by labeling each point
    based on which quadrant it falls in (XOR of rounded coordinates).

    Returns:
        X : (n_samples, 2) feature matrix
        y : (1, n_samples) label row-vector  (0 or 1)
    """
    rng = np.random.RandomState(random_state)
    X = rng.rand(n_samples, 2)                          # uniform in [0,1]^2
    # XOR label: top-right and bottom-left quadrants => class 1
    y = ((X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)).astype(float)
    # Optional: add Gaussian noise to the features
    X += rng.randn(n_samples, 2) * noise
    return X, y.reshape(1, -1)                          # y shape: (1, m)


def make_iris_binary_dataset():
    """
    Load Iris dataset and reduce to binary classification.
    We keep only setosa vs. versicolor (first 100 samples, 2 classes)
    and use only 2 features for easy 2-D decision-boundary visualization.

    Returns:
        X_train, X_test : (2, m) feature matrices (standardised)
        y_train, y_test : (1, m) label row-vectors
    """
    iris = load_iris()
    # Use only 2 features (sepal length, petal length) for visualisation
    X = iris.data[:100, [0, 2]]
    y = iris.target[:100]                               # 0=setosa, 1=versicolor

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardise: zero mean, unit variance (important for gradient descent)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Reshape to (n_features, m) column-major layout used throughout
    return (X_train.T, X_test.T,
            y_train.reshape(1, -1), y_test.reshape(1, -1),
            scaler)


# =============================================================================
# STEP 2: Weight Initialisation
# =============================================================================

def initialise_parameters(n_input, n_hidden, n_output, random_state=42):
    """
    Initialise network weights using Xavier (Glorot) initialisation.

    Xavier init: W ~ Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))
    This keeps variance of activations roughly constant across layers,
    preventing vanishing / exploding gradients during early training.

    Biases are initialised to zero — this is standard practice because
    symmetry breaking is handled by the random weights.

    Parameters:
        n_input  : number of input features
        n_hidden : number of neurons in the hidden layer
        n_output : number of output neurons (1 for binary classification)

    Returns:
        params : dict with W1, b1, W2, b2
    """
    rng = np.random.RandomState(random_state)

    # --- Layer 1: Input -> Hidden ---
    # W1 shape: (n_hidden, n_input)  — each row is one hidden neuron's weights
    W1 = rng.randn(n_hidden, n_input) * np.sqrt(1.0 / n_input)

    # b1 shape: (n_hidden, 1)  — broadcast-compatible with Z1 = W1·X + b1
    b1 = np.zeros((n_hidden, 1))

    # --- Layer 2: Hidden -> Output ---
    # W2 shape: (n_output, n_hidden)
    W2 = rng.randn(n_output, n_hidden) * np.sqrt(1.0 / n_hidden)

    # b2 shape: (n_output, 1)
    b2 = np.zeros((n_output, 1))

    params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return params


# =============================================================================
# STEP 3: Activation Functions
# =============================================================================

def sigmoid(z):
    """
    Sigmoid (logistic) activation function.

    Formula:  σ(z) = 1 / (1 + e^(-z))

    Properties:
        - Output range: (0, 1)  — interpretable as probability
        - Derivative:  σ'(z) = σ(z) · (1 - σ(z))   [used in backprop]
        - Saturates near 0 and 1, which can slow learning (vanishing gradient)

    Numerically stable implementation clips z to avoid overflow in exp(-z).
    """
    # Clip for numerical stability: exp(-z) overflows for very negative z
    z_clipped = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def sigmoid_derivative(a):
    """
    Derivative of sigmoid in terms of its OUTPUT (post-activation value a).

    Formula:  dσ/dz = a · (1 - a)   where  a = σ(z)

    This avoids re-computing sigmoid; we reuse the cached A1 from forward pass.
    """
    return a * (1.0 - a)


# =============================================================================
# STEP 4: Forward Propagation
# =============================================================================

def forward_propagation(X, params):
    """
    Compute activations for every layer (forward pass).

    Layer 1 (Hidden):
        Z1 = W1 · X  +  b1          (linear combination)
        A1 = sigmoid(Z1)             (non-linear activation)

    Layer 2 (Output):
        Z2 = W2 · A1  +  b2         (linear combination)
        A2 = sigmoid(Z2)             (output probability ŷ ∈ (0,1))

    Matrix shapes (m = number of samples):
        X  : (n_input,  m)
        W1 : (n_hidden, n_input)   Z1 : (n_hidden, m)   A1 : (n_hidden, m)
        W2 : (n_output, n_hidden)  Z2 : (n_output, m)   A2 : (n_output, m)

    Parameters:
        X      : input feature matrix, shape (n_input, m)
        params : dict containing W1, b1, W2, b2

    Returns:
        A2    : output predictions ŷ, shape (1, m)
        cache : intermediate values needed for backpropagation
    """
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]

    # --- Hidden layer ---
    Z1 = np.dot(W1, X) + b1        # (n_hidden, m): linear pre-activation
    A1 = sigmoid(Z1)               # (n_hidden, m): non-linear activation

    # --- Output layer ---
    Z2 = np.dot(W2, A1) + b2       # (n_output, m): linear pre-activation
    A2 = sigmoid(Z2)               # (n_output, m): prediction probability ŷ

    # Cache everything needed during backward pass
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache


# =============================================================================
# STEP 5: Cost Function — Binary Cross-Entropy
# =============================================================================

def compute_cost(A2, Y):
    """
    Compute binary cross-entropy (log-loss) between predictions and labels.

    Formula:
        L = - (1/m) * Σ [ y·log(ŷ) + (1-y)·log(1-ŷ) ]

    Why cross-entropy?
        - It measures the "surprise" of the model's predicted distribution
          relative to the true distribution.
        - Its gradient w.r.t. the final linear layer is simply (ŷ - y),
          which is clean and numerically stable for sigmoid outputs.
        - MSE with sigmoid leads to slow learning due to gradient saturation.

    Parameters:
        A2 : predicted probabilities ŷ, shape (1, m)
        Y  : true labels,            shape (1, m)

    Returns:
        cost : scalar loss value
    """
    m = Y.shape[1]                      # number of samples

    # Clip predictions to avoid log(0) which is -infinity
    eps = 1e-15
    A2_clipped = np.clip(A2, eps, 1 - eps)

    # Element-wise cross-entropy, then average over all m samples
    cost = -(1.0 / m) * np.sum(
        Y * np.log(A2_clipped) + (1 - Y) * np.log(1 - A2_clipped)
    )
    return float(np.squeeze(cost))      # ensure scalar


# =============================================================================
# STEP 6: Backpropagation
# =============================================================================

def backpropagation(X, Y, params, cache):
    """
    Compute gradients of the loss w.r.t. all parameters via chain rule.

    Chain rule summary (backprop equations):
    ─────────────────────────────────────────
    Output layer:
        dZ2 = A2 - Y                            # gradient of loss w.r.t. Z2
              (this simplifies from dL/dA2 * sigmoid'(Z2))
        dW2 = (1/m) * dZ2 · A1ᵀ                # gradient w.r.t. W2
        db2 = (1/m) * Σ dZ2                     # gradient w.r.t. b2

    Hidden layer:
        dA1 = W2ᵀ · dZ2                         # backprop through W2
        dZ1 = dA1 * sigmoid'(Z1)               # backprop through sigmoid
            = dA1 * A1 * (1 - A1)
        dW1 = (1/m) * dZ1 · Xᵀ                 # gradient w.r.t. W1
        db1 = (1/m) * Σ dZ1                     # gradient w.r.t. b1

    Parameters:
        X      : input matrix,    shape (n_input, m)
        Y      : true labels,     shape (1, m)
        params : dict of W1,b1,W2,b2
        cache  : dict of Z1,A1,Z2,A2 from forward pass

    Returns:
        grads : dict with dW1, db1, dW2, db2
    """
    m  = X.shape[1]
    W2 = params["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    # --- Output layer gradients ---
    # dL/dZ2 = A2 - Y   (derived from cross-entropy + sigmoid combined)
    dZ2 = A2 - Y                                        # (n_output, m)
    dW2 = (1.0 / m) * np.dot(dZ2, A1.T)                # (n_output, n_hidden)
    db2 = (1.0 / m) * np.sum(dZ2, axis=1, keepdims=True)  # (n_output, 1)

    # --- Hidden layer gradients ---
    # Backpropagate error through W2
    dA1 = np.dot(W2.T, dZ2)                             # (n_hidden, m)
    # Multiply by local gradient of sigmoid: σ'(z) = a(1-a)
    dZ1 = dA1 * sigmoid_derivative(A1)                  # (n_hidden, m)
    dW1 = (1.0 / m) * np.dot(dZ1, X.T)                 # (n_hidden, n_input)
    db1 = (1.0 / m) * np.sum(dZ1, axis=1, keepdims=True)  # (n_hidden, 1)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads


# =============================================================================
# STEP 7: Weight Update — Gradient Descent
# =============================================================================

def update_parameters(params, grads, learning_rate):
    """
    Update all parameters using vanilla (batch) gradient descent.

    Gradient Descent Update Rule:
        θ ← θ - α · ∂L/∂θ

    where:
        θ            is any parameter (W or b)
        α            is the learning rate (step size)
        ∂L/∂θ        is the gradient of the loss w.r.t. θ

    Intuition: we move each parameter in the direction that decreases loss.
    The learning rate α controls how large each step is:
        - Too large: overshoot, diverge
        - Too small: converge slowly, risk getting stuck

    Parameters:
        params        : current parameter dict {W1, b1, W2, b2}
        grads         : gradient dict  {dW1, db1, dW2, db2}
        learning_rate : scalar step size α

    Returns:
        params : updated parameter dict
    """
    params["W1"] -= learning_rate * grads["dW1"]
    params["b1"] -= learning_rate * grads["db1"]
    params["W2"] -= learning_rate * grads["dW2"]
    params["b2"] -= learning_rate * grads["db2"]
    return params


# =============================================================================
# STEP 8: Training Loop
# =============================================================================

def train(X, Y, n_hidden=4, learning_rate=0.5, n_epochs=1000,
          print_every=100, random_state=42):
    """
    Full training loop: initialise -> forward -> cost -> backward -> update.

    For each epoch:
        1. Forward pass:  compute ŷ = A2
        2. Compute cost:  L = cross_entropy(ŷ, y)
        3. Backward pass: compute gradients ∂L/∂W, ∂L/∂b
        4. Update params: W, b ← W - α·∂L/∂W,  b - α·∂L/∂b
        5. Log cost every `print_every` epochs

    Parameters:
        X             : input features,  shape (n_input, m)
        Y             : true labels,     shape (1, m)
        n_hidden      : number of hidden neurons
        learning_rate : gradient descent step size α
        n_epochs      : total number of training epochs
        print_every   : how often to print the loss
        random_state  : seed for reproducibility

    Returns:
        params     : trained parameters
        cost_history : list of loss values (one per epoch)
    """
    n_input  = X.shape[0]
    n_output = Y.shape[0]

    # Initialise weights (Step 2)
    params = initialise_parameters(n_input, n_hidden, n_output, random_state)

    cost_history = []

    print(f"Training MLP: {n_input} -> {n_hidden} -> {n_output}")
    print(f"Learning Rate: {learning_rate},  Epochs: {n_epochs}\n")

    for epoch in range(1, n_epochs + 1):

        # ---------- Step 4: Forward Propagation ----------
        A2, cache = forward_propagation(X, params)

        # ---------- Step 5: Compute Cost ------------------
        cost = compute_cost(A2, Y)
        cost_history.append(cost)

        # ---------- Step 6: Backpropagation ---------------
        grads = backpropagation(X, Y, params, cache)

        # ---------- Step 7: Update Weights ----------------
        params = update_parameters(params, grads, learning_rate)

        # Log progress
        if epoch % print_every == 0 or epoch == 1:
            Y_pred = (A2 >= 0.5).astype(int)
            accuracy = np.mean(Y_pred == Y) * 100
            print(f"  Epoch {epoch:>5}/{n_epochs} | "
                  f"Loss: {cost:.6f} | Accuracy: {accuracy:.1f}%")

    print("\nTraining complete.")
    return params, cost_history


# =============================================================================
# STEP 9 & 10: Visualisation — Loss Curve + Decision Boundaries
# =============================================================================

def predict(X, params):
    """Run forward pass and return binary predictions (0 or 1)."""
    A2, _ = forward_propagation(X, params)
    return (A2 >= 0.5).astype(int), A2   # labels and raw probabilities


def plot_results(X, Y, params, cost_history, dataset_name="XOR"):
    """
    Create a 3-panel figure:
        Panel 1: Training loss curve (cross-entropy vs epoch)
        Panel 2: Decision boundary with training data scatter
        Panel 3: Predicted probability heatmap
    """
    # ---- Setup figure ----
    fig = plt.figure(figsize=(18, 5))
    fig.suptitle(
        f"MLP from Scratch — {dataset_name} Dataset\n"
        f"Architecture: {X.shape[0]} inputs → hidden → 1 output  |  "
        f"Activation: Sigmoid  |  Loss: Binary Cross-Entropy",
        fontsize=12, fontweight='bold'
    )
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ─────────────────────────────────────────────────
    # Panel 1: Loss Curve
    # ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    epochs = np.arange(1, len(cost_history) + 1)
    ax1.plot(epochs, cost_history, color='steelblue', linewidth=1.5, alpha=0.9)
    ax1.fill_between(epochs, cost_history, alpha=0.15, color='steelblue')
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax1.set_title("Training Loss Curve", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Annotate final loss
    ax1.annotate(
        f"Final loss:\n{cost_history[-1]:.4f}",
        xy=(epochs[-1], cost_history[-1]),
        xytext=(-80, 20), textcoords='offset points',
        fontsize=9, color='steelblue',
        arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.2)
    )

    # ─────────────────────────────────────────────────
    # Shared: create a fine mesh grid for boundary / heatmap
    # ─────────────────────────────────────────────────
    # We always visualise using the first 2 dimensions of X
    x_min, x_max = X[0].min() - 0.3, X[0].max() + 0.3
    y_min, y_max = X[1].min() - 0.3, X[1].max() + 0.3
    h = 0.02  # mesh step size
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Build grid input: shape (2, n_grid_points)
    if X.shape[0] == 2:
        grid_input = np.c_[xx.ravel(), yy.ravel()].T
    else:
        # For higher-dim inputs pad remaining dims with their training mean
        grid_input = np.tile(X.mean(axis=1, keepdims=True),
                             (1, xx.ravel().shape[0]))
        grid_input[0] = xx.ravel()
        grid_input[1] = yy.ravel()

    _, grid_proba = predict(grid_input, params)
    Z_proba = grid_proba.reshape(xx.shape)
    Z_class = (Z_proba >= 0.5).astype(int)

    # ─────────────────────────────────────────────────
    # Panel 2: Decision Boundary (hard boundary)
    # ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    cmap_bg = plt.cm.RdYlGn
    ax2.contourf(xx, yy, Z_class, alpha=0.35, cmap=cmap_bg, levels=[-0.5, 0.5, 1.5])
    ax2.contour(xx, yy, Z_proba, levels=[0.5], colors='black',
                linewidths=2, linestyles='--')

    # Scatter training points
    colors = np.where(Y.ravel() == 1, 'green', 'red')
    ax2.scatter(X[0], X[1], c=colors, edgecolors='k',
                linewidths=0.5, s=40, zorder=3)

    # Legend proxies
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
               markeredgecolor='k', markersize=8, label='Class 1'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markeredgecolor='k', markersize=8, label='Class 0'),
        Line2D([0], [0], color='black', linestyle='--',
               linewidth=2, label='Decision Boundary (p=0.5)')
    ]
    ax2.legend(handles=legend_elements, fontsize=8, loc='upper right')
    ax2.set_xlabel("Feature 1", fontsize=11)
    ax2.set_ylabel("Feature 2", fontsize=11)
    ax2.set_title("Decision Boundary", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.3)

    # ─────────────────────────────────────────────────
    # Panel 3: Predicted Probability Heatmap
    # ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    pcm = ax3.contourf(xx, yy, Z_proba, levels=50, cmap='coolwarm', alpha=0.85)
    plt.colorbar(pcm, ax=ax3, label='P(class=1)')
    ax3.contour(xx, yy, Z_proba, levels=[0.5], colors='white',
                linewidths=2, linestyles='--')
    ax3.scatter(X[0], X[1], c=colors, edgecolors='k',
                linewidths=0.5, s=40, zorder=3)
    ax3.set_xlabel("Feature 1", fontsize=11)
    ax3.set_ylabel("Feature 2", fontsize=11)
    ax3.set_title("Predicted Probability Heatmap", fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.3)

    plt.savefig(f"mlp_{dataset_name.lower()}_results.png",
                dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Figure saved as mlp_{dataset_name.lower()}_results.png")


def print_summary(X_train, Y_train, X_test, Y_test, params):
    """Print final accuracy and a confusion-matrix-style summary."""
    Y_pred_train, _ = predict(X_train, params)
    Y_pred_test,  _ = predict(X_test,  params)

    train_acc = np.mean(Y_pred_train == Y_train) * 100
    test_acc  = np.mean(Y_pred_test  == Y_test)  * 100

    print("\n" + "=" * 45)
    print("       FINAL MODEL PERFORMANCE SUMMARY")
    print("=" * 45)
    print(f"  Training Accuracy : {train_acc:.2f}%")
    print(f"  Test     Accuracy : {test_acc:.2f}%")

    # Simple confusion matrix for test set
    TP = int(np.sum((Y_pred_test == 1) & (Y_test == 1)))
    TN = int(np.sum((Y_pred_test == 0) & (Y_test == 0)))
    FP = int(np.sum((Y_pred_test == 1) & (Y_test == 0)))
    FN = int(np.sum((Y_pred_test == 0) & (Y_test == 1)))

    print("\n  Confusion Matrix (Test Set):")
    print(f"  {'':18} Predicted 0  Predicted 1")
    print(f"  {'Actual 0':18} {TN:>10}   {FP:>10}")
    print(f"  {'Actual 1':18} {FN:>10}   {TP:>10}")
    print("=" * 45)


# =============================================================================
# MAIN: Run both XOR and Iris experiments
# =============================================================================

if __name__ == "__main__":

    # ─────────────────────────────────────────────────────────────────────────
    # Experiment A: XOR Problem
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 55)
    print("  EXPERIMENT A: XOR Classification")
    print("=" * 55)

    # Step 1: Create dataset
    X_xor, Y_xor = make_xor_dataset(n_samples=400, noise=0.05)
    # Reshape to (n_features, m) layout
    X_xor_T = X_xor.T                                  # (2, 400)

    # Split manually for evaluation
    idx = np.arange(X_xor_T.shape[1])
    np.random.seed(42)
    np.random.shuffle(idx)
    split = int(0.8 * len(idx))
    X_xor_train, Y_xor_train = X_xor_T[:, idx[:split]], Y_xor[:, idx[:split]]
    X_xor_test,  Y_xor_test  = X_xor_T[:, idx[split:]], Y_xor[:, idx[split:]]

    # Steps 2-8: Train
    params_xor, history_xor = train(
        X_xor_train, Y_xor_train,
        n_hidden=8,
        learning_rate=1.0,
        n_epochs=1000,
        print_every=100
    )

    # Summary + Visualisation
    print_summary(X_xor_train, Y_xor_train, X_xor_test, Y_xor_test, params_xor)

    # Steps 9-10: Plot
    plot_results(X_xor_train, Y_xor_train, params_xor, history_xor, "XOR")

    # ─────────────────────────────────────────────────────────────────────────
    # Experiment B: Iris Binary Classification
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  EXPERIMENT B: Iris (Setosa vs Versicolor)")
    print("=" * 55)

    # Step 1: Load Iris
    X_iris_train, X_iris_test, Y_iris_train, Y_iris_test, _ = \
        make_iris_binary_dataset()

    # Steps 2-8: Train
    params_iris, history_iris = train(
        X_iris_train, Y_iris_train,
        n_hidden=6,
        learning_rate=0.5,
        n_epochs=1000,
        print_every=100
    )

    # Summary + Visualisation
    print_summary(X_iris_train, Y_iris_train, X_iris_test, Y_iris_test, params_iris)

    # Steps 9-10: Plot
    plot_results(X_iris_train, Y_iris_train, params_iris, history_iris, "Iris")

    # ─────────────────────────────────────────────────────────────────────────
    # Print final learned weights for inspection
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Learned weights (XOR model) ---")
    print(f"W1 shape: {params_xor['W1'].shape}")
    print(f"W2 shape: {params_xor['W2'].shape}")
    print(f"W1:\n{np.round(params_xor['W1'], 3)}")
    print(f"W2:\n{np.round(params_xor['W2'], 3)}")
