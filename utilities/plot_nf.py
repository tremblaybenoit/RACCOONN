import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from utilities.plot import save_plot, flexible_gridspec, scatterplots
import random
import argparse


# --- 1D Neural Field Model ---
class SimpleNeuralField(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        # Simple MLP to map a 1D coordinate (x) to a 1D function value (f(x))
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.net(x)


# --- Simulation and Plotting ---
def fig_1d_curve_fitting():
    """Generates the 1D curve fitting analogy visualization for Neural Fields."""

    # 1. Define the True Complex Function (The target signal)
    x_range = np.linspace(0, 10, 500)
    # A complex, wobbly function: f(x) = sin(x) + sin(5x) * 0.5
    f_true = (np.sin(x_range) + np.sin(5 * x_range) * 0.5)

    # 2. Sample Training Data (Sparse coordinates)
    N_samples = 30
    x_train_np = np.linspace(0, 10, N_samples) + np.random.normal(0, 0.1, N_samples)
    y_train_np = (np.sin(x_train_np) + np.sin(5 * x_train_np) * 0.5) + np.random.normal(0, 0.1, N_samples)

    # Convert to PyTorch tensors
    x_train = torch.tensor(x_train_np).float().view(-1, 1)
    y_train = torch.tensor(y_train_np).float().view(-1, 1)

    # 3. Model Training Setup
    model = SimpleNeuralField()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Coordinates for smooth plotting (full range query)
    x_plot = torch.tensor(x_range).float().view(-1, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Neural Field: The 1D Function Approximator Analogy', fontsize=16)

    # --- Initial State Plot ---
    with torch.no_grad():
        y_initial = model(x_plot).numpy()

    ax1.plot(x_range, f_true, 'k--', label='True Function $f(x)$', linewidth=2, alpha=0.6)
    ax1.scatter(x_train_np, y_train_np, color='red', s=40, label='Sparse Training Coordinates $(x, f(x))$')
    ax1.plot(x_range, y_initial, color='gray', label='MLP Output (Initial)', linestyle=':')
    ax1.set_title('(A) Initial Random State')
    ax1.legend(fontsize=10)
    ax1.set_xlabel('Coordinate $x$')
    ax1.set_ylabel('Function Value $f(x)$')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Training Loop (Simplified for demonstration) ---
    losses = []

    for epoch in range(2000):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # --- Final Trained State Plot ---
    with torch.no_grad():
        y_final = model(x_plot).numpy()

    ax2.plot(x_range, f_true, 'k--', label='True Function $f(x)$', linewidth=2, alpha=0.6)
    ax2.scatter(x_train_np, y_train_np, color='red', s=40, label='Sparse Training Coordinates $(x, f(x))$')
    ax2.plot(x_range, y_final, color='blue', label='MLP Output (Trained)', linewidth=3)
    ax2.set_title('(B) Final Trained State (Interpolation)')
    ax2.legend(fontsize=10)
    ax2.set_xlabel('Coordinate $x$')
    ax2.set_ylabel('Function Value $f(x)$')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# --- Simulation and Plotting ---
def fig_1d_overfitting_validation(SEED, lr=0.01):
    """
    Generates the 1D curve fitting analogy visualization, focusing on
    overfitting and the role of the validation set (cross-validation).
    """

    # === REPRODUCIBILITY SEED ===
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # Force CPU
    device = torch.device('cpu')
    #if torch.cuda.is_available():
    #    torch.cuda.manual_seed_all(SEED)

    # 1. Define the True Smooth Function
    x_range = np.linspace(0, 10, 500)
    f_true = (np.sin(x_range) + np.sin(3 * x_range) * 0.3)

    # 2. Sample Training Data (Sparse coordinates + HIGH NOISE)
    N_train = 30
    sample_indices = np.random.choice(len(x_range), N_train, replace=False)
    x_train_np = x_range[sample_indices]
    f_sampled = f_true[sample_indices]
    # Add high-frequency noise that the model should ignore for generalization
    y_train_np = f_sampled + np.random.normal(0, 0.2, N_train)

    # 3. Create Validation Data (coordinates between training points)
    # This set should ideally be smooth and without the same noise profile
    N_val = 20
    sample_indices = np.random.choice(len(x_range), N_val, replace=False)
    x_val_np = x_range[sample_indices]
    y_val_np = (np.sin(x_val_np) + np.sin(3 * x_val_np) * 0.3)

    # Convert to PyTorch tensors
    x_train = torch.tensor(x_train_np).float().view(-1, 1)
    y_train = torch.tensor(y_train_np).float().view(-1, 1)
    x_val = torch.tensor(x_val_np).float().view(-1, 1)
    y_val = torch.tensor(y_val_np).float().view(-1, 1)

    # Coordinates for smooth plotting (full range query)
    x_plot = torch.tensor(x_range).float().view(-1, 1)

    # 4. Model Training Setup (Regularized vs. Overfit)
    model_regularized = SimpleNeuralField()
    model_overfit = SimpleNeuralField()

    # Ensure both models start identically for a fair comparison
    model_overfit.load_state_dict(model_regularized.state_dict())

    optimizer_reg = optim.Adam(model_regularized.parameters(), lr=lr)
    optimizer_over = optim.Adam(model_overfit.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Initialize for Early Stopping
    # Set patience (how many epochs to wait for improvement)
    patience = 500
    best_val_loss = float('inf')
    epochs_no_improve = 0
    should_stop_reg = False

    # We will simulate early stopping decision by tracking validation loss
    for epoch in range(5000):
        # =======================================================
        # 1. Regularized Model (Simulate Early Stopping)
        # =======================================================
        if not should_stop_reg:
            # a. Training step (Calculate training loss)
            model_regularized.train()
            optimizer_reg.zero_grad()
            loss_train = criterion(model_regularized(x_train), y_train)
            loss_train.backward()
            optimizer_reg.step()

            # b. Validation Check (Calculate validation loss)
            model_regularized.eval()
            with torch.no_grad():
                # Calculate loss on the *unseen* validation set
                val_loss = criterion(model_regularized(x_val), y_val)

            # c. Early Stopping Logic (Decision)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # OPTIONAL: Save the best model state here
                best_model_state = model_regularized.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    # The validation loss has stopped improving, so we stop training
                    should_stop_reg = True
                    print(f"Regularized Model stopped training at epoch {epoch} due to no validation loss improvement.")
                    # OPTIONAL: Load the best saved model state here
                    model_regularized.load_state_dict(best_model_state)

        # =======================================================
        # 2. Overfit Model (Train to completion)
        # =======================================================
        model_overfit.train()
        optimizer_over.zero_grad()
        loss_over = criterion(model_overfit(x_train), y_train)
        # Use a high L2 regularization weight to enhance the contrast between models
        # This is a common technique to control complexity, but let the overfit model ignore it
        # loss_over += 0.001 * sum(p.pow(2).sum() for p in model_overfit.parameters())
        loss_over.backward()
        optimizer_over.step()

        # Optional: Print status
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Reg Loss={loss_train.item():.5f}, Val Loss={val_loss.item():.5f}")

    # 5. Final Query and Plotting
    with torch.no_grad():
        y_reg = model_regularized(x_plot).numpy()
        y_over = model_overfit(x_plot).numpy()

    # Plot #1
    cell_widths = [4.0]
    cell_heights = [4.0]
    lefts = [0.75]
    rights = [0.75]
    bottoms = [0.75]
    tops = [0.75]
    colors = {'blue': '#1f77b4',
              'orange': '#ff7f0e',
              'green': '#2ca02c',
              'red': '#d62728',
              'purple': '#9467bd',
              'brown': '#8c564b',
              'pink': '#e377c2',
              'gray': '#7f7f7f',
              'olive': '#bcbd22',
              'cyan': '#17becf'}

    # Scatterplot
    fig1, get_axes = flexible_gridspec(cell_widths, cell_heights, lefts, rights, bottoms, tops)
    ax1 = get_axes(0, 0)
    scatterplots(ax1, [x_train_np, x_val_np], [y_train_np, np.ones_like(x_val_np)+100], y_labelpad=2,
                 x_range=(-1.5, 11.5), y_range=(-2, 2), x_label='Coordinate', y_label='Retrieval value',
                 color=[colors['blue'], colors['orange']], marker=['o', 'x'], markersize=10,
                 label=['Training set', 'Validation set'], xy_symmetric=False)
    ax1.plot(x_range, -100*np.ones_like(x_range), color=colors['green'], label='Model (overfit)', linewidth=1., linestyle='-',
            zorder=0)
    ax1.plot(x_range, -100*np.ones_like(x_range), color=colors['brown'], label='Model (cross-validation)', linewidth=1., linestyle='-', zorder=0)
    # Update legend to show model types
    ax1.legend(loc='lower left', fontsize=10, labelspacing=0.05, numpoints=1, ncol=1,
              markerscale=1.0, fancybox=False)

    fig2, get_axes = flexible_gridspec(cell_widths, cell_heights, lefts, rights, bottoms, tops)
    ax2 = get_axes(0, 0)
    scatterplots(ax2, [x_train_np, x_val_np], [y_train_np, y_val_np], y_labelpad=2,
                 x_range=(-1.5, 11.5), y_range=(-2, 2), x_label='Coordinate', y_label='Retrieval value',
                 color=[colors['blue'], colors['orange']], marker=['o', 'x'], markersize=10,
                 label=['Training set', 'Validation set'], xy_symmetric=False)
    ax2.plot(x_range, -100 * np.ones_like(x_range), color=colors['green'], label='Model (overfit)',
            linewidth=1., linestyle='-',
            zorder=0)
    ax2.plot(x_range, -100 * np.ones_like(x_range), color=colors['brown'], label='Model (cross-validation)',
            linewidth=1., linestyle='-', zorder=0)
    # Update legend to show model types
    ax2.legend(loc='lower left', fontsize=10, labelspacing=0.05, numpoints=1, ncol=1,
              markerscale=1.0, fancybox=False)

    fig3, get_axes = flexible_gridspec(cell_widths, cell_heights, lefts, rights, bottoms, tops)
    ax3 = get_axes(0, 0)
    scatterplots(ax3, [x_train_np, x_val_np], [y_train_np, y_val_np], y_labelpad=2,
                 x_range=(-1.5, 11.5), y_range=(-2, 2), x_label='Coordinate', y_label='Retrieval value',
                 color=[colors['blue'], colors['orange']], marker=['o', 'x'], markersize=10,
                 label=['Training set', 'Validation set'], xy_symmetric=False)
    ax3.plot(x_range, y_over, color=colors['green'], label='Model (overfit)', linewidth=1., linestyle='-',
             zorder=0)
    ax3.plot(x_range, -100 * np.ones_like(x_range), color=colors['brown'], label='Model (cross-validation)',
            linewidth=1., linestyle='-', zorder=0)
    # Update legend to show model types
    ax3.legend(loc='lower left', fontsize=10, labelspacing=0.05, numpoints=1, ncol=1,
               markerscale=1.0, fancybox=False)

    fig4, get_axes = flexible_gridspec(cell_widths, cell_heights, lefts, rights, bottoms, tops)
    ax4 = get_axes(0, 0)
    scatterplots(ax4, [x_train_np, x_val_np], [y_train_np, y_val_np], y_labelpad=2,
                 x_range=(-1.5, 11.5), y_range=(-2, 2), x_label='Coordinate', y_label='Retrieval value',
                 color=[colors['blue'], colors['orange']], marker=['o', 'x'], markersize=10,
                 label=['Training set', 'Validation set'], xy_symmetric=False)
    ax4.plot(x_range, y_over, color=colors['green'], label='Model (overfit)', linewidth=1.,
             linestyle='-',
             zorder=0)
    ax4.plot(x_range, y_reg, color=colors['brown'], label='Model (cross-validation)', linewidth=1., linestyle='-',
             zorder=0)
    # Update legend to show model types
    ax4.legend(loc='lower left', fontsize=10, labelspacing=0.05, numpoints=1, ncol=1,
               markerscale=1.0, fancybox=False)

    return fig1, fig2, fig3, fig4

    # Title
    ax.set_title('Neural Field: Overfitting and the Role of the Validation Set', fontsize=13)

    # Plot True Function and Training/Validation Data
    ax.plot(x_range, f_true, 'k--', label='True Generalizing Function', linewidth=1.5, alpha=0.6)
    ax.scatter(x_train_np, y_train_np, color='red', s=40, label='Training Coordinates (with Noise)', zorder=5)
    ax.scatter(x_val_np, y_val_np, color='green', marker='x', s=100, linewidth=2,
               label='Validation Coordinates (Interpolation Check)', zorder=5)

    # Plot Model Outputs

    # Calculate and display Validation Error (MSE)
    val_pred_reg = model_regularized(x_val)
    val_error_reg = criterion(val_pred_reg, y_val).item()
    val_pred_over = model_overfit(x_val)
    val_error_over = criterion(val_pred_over, y_val).item()

    # Add key text explaining the result
    ax.text(0.05, 0.2,
            f'Validation MSE:\n'
            f'  Ideal Model: {val_error_reg:.4f}\n'
            f'  Overfit Model: {val_error_over:.4f}',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5'),
            fontsize=10)

    ax.legend(loc='lower left', fontsize=10)
    ax.set_xlabel('Coordinate $x$')
    ax.set_ylabel('Function Value $f(x)$')
    ax.set_ylim(f_true.min() - 0.5, f_true.max() + 0.5)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-lr', type=float, default=0.01)
    args = parser.parse_args()

    fig1, fig2, fig3, fig4 = fig_1d_overfitting_validation(args.seed, args.lr)
    # Save figure
    save_plot(fig1, f'1D_neural_field_curve_fitting_{args.seed}_1.png')
    save_plot(fig2, f'1D_neural_field_curve_fitting_{args.seed}_2.png')
    save_plot(fig3, f'1D_neural_field_curve_fitting_{args.seed}_3.png')
    save_plot(fig4, f'1D_neural_field_curve_fitting_{args.seed}_4.png')
