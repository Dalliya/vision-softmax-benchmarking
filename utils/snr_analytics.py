import torch
import numpy as np
import matplotlib.pyplot as plt

def calculate_dataset_snr(dataloader) -> float:
    """
    Calculates the average Signal-to-Noise Ratio (SNR) across a dataset.
    
    Approximates 'signal' as mean absolute intensity and 'noise' as 
    standard deviation. Uses absolute values to handle normalized tensors.
    
    Args:
        dataloader (torch.utils.data.DataLoader): The dataset loader.
        
    Returns:
        float: Average SNR in decibels (dB).
    """
    snr_values = []
    
    for images, _ in dataloader:
        for i in range(images.size(0)):
            # Use absolute values to prevent negative means after normalization
            img = torch.abs(images[i])
            
            mu_signal = img.mean().item()
            sigma_noise = img.std().item()
            
            # Prevent division by zero for uniform images
            if sigma_noise == 0:
                sigma_noise = 1e-8
                
            # Compute SNR and convert to decibels
            snr = mu_signal / sigma_noise
            snr_db = 20 * np.log10(snr + 1e-8) 
            snr_values.append(snr_db)
            
    return float(np.mean(snr_values))


def plot_snr_correlation(datasets: list, snr_values: list, accuracies: list, save_path: str = "results/snr_correlation.png"):
    """
    Plots the correlation between Dataset SNR and Model Accuracy, 
    using the Matrix Terminal aesthetic with detailed annotations.
    
    Args:
        datasets (list): Names of the evaluated datasets.
        snr_values (list): Computed SNR values (in dB) corresponding to each dataset.
        accuracies (list): Model accuracy percentages corresponding to each dataset.
        save_path (str): Directory path to save the output figure.
    """
    plt.style.use('dark_background') 
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Matrix Terminal Colors
    COLOR_BG = "#050505"
    COLOR_TEXT_G = "#00FF41"    # Neon Green
    COLOR_WARN_Y = "#F5D300"    # Warning Yellow
    COLOR_ERROR_R = "#FF003C"   # Critical Red
    COLOR_GRID = "#003B00"
    
    # Apply background colors
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    
    # Plot individual data points and add bounding boxes
    for i in range(len(datasets)):
        name = datasets[i].upper()
        snr = snr_values[i]
        acc = accuracies[i]
        
        # Determine status and color based on accuracy thresholds
        if acc > 70:
            status = "PASSED: OPTIMAL"
            color = COLOR_TEXT_G
        elif acc > 35:
            status = "WARNING: DEGRADATION"
            color = COLOR_WARN_Y
        else:
            status = "CRITICAL: COLLAPSE"
            color = COLOR_ERROR_R

        # Scatter the point (Large dot)
        ax.scatter(snr, acc, color=color, s=300, edgecolor='white', lw=2, zorder=5)
        
        # Create the annotation box text
        box_text = (f"[{name}]\n"
                    f"STATUS : {status}\n"
                    f"SNR    : {snr:.2f} dB\n"
                    f"ACC    : {acc:.1f}%")
        
        # Add the text box next to the point
        x_offset = (max(snr_values) - min(snr_values)) * 0.05 if len(snr_values) > 1 else 0.5
        ax.text(snr + x_offset, acc, box_text, color=color,
                bbox=dict(boxstyle="square,pad=0.5", fc='#080808', 
                          ec=color, lw=2.5, alpha=0.95), 
                va='center', ha='left', fontsize=11, fontweight='bold', family='monospace')
    
    # Sort points for a clean trend line
    sorted_indices = np.argsort(snr_values)
    snr_sorted = np.array(snr_values)[sorted_indices]
    acc_sorted = np.array(accuracies)[sorted_indices]
    
    # Add a dashed trend line
    ax.plot(snr_sorted, acc_sorted, color='white', linestyle='--', lw=2, alpha=0.4, zorder=1)

    # Styling and typography
    ax.set_title(">> SNR CORRELATION: BACKGROUND NOISE VS ACCURACY", loc='left', pad=25, 
                 fontsize=16, color=COLOR_TEXT_G, fontweight='bold', family='monospace')
    ax.set_xlabel("SIGNAL-TO-NOISE RATIO (dB) -> HIGHER MEANS CLEANER DATA", fontweight='bold', color=COLOR_TEXT_G, family='monospace')
    ax.set_ylabel("ACCURACY (%)", fontweight='bold', color=COLOR_TEXT_G, family='monospace')
    
    # Grid and axis styling
    ax.tick_params(colors=COLOR_TEXT_G)
    for spine in ax.spines.values():
        spine.set_color(COLOR_GRID)
        spine.set_linewidth(1.5)
        
    ax.grid(True, linestyle=':', alpha=0.3, color=COLOR_TEXT_G)
    
    # Expand the right X-axis limit so text boxes don't get cut off
    x_min, x_max = ax.get_xlim()
    ax.set_xlim(x_min - 1, x_max + (x_max - x_min) * 0.4) 
    
    # Save and display
    plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor=COLOR_BG)
    plt.show() 
    plt.close()