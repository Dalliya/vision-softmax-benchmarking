import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from typing import Dict, List

class Visualizer:
    # High-Definition Matrix Terminal Palette
    COLOR_BG = "#050505"
    COLOR_TEXT_G = "#00FF41"    # Neon Green (Success)
    COLOR_ACCENT_C = "#08F7FE"  # Cyber Cyan
    COLOR_WARN_Y = "#F5D300"    # Warning Yellow (Degradation)
    COLOR_ERROR_R = "#FF003C"   # Critical Red (Failure)
    COLOR_GRID = "#003B00"

    @staticmethod
    def _apply_theme():
        plt.style.use('dark_background')
        plt.rcParams.update({
            'figure.facecolor': Visualizer.COLOR_BG,
            'axes.facecolor': Visualizer.COLOR_BG,
            'font.family': 'monospace',
            'text.color': Visualizer.COLOR_TEXT_G,
            'axes.labelcolor': Visualizer.COLOR_TEXT_G,
            'xtick.color': Visualizer.COLOR_TEXT_G,
            'ytick.color': Visualizer.COLOR_TEXT_G,
            'axes.edgecolor': Visualizer.COLOR_GRID
        })

    @staticmethod
    def _add_matrix_background(fig):
        ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-1)
        ax_bg.set_axis_off()
        ax_bg.set_facecolor(Visualizer.COLOR_BG)
        
        chars = ['0', '1']
        np.random.seed(42)
        
        for _ in range(150): 
            x, y = np.random.uniform(0, 1), np.random.uniform(0.1, 1.2)
            length = np.random.randint(15, 40)
            
            for j in range(length):
                char_y = y - (j * 0.02)
                if 0 <= char_y <= 1:
                    alpha = max(0.02, 0.25 * (1.0 - (j / length)))
                    color = '#ffffff' if j == 0 else Visualizer.COLOR_TEXT_G
                    ax_bg.text(x, char_y, np.random.choice(chars), color=color, 
                               alpha=alpha, fontsize=9, transform=ax_bg.transAxes, 
                               va='center', ha='center', clip_on=True)

    @staticmethod
    def plot_telemetry_dashboard(history: Dict[str, Dict[str, List[float]]]):
        Visualizer._apply_theme()
        
        fig, ax_loss = plt.subplots(figsize=(15, 8))
        Visualizer._add_matrix_background(fig)
        
        ax_loss.set_facecolor('none')
        ax_acc = ax_loss.twinx() 

        color_map = {
            "fashion-mnist": Visualizer.COLOR_TEXT_G,
            "stl-10": Visualizer.COLOR_WARN_Y,
            "eurosat": Visualizer.COLOR_ERROR_R
        }

        ax_loss.set_xlim(0.8, 6.5)

        for name in history.keys():
            losses = history[name]['loss']
            accs = history[name]['acc']
            epochs = np.arange(1, len(losses) + 1)
            color = color_map.get(name, Visualizer.COLOR_ACCENT_C)

            ax_loss.plot(epochs, losses, color=color, ls='--', marker='s', lw=2.5, alpha=0.8)
            ax_acc.plot(epochs, accs, color=color, ls='-', marker='o', lw=3.5)

            final_acc = accs[-1]
            if final_acc > 70:
                status = "PASSED: OPTIMAL"
            elif final_acc > 35:
                status = "WARNING: DEGRADATION"
            else:
                status = "CRITICAL: COLLAPSE"

            # Точное указание CE LOSS для технической грамотности
            box_text = (f"[{name.upper()}]\n"
                        f"STATUS : {status}\n"
                        f"CE LOSS: {losses[-1]:.2f} (--)\n"
                        f"ACC    : {accs[-1]:.1f}% (-)")
            
            ax_acc.text(epochs[-1] + 0.15, accs[-1], box_text, color=color,
                        bbox=dict(boxstyle="square,pad=0.5", fc='#080808', 
                                  ec=color, lw=2.5, alpha=0.95), 
                        va='center', ha='left', fontsize=10, fontweight='bold')

        ax_loss.set_xlabel("EPOCH SEQUENCE", fontweight='bold', fontsize=12, color=Visualizer.COLOR_TEXT_G)
        
        # Обновленные заголовки осей
        ax_loss.set_ylabel("CROSS-ENTROPY LOSS (DASHED)", fontweight='bold', color=Visualizer.COLOR_TEXT_G)
        ax_acc.set_ylabel("ACCURACY % (SOLID)", fontweight='bold', color=Visualizer.COLOR_TEXT_G)
        
        ax_loss.grid(True, alpha=0.15, color=Visualizer.COLOR_TEXT_G, linestyle=':')
        plt.title(">> INDUSTRIAL PERFORMANCE TELEMETRY: LINEAR DECONSTRUCTION [CRASH TEST]", 
                  loc='left', pad=25, fontsize=16, color=Visualizer.COLOR_TEXT_G, fontweight='bold')
        
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/loss_comparison.png", dpi=300, bbox_inches='tight')

    @staticmethod
    def plot_prediction_grid(samples: torch.Tensor, targets: torch.Tensor, outputs: torch.Tensor, 
                             class_names: List[str], save_path: str, dataset_name: str, 
                             accuracy: float, loss: float):
        Visualizer._apply_theme()
        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        
        status = "PASSED" if accuracy > 70 else "WARNING" if accuracy > 35 else "CRITICAL FAILURE"
        color_title = Visualizer.COLOR_TEXT_G if accuracy > 70 else Visualizer.COLOR_WARN_Y if accuracy > 35 else Visualizer.COLOR_ERROR_R
        
        # Заголовок теперь содержит CE LOSS
        fig.suptitle(f">> {dataset_name.upper()} ACQUISITION: {status} | ACC: {accuracy:.1f}% | CE LOSS: {loss:.2f}", 
                     fontsize=15, fontweight='bold', y=0.96, color=color_title)

        samples = samples[:8].cpu() * 0.5 + 0.5 
        
        for i, ax in enumerate(axes.flat):
            img = samples[i].permute(1, 2, 0).numpy() if samples[i].shape[0] > 1 else samples[i].squeeze()
            ax.imshow(img, cmap='gray' if samples[i].shape[0] == 1 else None)
            
            correct = targets[i] == outputs[i]
            c = Visualizer.COLOR_TEXT_G if correct else Visualizer.COLOR_ERROR_R
            
            ax.set_title(f"TRUE: {class_names[targets[i]]}\nPRED: {class_names[outputs[i]]}", 
                         color=c, fontsize=10, fontweight='bold')
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(c)
                spine.set_linewidth(4.0)

        plt.tight_layout(rect=[0, 0.03, 1, 0.90])
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, facecolor=Visualizer.COLOR_BG)

    @staticmethod
    def show_all():
        plt.show()