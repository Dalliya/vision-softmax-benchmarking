<h1 align="center">>> INDUSTRIAL TELEMETRY: LINEAR DECONSTRUCTION OF VISUAL MANIFOLDS</h1>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-Scratch_Engine-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Math-Linear_Algebra-00FF41?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Crash_Test_Complete-FF003C?style=for-the-badge" />
</p>

## 🔬 Abstract: The Linear Boundary
This project is an industrial-grade stress test of linear logic. While modern Deep Learning relies on deep, non-linear feature extraction (CNNs, ViTs), it is mathematically critical to understand the exact point where basic linear separability collapses. 

In this study, I engineered a **Softmax Regression Classifier from absolute scratch** (using pure tensor operations, bypassing `torch.nn.Linear` and `torch.nn.CrossEntropyLoss`) to deconstruct visual manifolds across three distinct levels of complexity:
1. **Geometric Simplicity:** `Fashion-MNIST` (Apparel silhouettes)
2. **Natural Chaos:** `STL-10` (Animals/Vehicles with complex backgrounds)
3. **Spatial Textures:** `EuroSAT` (Multispectral satellite imagery)

---

## ⚙️ The Mathematical Engine
The core of this system is a manual forward pass and mathematically stable Cross-Entropy computation, optimized for hardware acceleration (MPS/CUDA).

**Forward Pass (Hyperplane Projection):**
$$y = \text{softmax}(XW + b)$$

**Loss Function (Information Distance):**
$$L = -\frac{1}{N} \sum_{i=1}^{N} \log(p_{i, y_i})$$

By forcing 2D/3D images into flattened 1D vectors ($X_{flat}$), we strip the model of spatial awareness, forcing it to rely purely on raw pixel-intensity correlations.

---

## 📊 System Telemetry: Loss vs Accuracy Analysis

The core finding of this research is visualized in the dual-axis telemetry dashboard below. It clearly demonstrates the "Neural Bottleneck" — the exact moment a linear architecture fails to capture high-dimensional complexity.

*(Ensure you have run the code to generate this image in the `results/` folder)*
![Performance Dashboard](results/loss_comparison.png)

### 📈 Analytical Breakdown of the Curves:
* **🟢 [PASSED] Fashion-MNIST (Optimal):** The green solid line (Accuracy) climbs steadily to **~80.6%**, while the dashed line (Cross-Entropy Loss) drops below **0.90**. The model easily finds hyperplanes to separate dark boots from light t-shirts because the objects are centered and backgrounds are uniformly black. Linear logic holds perfectly.
* **🟡 [WARNING] STL-10 (Degradation):** The yellow line plateaus early at **~38.4%**, and the loss gets stuck around **7.15**. Why? A dog can be on grass, on a bed, facing left, or right. A linear model cannot dynamically adjust to these translations and backgrounds. It attempts to average out all dogs into a single "template," resulting in severe mode collapse.
* **🔴 [CRITICAL] EuroSAT (Complete Collapse):** The red line flatlines at **~27.7%** with a massive loss of **8.74**. Satellite images of forests and rivers share exact pixel values (greens, blues, browns); their only difference is *spatial texture* (how the pixels are arranged). Since our $X_{flat}$ engine destroys spatial topology, the model is effectively blind. 

---

## 🎯 Target Acquisition Logs

To further prove the hypothesis, the system generates visual inference grids. The neon frames indicate success (Green) or failure (Red).

### 1. Level 1: Fashion-MNIST (Linear Success)
![Fashion-MNIST Predictions](results/fashion-mnist_report.png)
> **Observation:** High accuracy. Mistakes are mathematically logical (e.g., confusing a Shirt with a Boot due to similar edge contours).

### 2. Level 2: STL-10 (Feature Confusion)
![STL-10 Predictions](results/stl-10_report.png)
> **Observation:** The model struggles significantly. Without convolutional filters to detect edges and shapes, it guesses based on dominant background colors (e.g., classifying a blue sky background as a "Ship" regardless of the object).

### 3. Level 3: EuroSAT (Topological Blindness)
![EuroSAT Predictions](results/eurosat_report.png)
> **Observation:** Critical failure. The linear weights cannot distinguish between the chaotic green pixels of a "Forest" and the organized green pixels of an "AnnualCrop".

---

## 🚀 Quick Start
To run this benchmark on your own machine and generate the Matrix-styled telemetry dashboards:

```bash
# 1. Clone the repository
git clone [https://github.com/Dalliya/vision-softmax-benchmarking.git](https://github.com/Dalliya/vision-softmax-benchmarking.git)
cd vision-softmax-benchmarking

# 2. Install dependencies (PyTorch & Matplotlib)
pip install -r requirements.txt

# 3. Initiate the Matrix
python main.py

Note: The script automatically handles dataset downloading and caching into the ./data directory. It uses MPS acceleration on Apple Silicon natively.

👩‍💻 About the Author
Dariia Zhdanova (@Dalliya)
ML Explorer | Architect of Neural Topology

I specialize in deconstructing complex Deep Learning concepts down to their mathematical foundations. I believe that true engineering isn't about calling model.fit(), but about understanding the exact geometry of the hyperplanes we build.

"In this study, I transitioned from manual mathematical foundations to automated linear stress-testing, proving that the massive 52.9% performance gap between structured silhouettes and chaotic satellite textures is the exact point where pure logic demands deeper neural connections."

📫 Connect with me:

GitHub: @Dalliya

LinkedIn: Dariia Zhdanova (https://www.linkedin.com/in/dariia-z-b7146223a)