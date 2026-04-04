import torch, os, gc
from models.engine import SoftmaxClassifier
from utils.data_handlers import get_loaders, get_data_config
from utils.visualizer import Visualizer
from utils.snr_analytics import calculate_dataset_snr, plot_snr_correlation

def run_bench(name: str):
    """Executes the training and evaluation pipeline for a specific dataset."""
    train_ld, test_ld, dim, n_cls = get_loaders(name, 128)
    names = get_data_config(name)["names"]
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = SoftmaxClassifier(dim, n_cls, device=device)
    hist = {"loss": [], "acc": []}

    # Standard training loop
    for ep in range(5):
        model.train(); l_sum, ok, total = 0, 0, 0
        for X, y in train_ld:
            X, y = X.to(device), y.to(device)
            out = model.forward(X); loss = model.cross_entropy(out, y)
            loss.backward(); model.step()
            l_sum += loss.item()
            ok += (torch.argmax(out, 1) == y).sum().item()
            total += y.size(0)

        epoch_metrics = {"loss": l_sum / len(train_ld), "acc": 100 * ok / total}
        hist["loss"].append(epoch_metrics["loss"])
        hist["acc"].append(epoch_metrics["acc"])
        print(f"[{name.upper()}] Ep {ep+1} | Acc: {epoch_metrics['acc']:.1f}% | Loss: {epoch_metrics['loss']:.2f}")

    # Inference and visualization
    model.eval()
    with torch.no_grad():
        X, y = next(iter(test_ld))
        p = torch.argmax(model.forward(X.to(device)), 1)
        Visualizer.plot_prediction_grid(X, y, p.cpu(), names, f"results/{name}_report.png", name, hist['acc'][-1], hist['loss'][-1])
    
    # Quantitative noise analysis
    print(f"[{name.upper()}] Calculating Signal-to-Noise Ratio...")
    hist['snr'] = calculate_dataset_snr(test_ld)
    
    # Memory management
    del model, train_ld, test_ld; gc.collect()
    torch.mps.empty_cache() if device.type == 'mps' else None
    return hist

if __name__ == "__main__":
    # Fix random seeds for reproducible and stable results
    torch.manual_seed(23)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(23)

    os.makedirs("results", exist_ok=True)
    print("--- COMPUTES INITIALIZED ---")
    
    AVAILABLE_DS = ["fashion-mnist", "stl-10", "eurosat"] 
    all_results = {}
    
    for ds in AVAILABLE_DS:
        try:
            print(f"\n[SYSTEM] Loading {ds.upper()}...")
            session_data = run_bench(ds)
            if session_data: 
                all_results[ds] = session_data
        except Exception as e:
            print(f"[CRITICAL ERROR] Failed to process {ds.upper()}: {e}")

    # Final analytics and visualization
    if all_results:
        Visualizer.plot_telemetry_dashboard(all_results)
        
        print("\n>> INITIATING SNR CORRELATION ANALYSIS...")
        datasets_list = list(all_results.keys())
        snr_list = [all_results[ds]['snr'] for ds in datasets_list]
        acc_list = [all_results[ds]['acc'][-1] for ds in datasets_list]
        
        plot_snr_correlation(datasets_list, snr_list, acc_list)
        print(">>> ALL SYSTEMS NOMINAL. Results exported to 'results/' folder.")

        Visualizer.show_all()