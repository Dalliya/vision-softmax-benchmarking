import torch
import torch.optim as optim
import torch.nn.functional as F

class SoftmaxClassifier:
    """Manual implementation of Softmax Regression for stress-testing linear logic."""
    
    def __init__(self, input_dim: int, num_classes: int, lr: float = 0.1, device: str = 'cpu'):
        self.device = device
        
        # Initialize weights and biases from scratch
        # Small standard deviation prevents initial numerical singularities
        self.W = torch.randn(input_dim, num_classes, device=device).mul_(0.01).requires_grad_()
        self.b = torch.zeros(num_classes, device=device).requires_grad_()
        
        # Optimization engine (Stochastic Gradient Descent)
        self.optimizer = optim.SGD([self.W, self.b], lr=lr)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Matrix multiplication with automated flattening.
        Transforms [Batch, C, H, W] -> [Batch, C*H*W] before multiplication.
        """
        X_flat = X.view(X.size(0), -1) 
        return X_flat @ self.W + self.b

    def cross_entropy(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Numerically stable Cross-Entropy loss with an epsilon buffer."""
        # EPSILON trick prevents log(0) exceptions during mode collapse
        probs = F.softmax(logits, dim=1) + 1e-8
        target_probs = torch.gather(probs, 1, y.view(-1, 1)).squeeze()
        return -torch.log(target_probs).mean()

    def step(self):
        """Perform manual backpropagation and SGD optimization step."""
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def train(self): 
        """Enable gradient tracking."""
        self.W.requires_grad_(True)
        self.b.requires_grad_(True)
        
    def eval(self): 
        """Disable gradient tracking for inference/evaluation."""
        self.W.requires_grad_(False)
        self.b.requires_grad_(False)