import torch
from pathlib import Path

class InspectModelCommand:
    """
    Load the latest DQN model checkpoint and report per‐layer
    parameter statistics (mean, std, shape).
    """

    def __init__(self, model_path: str = None):
        # Default to the standard DQN path
        if model_path:
            self.model_path = Path(model_path)
        else:
            self.model_path = Path(__file__).parents[1] / "models" / "dqn_model.pth"

    def execute(self, args=None) -> str:
        if not self.model_path.exists():
            return f"❌ Model not found at {self.model_path}"

        stats = []
        ckpt = torch.load(self.model_path, map_location="cpu")
        for name, tensor in ckpt.items():
            m = float(tensor.mean().item())
            s = float(tensor.std().item())
            shp = tuple(tensor.shape)
            stats.append(f"{name} → shape={shp}, mean={m:.4f}, std={s:.4f}")

        return "🔍 Model parameter stats:\n" + "\n".join(stats)
