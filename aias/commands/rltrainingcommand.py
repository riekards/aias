import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Any, Dict

class RLTrainingCommand:
    """
    A command to train a simple DQN-style agent using configuration
    parameters defined in aias/config.yaml.
    """

    def __init__(self):
        # Load RL hyperparameters from config.yaml
        cfg_path = Path(__file__).parents[1] / "config.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        rl_cfg = cfg.get("ml_algorithms", {})
        self.learning_rate = cfg.get("learning_rate", 0.001)
        self.max_iterations = cfg.get("max_iterations", 1000)
        self.state_size = cfg.get("state_size", 10)       # default placeholder
        self.action_size = cfg.get("action_size", 4)      # default placeholder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define a simple feed-forward network
        class DQN(nn.Module):
            def __init__(self, s_dim, a_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(s_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, a_dim)
                )
            def forward(self, x):
                return self.net(x)

        self.model = DQN(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def execute(self, args: Any) -> None:
        """
        Run the training loop. `args` can supply any environment or data loader.
        """
        print(f"🔄 Starting RL training for {self.max_iterations} iterations on {self.device}")
        for episode in range(self.max_iterations):
            # Placeholder: generate dummy state/action/reward/next_state
            state = torch.randn(self.state_size, device=self.device)
            next_state = torch.randn(self.state_size, device=self.device)
            action = torch.randint(0, self.action_size, (1,), device=self.device)
            reward = torch.randn(1, device=self.device)
            done = torch.rand(1).item() > 0.95

            # Predict Q-values and compute target
            q_values = self.model(state)
            q_value = q_values[action]

            with torch.no_grad():
                next_q_values = self.model(next_state)
                max_next_q = next_q_values.max()
                target = reward + (0.99 * max_next_q * (1 - int(done)))

            # Compute loss & backprop
            loss = self.loss_fn(q_value, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if episode % 100 == 0:
                print(f"Episode {episode}/{self.max_iterations}, loss={loss.item():.4f}")

        print("✅ Training complete.")

    def clean_up(self, args: Any) -> None:
        """
        Save the trained model to disk.
        """
        out_dir = Path(__file__).parents[1] / "models"
        out_dir.mkdir(exist_ok=True)
        model_path = out_dir / "dqn_model.pth"
        torch.save(self.model.state_dict(), model_path)
        print(f"💾 Model saved to {model_path}")
