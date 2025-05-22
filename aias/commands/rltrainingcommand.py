import os
import json
import random
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Any, Dict, List
from aias.envs.procedural_conversation_env import ProceduralConversationEnv

class RLTrainingCommand:
    """
    A command to train a DQN-style conversational agent using
    a procedurally generated environment from past logs.
    """

    def __init__(self):
        cfg_path = Path(__file__).parents[1] / "config.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # Hyperparameters
        self.lr           = cfg.get("learning_rate", 0.001)
        self.max_eps      = cfg.get("max_iterations", 2000)
        self.gamma        = cfg.get("discount_factor", 0.99)
        self.batch_size   = cfg.get("batch_size", 16)
        self.epsilon_start= cfg.get("epsilon_start", 0.3)
        self.epsilon_end  = cfg.get("epsilon_end", 0.05)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = None
        self.opt    = None
        self.loss_fn= nn.MSELoss()

        # Replay buffer path
        self.replay_path = Path("memory/experience_replay.jsonl")
        self.replay_path.parent.mkdir(exist_ok=True, parents=True)

    def _build_model(self, s_dim: int, a_dim: int):
        class DQN(nn.Module):
            def __init__(self, s, a):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(s, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, a),
                )
            def forward(self, x):
                return self.net(x)

        self.model = DQN(s_dim, a_dim).to(self.device)
        self.opt   = optim.Adam(self.model.parameters(), lr=self.lr)

    def _sample_replay(self) -> List[Dict]:
        lines = self.replay_path.read_text().strip().splitlines()
        buffer = [json.loads(l) for l in lines]
        return random.sample(buffer, min(self.batch_size, len(buffer)))

    def execute(self, args: Any = None) -> None:
        # Instantiate the procedural environment
        env = ProceduralConversationEnv(
            logs_path="memory/logs.jsonl",
            embed_model_name="all-MiniLM-L6-v2",
            sample_size=200
        )
        s_dim, a_dim = env.state_size, env.action_size
        self._build_model(s_dim, a_dim)

        print(f"🔄 Starting RL training for {self.max_eps} episodes on {self.device}")
        for ep in range(1, self.max_eps + 1):
            state = env.reset().to(self.device)
            done = False
            transitions: List[Dict] = []
            episode_loss = 0.0

            # Decaying epsilon
            eps = max(
                self.epsilon_end,
                self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (ep / self.max_eps)
            )

            # Run episode
            while not done:
                if random.random() < eps:
                    action = random.randrange(a_dim)
                else:
                    with torch.no_grad():
                        q_vals = self.model(state)
                        action = torch.argmax(q_vals).item()

                next_state, reward, done, info = env.step(action)
                next_state = next_state.to(self.device)

                # Record transition
                transitions.append({
                    "state":      state.cpu().tolist(),
                    "action":     action,
                    "reward":     reward,
                    "next_state": next_state.cpu().tolist(),
                    "done":       done
                })

                state = next_state

            # Append episode to replay buffer
            with open(self.replay_path, "a", encoding="utf-8") as f:
                for t in transitions:
                    f.write(json.dumps(t) + "\n")

            # Sample a mini-batch and learn
            batch = self._sample_replay()
            if batch:
                states      = torch.tensor([d["state"]     for d in batch], dtype=torch.float32, device=self.device)
                actions     = torch.tensor([d["action"]    for d in batch], dtype=torch.int64, device=self.device)
                rewards     = torch.tensor([d["reward"]    for d in batch], dtype=torch.float32, device=self.device)
                next_states = torch.tensor([d["next_state"]for d in batch], dtype=torch.float32, device=self.device)
                dones       = torch.tensor([d["done"]      for d in batch], dtype=torch.float32, device=self.device)

                q_vals = self.model(states)
                q_sel  = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q = self.model(next_states).max(1).values
                    targets = rewards + self.gamma * next_q * (1 - dones)

                loss = self.loss_fn(q_sel, targets)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                episode_loss = loss.item()

            if ep % 100 == 0:
                print(f"Episode {ep}/{self.max_eps}, loss={episode_loss:.4f}, ε={eps:.3f}")

        print("✅ Training complete.")

    def clean_up(self, args: Any = None) -> None:
        out_dir = Path(__file__).parents[1] / "models"
        out_dir.mkdir(exist_ok=True)
        save_path = out_dir / "dqn_model.pth"
        torch.save(self.model.state_dict(), save_path)
        print(f"💾 Model saved to {save_path}")
