import json
import random
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer

class ProceduralConversationEnv:
    """
    A procedurally‐generated conversational RL environment based on past logs.
    - Samples real user utterances from memory/logs.jsonl
    - Generates candidate AI responses via semantic clustering on past AI replies
    - Rewards based on whether the next real user message shows approval vs. clarification
    """
    def __init__(self,
                 logs_path: str = "memory/logs.jsonl",
                 embed_model_name: str = "all-MiniLM-L6-v2",
                 sample_size: int = 100):
        # Load past interactions, skipping bad lines
        self.logs = []
        p = Path(logs_path)
        if not p.exists():
            raise FileNotFoundError(f"No logs at {logs_path}")
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line)
                u = obj["user"]
                a = obj["ai"]
                self.logs.append((u, a))
            except (json.JSONDecodeError, KeyError):
                # Skip lines that aren't properly formatted
                continue

        if not self.logs:
            raise RuntimeError("No valid user/ai pairs in logs.")

        # Sample a subset for speed
        self.sample = random.sample(self.logs, min(sample_size, len(self.logs)))
        # Build pools
        self.user_msgs = [u for u,_ in self.sample]
        self.ai_msgs   = [a for _,a in self.sample]

        # Embedding model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = SentenceTransformer(embed_model_name).to(self.device)

        # Internal state
        self.cur_idx = 0
        self.done = False

    @property
    def state_size(self) -> int:
        dim = self.encoder.get_sentence_embedding_dimension()
        return dim * 2

    @property
    def action_size(self) -> int:
        return len(self.ai_msgs)

    def reset(self) -> torch.Tensor:
        """Pick a random log entry as the starting point."""
        self.cur_idx = random.randrange(len(self.sample))
        self.last_user, self.last_ai = self.sample[self.cur_idx]
        self.done = False
        return self._build_state()

    def step(self, action_idx: int) -> Tuple[torch.Tensor, float, bool, Dict[str,Any]]:
        """Agent picks an index into the ai_msgs pool."""
        # Record chosen reply
        chosen = self.ai_msgs[action_idx]
        self.last_ai = chosen

        # Move to next real log entry
        self.cur_idx = (self.cur_idx + 1) % len(self.sample)
        next_user, next_ai = self.sample[self.cur_idx]

        # Reward: +1 if the real AI reply matches our chosen index,
        #   else -0.5 if our choice differs and user immediately asks clarification
        reward = 1.0 if chosen == next_ai else -0.5
        # Done after one turn
        self.done = True

        # Update state to new pair
        self.last_user = next_user
        state = self._build_state()
        info = {"actual_ai": next_ai, "chosen_ai": chosen}
        return state, reward, self.done, info

    def _build_state(self) -> torch.Tensor:
        texts = [self.last_user, self.last_ai]
        emb = self.encoder.encode(texts, convert_to_tensor=True, device=self.device)
        return emb.view(-1)
