import random
import torch
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer

class UserSimulator:
    """
    A very simple user simulator: pulls from a list of predefined utterances,
    cycling through them. You can replace this with a more sophisticated model.
    """
    def __init__(self, script: List[str]):
        self.script = script
        self.idx = 0

    def next_utterance(self) -> str:
        utt = self.script[self.idx % len(self.script)]
        self.idx += 1
        return utt

class ConversationEnv:
    """
    A Gym-style environment for conversational RL.
    - state: embedding of [last user utterance, last AI response]
    - action: index into a discrete set of canned responses
    - reward: +1 if the AI’s choice “matches” an expected response, else 0
    """
    def __init__(self,
                 user_script: List[str],
                 ai_responses: List[str],
                 expected_map: Dict[str, List[int]],
                 embed_model_name: str = "all-MiniLM-L6-v2"):
        self.user = UserSimulator(user_script)
        self.ai_responses = ai_responses
        self.expected = expected_map
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Embedding model for states
        self.encoder = SentenceTransformer(embed_model_name).to(self.device)
        # Initialize the “last” messages
        self.last_user = ""
        self.last_ai = ""
        self.done = False

    @property
    def state_size(self) -> int:
        emb_dim = self.encoder.get_sentence_embedding_dimension()
        return emb_dim * 2

    @property
    def action_size(self) -> int:
        return len(self.ai_responses)

    def reset(self) -> torch.Tensor:
        """Start a new episode; return initial state."""
        self.last_user = self.user.next_utterance()
        self.last_ai = ""
        self.done = False
        return self._build_state()

    def step(self, action_idx: int) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]:
        """
        Take an action (index into ai_responses).
        Returns (next_state, reward, done, info).
        """
        self.last_ai = self.ai_responses[action_idx]

        # Compute reward: 1 if action in expected list for this user utterance
        good_actions = self.expected.get(self.last_user, [])
        reward = 1.0 if action_idx in good_actions else 0.0

        # Advance to next user utterance
        next_user = self.user.next_utterance()
        # For simplicity, end the episode after one turn; you can extend this
        self.done = True

        # Update state
        self.last_user = next_user
        next_state = self._build_state()
        info = {"last_reward": reward, "user_utt": next_user}
        return next_state, reward, self.done, info

    def _build_state(self) -> torch.Tensor:
        """
        Encode [last_user, last_ai] into a single tensor of shape (state_size,).
        """
        texts = [self.last_user, self.last_ai or ""]
        embeddings = self.encoder.encode(texts, convert_to_tensor=True, device=self.device)
        # embeddings shape: (2, emb_dim)
        return embeddings.view(-1)

if __name__ == "__main__":
    # Smoke test for the environment
    user_script = [
        "Hi AIAS, how are you?",
        "Can you locate agent.py?",
        "Thanks, bye!"
    ]
    ai_responses = [
        "Hello! I’m doing great, thanks for asking.",
        "Sure—agent.py is in aias/agent.py.",
        "You’re welcome! Talk to you later."
    ]
    expected_map = {
        user_script[0]: [0],
        user_script[1]: [1],
        user_script[2]: [2]
    }

    env = ConversationEnv(user_script, ai_responses, expected_map)
    state = env.reset()
    print("Initial state shape:", state.shape)

    action = random.randrange(env.action_size)
    next_state, reward, done, info = env.step(action)

    print(f"Chose action #{action}: \"{ai_responses[action]}\"")
    print("Next state shape:", next_state.shape)
    print("Reward:", reward)
    print("Done flag:", done)
    print("Info dict:", info)
