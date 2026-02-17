import os
import shutil

import gymnasium as gym

from src.ml.data.embedding import bench_to_embed
from src.util.aig import bench_to_aig_file

STAGING_DIR = "data/staging"
AIG_PATH = os.path.join(STAGING_DIR, "circuit.bench")


class SimplyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,))

    def reset(self):
        return self.observation_space.sample(), {}

    def load_circuit(self, circuit_path: str):
        if os.path.exists(STAGING_DIR):
            for filename in os.listdir(STAGING_DIR):
                file_path = os.path.join(STAGING_DIR, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")

        shutil.copy(circuit_path, STAGING_DIR)
        self.mapping = bench_to_aig_file(circuit_path, AIG_PATH)
        self.struct_emb, self.func_emb = bench_to_embed(AIG_PATH)
