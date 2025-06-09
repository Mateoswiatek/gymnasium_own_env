import torch
from main import CustomNetwork
import gymnasium as gym

# Utwórz przykładową sieć (np. architektura "deep")
env = gym.make("LunarLanderContinuous-v3")
model = CustomNetwork(env.observation_space, features_dim=256, architecture="wide")

# Zapisz model do pliku
torch.save(model, "network_wide.pt")