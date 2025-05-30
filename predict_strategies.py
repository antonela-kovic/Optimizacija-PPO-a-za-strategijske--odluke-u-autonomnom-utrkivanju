# === predict_strategies.py ===
# === Nadogradnja 4: Predikcija strategija drugih vozača ===
# Dodajemo novu funkciju koja koristi trenirani agent da predvidi strategiju za stvarne lapove
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from utils import selected_features

# Učitaj model i podatke
model = PPO.load("saved_models/ppo_f1_rl.zip")
obs_array = np.load("data/obs_array.npy")
true_actions = np.load("data/true_actions.npy")

# Predikcije
predicted_actions = []
for obs in obs_array:
    action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
    action = action[0] if isinstance(action[0], (list, np.ndarray)) else action
    predicted_actions.append(action)

predicted_actions = np.array(predicted_actions)

# Spremi kao CSV
df = pd.DataFrame(predicted_actions, columns=["Pit", "Compound", "Style"])
df.to_csv("data/predicted_strategies.csv", index=False)
print("Strategije spremljene u data/predicted_strategies.csv")

# === Evaluacija točnosti ===
acc_pit = accuracy_score(true_actions[:, 0], predicted_actions[:, 0])
acc_comp = accuracy_score(true_actions[:, 1], predicted_actions[:, 1])
acc_style = accuracy_score(true_actions[:, 2], predicted_actions[:, 2])

print(f"Točnost predikcije:")
print(f"Pit stop:     {acc_pit:.2%}")
print(f"Compound:     {acc_comp:.2%}")
print(f"Driving style:{acc_style:.2%}")

# === Grafikon distribucije ===
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(predicted_actions[:, 0], bins=np.arange(-0.5, 2), rwidth=0.6)
plt.title("Pit stop odluke")
plt.xticks([0, 1])
plt.xlabel("Pit (0/1)")

plt.subplot(1, 3, 2)
plt.hist(predicted_actions[:, 1], bins=np.arange(-0.5, 6), rwidth=0.6)
plt.title("Odabrani compoundi")
plt.xlabel("Compound kod")

plt.subplot(1, 3, 3)
plt.hist(predicted_actions[:, 2], bins=np.arange(-0.5, 4), rwidth=0.6)
plt.title("Stil vožnje")
plt.xlabel("Stil (0-2)")

plt.tight_layout()
plt.show()
