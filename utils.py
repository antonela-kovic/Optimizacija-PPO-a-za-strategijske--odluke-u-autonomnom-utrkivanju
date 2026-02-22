# === utils.py ===
import numpy as np

# Može se ukloniti ova funkcija(generate_weather_sequence) jer je trenutno train.py prešao na Weather 
# kao značajku u prepare_data.py te se više ne koristi u f1_env.py.

# def generate_weather_sequence(num_laps):
#     weather = [1]
#     for _ in range(1, num_laps):
#         last = weather[-1]
#         change = np.random.choice([-1, 0, 1], p=[0.1, 0.8, 0.1])
#         new = min(max(last + change, 1), 3)
#         weather.append(new)
#     return weather

# === utils.py ===
selected_features = [
    "TyreLife", "CompoundEncoded", "LapNumber", "TrackStatus",
    "IsAccurate", "Position", "LapTime", "Sector1Time", "Sector2Time",
    "FrontGap", "RearGap", "Stint", "TrackTemperature", "AirTemperature",
    "Weather"  # dodajemo jer se Weather sada gleda kao zančajka i ne uzima se iz funkcije iznad
]


# Ako pokrecemo test_env.py za brzo testiranje
def load_obs_array(path="data/obs_array.npy"):
    return np.load(path)

def load_true_actions(path="data/true_actions.npy"):
    return np.load(path)
