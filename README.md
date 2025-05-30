# F1 RL Agent – PPO Strategija

Ovaj projekt koristi PPO (Proximal Policy Optimization) za treniranje agenta koji optimizira strategiju utrkivanja (pit stop, izbor guma, stil vožnje) koristeći stvarne podatke iz FastF1 biblioteke.

---

## KORACI ZA POKRETANJE

### 1. Priprema i treniranje agenta

```bash
python prepare_data.py        # Generira podatke za treniranje (više vozača i trka)
python train.py               # Pokreće PPO treniranje s curriculum logikom
```

> Koriste se podaci iz trka: Silverstone, Monza, Spa (2023), vozači: VER, HAM, LEC, PER

---

### 2. Pokretanje REST API-ja

U korijenskom direktoriju pokreni FastAPI:

```bash
uvicorn api:app --reload
```

Alternativa:

```bash
python -m uvicorn api:app --reload
```

* API dostupan na: [http://127.0.0.1:8000](http://127.0.0.1:8000)
* Swagger sučelje: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### 3. Evaluacija i testiranje

#### U VS Code

* Otvori `main.ipynb` i izvrši sve ćelije (`Shift + Enter`)
* Provjeri točnost agenta i ponašanje po tipu vremena

#### Alternativa (Jupyter)

```bash
jupyter notebook
```

Zatim otvori `main.ipynb`.

---

## Funkcionalnosti

*  Višestazni podaci i više vozača
*  Imitacija ljudskih odluka (PIT, compound, stil)
*  Vremenski uvjeti: suho, mokro, kiša
*  Curriculum težina (easy → medium → full)
*  REST API s objašnjenjima odluka
*  Vizualizacije:

  * Reward po epizodi (smoothed)
  * Agent vs Vozač ponašanje
  * Reward po vremenskim uvjetima
  * Distribucija guma

---

## Evaluacija agenta

U `main.ipynb` dostupni su grafovi i točnost predikcije:

* `Točnost izbora PIT-a`
* `Točnost izbora GUMA`
* `Točnost STILA vožnje`
* Interpretacija akcije (npr. "PIT: DA, Gume: Inter, Stil: Štedljivo")

---

## REST API endpointi

| Endpoint                       | Opis                                              |
| ------------------------------ | ------------------------------------------------- |
| `GET /`                        | Test da je server pokrenut                        |
| `POST /predict_strategy`       | Predikcija strategije za jedno stanje             |
| `POST /predict_strategy_batch` | Predikcija strategije za više stanja (lista dict) |
| `POST /explain_strategy`       | Objašnjenje strategije na temelju ulaznog stanja  |

> Napomena: za `batch`, očekuje se: `[{"state": [...]}, {"state": [...]}]`

---

## Testiranje API-ja

Pokreni `main.ipynb` za:

* Slanje zahtjeva prema API-ju
* Provjeru interpretacija i odgovora
* Prikaz grafova po koraku (reward, akcije)

---

## Ovisnosti

Instaliraj sve potrebne pakete:

```bash
pip install -r requirements.txt
```

Dodatno za API:

```bash
pip install fastapi uvicorn "shimmy>=2.0"
```

---

## Struktura projekta

```
VS_Code verzija/
├\2500 api.py                   # FastAPI sučelje
├\2500 f1_env.py                # Okruženje s nagradnom funkcijom i vremenskim uvjetima
├\2500 model.py                 # LSTM feature extractor
├\2500 train.py                 # PPO treniranje s evaluacijama
├\2500 prepare_data.py          # Učitavanje i spremanje FastF1 podataka
├\2500 main.ipynb               # Evaluacija + API testiranje
├\2500 predict_strategies.py    # (opcionalno) batch testovi
├\2500 utils.py                 # Helper funkcije i značajke
├\2500 saved_models/            # Trenirani modeli
├\2500 data/                    # .npy i CSV podaci
└\2500 requirements.txt         # Python ovisnosti
```

---


