# Formula 1 Data Science Experiments

This repository contains experiments using the `fastf1` library.

## Predicting Canadian GP 2025

A new script `predict_canadian_gp_2025.py` trains a Random Forest model on
2024 race results and estimates win probabilities for the 2025 Canadian Grand
Prix using qualifying data. It requires `fastf1` and scikit-learn.

To run:

```bash
python3 predict_canadian_gp_2025.py
```

The script will print predicted win probabilities for all drivers in the
qualifying results.
