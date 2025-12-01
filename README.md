
# Atlas Vision – ShotQuality-Style Rebounding (SportsRadar)

This repo runs an end-to-end pipeline for **one NBA game**:

- Pulls **official PBP** (all missed shots + rebound type) from **SportsRadar**.
- Aligns each missed shot to a **broadcast video** (`game.mp4`).
- Uses **YOLO** to detect players + ball at shot time.
- Maps players to **94×50 court coordinates** and builds ShotQuality-style features.
- Trains a **rebound model** on the ShotQuality Kaggle dataset and predicts **P(OREB)** for every missed shot.
- Outputs a game-level CSV: `outputs/game_shotquality_cv.csv`.

---

## Setup (to run in collab)

1. **Clone / download the repo zip**

   ```bash
   git clone https://github.com/karanwxliaa/atlas-vision-shotquality-sportsradar.git
   cd atlas-vision-shotquality-sportsradar

**Make sure to upload these 2 files to your google drive for this to work. 
1) the repo's zip
2) game.zip (attached in my email)**

Then follow the steps in the tutorial notebook further.

This will:

1. Fetch PBP from SportsRadar and build `data/game_pbp_mapped.csv`.
2. Run CV on `game.mp4`, build `outputs/work/nba_locs.csv` and `nba_features_raw.csv`.
3. Train the rebound model from Kaggle (or reuse a cached one).
4. Write the final per-shot CSV to:

```text
outputs/game_shotquality_cv.csv
```
