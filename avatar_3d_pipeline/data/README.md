# Data Layout for Fine-Tuning

Put paired files into these folders:
- `selfies/`: input selfie images
- `albedo/`: target albedo maps
- `normal/`: target normal maps

Pairs are matched by the same filename stem.

Example:
- `selfies/0001.jpg`
- `albedo/0001.png`
- `normal/0001.png`

You can train in two ways:
1. Auto-scan folders (no manifest):
   `python training/fine_tune_texture.py --data-root ./data --weights-dir ./weights`
2. Manifest mode:
   Keep `manifest.csv` with rows like
   `selfies/0001.jpg,albedo/0001.png,normal/0001.png`
   then run
   `python training/fine_tune_texture.py --data-root ./data --manifest ./data/manifest.csv --weights-dir ./weights`
