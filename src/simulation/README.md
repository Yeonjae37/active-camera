# simulation

Isaac Sim camera sweeps of an AMASS walking motion and 3D knee-flexion evaluation (NLF).

| step | where | how |
|---|---|---|
| 1. motion → SMPL mesh + joints | `export_motion.py` | `python -m src.simulation.export_motion --motion <AMASS npz>` (needs `models/smpl`) |
| 2. load into the scene | `isaac/load_motion.py` | Isaac Sim Script Editor |
| 3. render a sweep | `isaac/capture_sweep.py` | Script Editor, set `PRESET` (`pan_tilt`, `rail`, `orbit`) |
| 4. evaluate | `analyze.py` | `python -m src.simulation.analyze output/02_rail_viewpoint/capture` |
| 5. photo comparison | `figures.py` | `python -m src.simulation.figures output/03_orbit_azimuth/capture` |

Library modules: `common.py` (sweep loading, projection, flexion, gait phase, drawing), `estimators.py` (NLF,
cached inference), `planning.py` (policies, motion-constrained rail oracle). Outputs: `output/<experiment>/{capture,analysis}/`.
