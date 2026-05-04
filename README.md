CLI-first, research-oriented agent-based mobility model (ABM) built with GeoMesa.

**What it does**
- Loads a district building layer (homes/workplaces) and two line-network layers.
- Simulates commuters moving between home and work.
- Logs per-agent trajectories and mode over time to `CSV` (or `GeoJSON`).

**Quickstart**
1. Install dependencies:
	- `pip install -r requirements.txt`
2. Run the refactoring model:
	- Example run (500 commuters, 2000 steps, WGS84 output, CSV, 50% initial car share):
	  - `python app.py --num-commuters 500 --steps 2000 --output-crs epsg:4326 --output-format csv --initial-car-share 0.5`

**Outputs**
- CSV: [data/outputs/agent_history.csv](models/refactoring/data/outputs/agent_history.csv)
- GeoJSON (optional): [data/outputs/agent_history.geojson](models/refactoring/data/outputs/agent_history.geojson)

## Parameter Cheat Sheet

Parameters are organized into four logical groups:

- **Global**: Population size (`--num-commuters`), simulation length (`--steps`), learning rates (`--alpha`, `--epsilon`), initial conditions (`--initial-car-share`), and shared spatial resolution (`--stress-bin-size-m`).
- **Bike**: Cycling experience parameters including sweet spot, social benefit, overcrowding penalty, nonlinearity, and car-induced stress.
- **Car**: Driving experience parameters including congestion threshold, overcrowding penalty, bike-induced stress, and distance-based mode probability rules.
- **Data & Output**: Input file paths, output format, CRS settings, logging, and simulation clock (`--start-day`, `--start-hour`, `--start-minute`).

Command used:

`python app.py --num-commuters 1000 --steps 10000 --initial-car-share 0.3 --bike-speed 520 --car-speed 520 --bike-sweet-spot 10 --bike-social-benefit-weight 5.0 --bike-overcrowding-weight 0.0 --bike-overcrowding-exponent 1.0 --bike-car-traffic-weight 3 --car-congestion-threshold 2 --car-overcrowding-weight 3.0 --car-bike-traffic-weight 1.5 --car-bike-crowding-threshold 1 --epsilon 0.01 --alpha 0.01 --stress-bin-size-m 220 --car-prob-below-threshold 0.0 --car-prob-max 0.0 --car-prob-ramp-m 1 --output-name 1K_80bikeshareagents_seed42 --output-format daily_stats --seed 42 --log-level INFO`

| Parameter | What it controls | Interpretation for this run |
|---|---|---|
| **Global** |
| `--num-commuters 1000` | Number of simulated agents | Medium-sized population; avoids extreme density saturation. |
| `--steps 10000` | Simulation length (5-min ticks) | ~34.7 simulated days (`10000 / 288`). |
| `--initial-car-share 0.3` | Initial fraction of car users | Starts at 30% car / 70% bike. |
| `--epsilon 0.01` | Exploration probability (random mode choice) | Very low randomness; mostly deterministic choice. |
| `--alpha 0.01` | Learning rate for EWMA expectations | Slow learning; gradual adaptation and smoother dynamics. |
| `--bike-speed 520` | Bike movement speed per tick | Bikes and cars are equal-speed in this setup. |
| `--car-speed 520` | Car movement speed per tick | No travel-time advantage for cars. |
| `--stress-bin-size-m 220` | Spatial bin size for co-location stress sampling | Coarse neighborhood interaction radius. |
| **Bike** |
| `--bike-sweet-spot 10` | Bike count before crowding penalty starts | Bike social benefit grows up to ~10 nearby cyclists. |
| `--bike-social-benefit-weight 5.0` | Per-step bike stress reduction from nearby cyclists | Moderate in-trip safety-in-numbers effect while traveling. |
| `--bike-overcrowding-weight 0.0` | Bike overcrowding penalty strength | Overcrowding is disabled for bikes. |
| `--bike-overcrowding-exponent 1.0` | Nonlinearity of bike overcrowding penalty | Linear penalty if overcrowding were enabled. |
| `--bike-car-traffic-weight 3` | Car presence penalty felt by cyclists | Cars strongly increase bike stress. |
| **Car** |
| `--car-congestion-threshold 2` | Car count before car congestion stress starts | Cars start congesting quickly (after 2 nearby cars). |
| `--car-overcrowding-weight 3.0` | Congestion penalty among cars | Strong car-car congestion penalty. |
| `--car-bike-traffic-weight 1.5` | Bike presence penalty felt by drivers | Drivers are moderately stressed by many cyclists. |
| `--car-bike-crowding-threshold 1` | Bike count before drivers feel bike pressure | Driver bike-related stress starts almost immediately. |
| `--car-prob-below-threshold 0.0` | Distance-rule baseline car probability | Distance rule is effectively disabled. |
| `--car-prob-max 0.0` | Max car probability from distance rule | No forced car adoption at long distance. |
| `--car-prob-ramp-m 1` | Distance ramp length to max car probability | Irrelevant here because max is 0. |
| **Data & Output** |
| `--output-name 1K_80bikeshareagents_seed42` | Output file base name | Produces a run-specific output filename. |
| `--output-format daily_stats` | Output schema | Writes compact daily aggregates (shares/switches). |
| `--seed 42` | RNG seed | Reproducible stochastic behavior for this scenario. |
| `--log-level INFO` | Logging verbosity | Prints hourly-like run diagnostics and status counts. |

### Quick interpretation of this configuration

- Starts bike-heavy (70% bike), then lets behavior adapt slowly (`alpha=0.01`).
- Removes distance-based car forcing (`car-prob-*` all effectively off).
- Keeps strong mutual interaction stress (cars stress bikes; bikes stress cars).
- Uses compact output for analysis (`daily_stats`) instead of huge per-agent trajectory logs.


