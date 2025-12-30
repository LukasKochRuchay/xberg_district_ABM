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


