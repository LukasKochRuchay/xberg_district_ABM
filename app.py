"""Refactoring entrypoint (CLI-only).

Visualization has been removed; run the model via:

  cd models/refactoring
  python app.py --help
  python app.py --steps 288
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import argparse

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
  sys.path.insert(0, str(THIS_DIR))

from src.model.model import BikePedModel  # type: ignore[reportMissingImports]


def build_parser():
  parser = argparse.ArgumentParser(
    prog="gabm-refactoring",
    description="Run the bike-vs-car stress model and write per-agent history.",
  )

  parser.add_argument("--data-crs", default="epsg:4326", help="CRS of input data")
  parser.add_argument(
    "--model-crs",
    default="epsg:3857",
    help="CRS used internally for routing/space",
  )

  parser.add_argument(
    "--output-crs",
    default="epsg:4326",
    help="CRS for CSV x/y output. Defaults to epsg:4326 (WGS84).",
  )

  parser.add_argument(
    "--output-format",
    default="csv",
    choices=["csv", "geojson"],
    help="Output format for agent history (csv or geojson).",
  )

  parser.add_argument(
     "--buildings-file",
      default="data/district_bld.zip",
    help="Path (relative to models/refactoring/) to buildings dataset",
  )
  parser.add_argument(
     "--walkways-file",
      default="data/district_walkway_line.zip",
    help="Path (relative to models/refactoring/) to walking network lines",
  )
  parser.add_argument(
     "--bikeways-file",
      default="data/district_bikeway_line.zip",
    help="Path (relative to models/refactoring/) to biking network lines",
  )
  parser.add_argument(
    "--output-dir",
    default="data/outputs",
    help="Output directory (relative to models/refactoring/) for agent_history.csv",
  )
  parser.add_argument(
    "--output-name",
    default="agent_history",
    help="Base filename for simulation output, without extension.",
  )

  parser.add_argument("--num-commuters", type=int, default=50)
  parser.add_argument("--steps", type=int, default=288, help="Number of 5-min ticks to run")

  parser.add_argument(
    "--bike-speed",
    "--walk-speed",
    dest="bike_speed",
    type=float,
    default=300.0,
    help="Bike speed (m per tick). Legacy alias: --walk-speed.",
  )
  parser.add_argument(
    "--car-speed",
    dest="car_speed",
    type=float,
    default=600.0,
    help="Car speed (m per tick).",
  )
  parser.add_argument(
    "--epsilon",
    type=float,
    default=0.15,
    help="Probability of exploratory mode choice instead of following the lower-stress mode.",
  )
  parser.add_argument(
    "--alpha",
    type=float,
    default=0.6,
    help="Learning rate for expected stress updates after each trip; 0 ignores new experience, 1 fully replaces prior expectations.",
  )

  parser.add_argument(
    "--initial-car-share",
    type=float,
    default=0.8,
    help="Initial share of commuters seeded as car users (0..1)",
  )
  # Backwards compatibility with previous PoC runs.
  parser.add_argument(
    "--initial-bike-share",
    dest="initial_car_share",
    type=float,
    help=argparse.SUPPRESS,
  )
  parser.add_argument(
    "--stress-bin-size-m",
    "--crowding-bin-size-m",
    dest="stress_bin_size_m",
    type=float,
    default=25.0,
    help="Bin size (meters) used to approximate co-location stress exposure.",
  )
  parser.add_argument(
    "--bike-sweet-spot",
    type=int,
    default=5,
    help="Number of nearby cyclists that lowers bike stress before overcrowding begins.",
  )
  parser.add_argument(
    "--overcrowding-weight",
    type=float,
    default=0.5,
    help="Rate at which bike stress rises again beyond the bike sweet spot.",
  )
  parser.add_argument(
    "--bike-social-benefit-weight",
    type=float,
    default=2.0,
    help="Strength of stress reduction from nearby bikes before the sweet spot.",
  )
  parser.add_argument(
    "--bike-overcrowding-exponent",
    type=float,
    default=1.5,
    help="Nonlinear exponent for bike overcrowding penalty beyond the sweet spot.",
  )

  parser.add_argument(
    "--car-congestion-threshold",
    "--car-sweet-spot",
    dest="car_congestion_threshold",
    type=int,
    default=3,
    help="Nearby car count above which car users start accumulating congestion stress.",
  )
  parser.add_argument(
    "--car-overcrowding-weight",
    type=float,
    default=0.3,
    help="Rate at which car stress rises beyond the car sweet spot (congestion penalty).",
  )
  parser.add_argument(
    "--car-bike-traffic-weight",
    type=float,
    default=0.8,
    help="Stress contribution from nearby bikes to car users.",
  )
  parser.add_argument(
    "--car-bike-crowding-threshold",
    type=int,
    default=1,
    help="Nearby bike count threshold before car users start feeling bike-induced stress.",
  )

  parser.add_argument(
    "--car-distance-threshold-m",
    type=float,
    default=5000.0,
    help="Distance threshold (m). Beyond this, car becomes more likely.",
  )
  parser.add_argument(
    "--car-prob-below-threshold",
    type=float,
    default=0.1,
    help="Base probability of choosing car when distance <= threshold.",
  )
  parser.add_argument(
    "--car-prob-max",
    type=float,
    default=0.9,
    help="Maximum probability of choosing car for long distances.",
  )
  parser.add_argument(
    "--car-prob-ramp-m",
    type=float,
    default=5000.0,
    help="Ramp length (m) beyond threshold to reach car-prob-max.",
  )

  parser.add_argument("--start-day", type=int, default=0)
  parser.add_argument("--start-hour", type=int, default=5)
  parser.add_argument("--start-minute", type=int, default=55)

  parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Optional RNG seed (best-effort; depends on Mesa version)",
  )

  parser.add_argument(
    "--log-interval",
    type=int,
    default=12,
    help="Log mode-share summary every N steps (default: 12 = once per simulated hour).",
  )
  parser.add_argument(
    "--log-level",
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    help="Logging verbosity",
  )

  return parser


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)

  logging.basicConfig(
    level=getattr(logging, str(args.log_level).upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
  )

  # Expect to run from models/refactoring so defaults like data/bld.zip work.
  base_dir = Path.cwd()

  buildings_file = base_dir / args.buildings_file
  walkways_file = base_dir / args.walkways_file
  bikeways_file = base_dir / args.bikeways_file
  output_dir = base_dir / args.output_dir

  model = BikePedModel(
    district="district",
    data_crs=args.data_crs,
    buildings_file=buildings_file,
    walkways_file=walkways_file,
    bikeways_file=bikeways_file,
    output_dir=output_dir,
    output_name=args.output_name,
    num_commuters=args.num_commuters,
    commuter_bike_speed_m_per_tick=args.bike_speed,
    commuter_car_speed_m_per_tick=args.car_speed,
    commuter_mode_choice_epsilon=args.epsilon,
    commuter_stress_ewma_alpha=args.alpha,
    bike_sweet_spot=args.bike_sweet_spot,
    overcrowding_weight=args.overcrowding_weight,
    bike_social_benefit_weight=args.bike_social_benefit_weight,
    bike_overcrowding_exponent=args.bike_overcrowding_exponent,
    car_congestion_threshold=args.car_congestion_threshold,
    car_overcrowding_weight=args.car_overcrowding_weight,
    car_bike_traffic_weight=args.car_bike_traffic_weight,
    car_bike_crowding_threshold=args.car_bike_crowding_threshold,
    initial_car_share=args.initial_car_share,
    stress_bin_size_m=args.stress_bin_size_m,
    car_distance_threshold_m=args.car_distance_threshold_m,
    car_prob_below_threshold=args.car_prob_below_threshold,
    car_prob_max=args.car_prob_max,
    car_prob_ramp_m=args.car_prob_ramp_m,
    model_crs=args.model_crs,
    output_crs=args.output_crs,
    output_format=args.output_format,
    start_day=args.start_day,
    start_hour=args.start_hour,
    start_minute=args.start_minute,
  )
  model.log_interval = args.log_interval

  # Mesa seed support varies; set if available.
  if args.seed is not None:
    try:
      model.reset_randomizer(args.seed)
    except Exception:
      pass

  for _ in range(args.steps):
    model.step()

  if hasattr(model, "finalize"):
    model.finalize()

  suffix = ".csv" if args.output_format == "csv" else ".geojson"
  out_name = f"{args.output_name}{suffix}"
  print(str((Path(model.output_dir) / out_name).resolve()))
  return 0


# Instantiate once so `python app.py --help` is fast and to make the available
# arguments visible at this entrypoint.
PARSER = build_parser()


if __name__ == "__main__":
  raise SystemExit(main())

