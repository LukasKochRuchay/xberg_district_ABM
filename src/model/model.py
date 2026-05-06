from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import logging
import random
from pathlib import Path

import geopandas as gpd
import pandas as pd
import mesa
import mesa_geo as mg
import pyproj
from shapely.geometry import Point

from ..agent.building import Building
from ..agent.commuter import Commuter
from ..space.district import District
from ..space.road_network import DistrictWalkway


script_directory = Path(__file__).resolve().parent
repo_root = script_directory.parents[3]

logger = logging.getLogger(__name__)


def get_time(model) -> pd.Timedelta:
    return pd.Timedelta(days=model.day, hours=model.hour, minutes=model.minute)


def get_unix_time_ms(model) -> int:
    base = datetime(model.base_year, 1, 1, tzinfo=timezone.utc) + timedelta(
        days=model.day,
        hours=model.hour,
        minutes=model.minute,
    )
    return int(base.timestamp() * 1000)


class BikePedModel(mesa.Model):
    """Refactoring model: commuters choose between bike and car based on stress."""

    running: bool
    space: District
    car_network: DistrictWalkway
    bike_network: DistrictWalkway
    num_commuters: int
    day: int
    hour: int
    minute: int

    def __init__(
        self,
        *,
        district: str = "district",
        data_crs: str = "epsg:4326",
        buildings_file: str | Path = repo_root / "data/bld.zip",
        carways_file: str | Path = repo_root / "data/district_carway_line.zip",
        bikeways_file: str | Path = repo_root / "data/district_bikeway_line.zip",
        output_dir: str | Path = repo_root / "data/outputs",
        output_name: str = "agent_history",
        num_commuters: int = 50,
        commuter_bike_speed_m_per_tick: float = 300.0,
        commuter_car_speed_m_per_tick: float = 600.0,
        commuter_mode_choice_epsilon: float = 0.05,
        commuter_stress_ewma_alpha: float = 0.25,
        bike_sweet_spot: int = 5,
        overcrowding_weight: float = 0.0,
        bike_car_traffic_weight: float = 2.0,
        bike_social_benefit_weight: float = 5.0,
        bike_overcrowding_exponent: float = 1.5,
        car_congestion_threshold: int = 3,
        car_sweet_spot: int | None = None,
        car_overcrowding_weight: float = 0.3,
        car_bike_traffic_weight: float = 0.8,
        car_bike_crowding_threshold: int = 1,
        initial_car_share: float = 0.8,
        stress_bin_size_m: float = 25.0,
        crowding_bin_size_m: float | None = None,
        car_distance_threshold_m: float = 5000.0,
        car_prob_below_threshold: float = 0.1,
        car_prob_max: float = 0.9,
        car_prob_ramp_m: float = 5000.0,
        model_crs: str = "epsg:3857",
        output_crs: str | None = None,
        output_format: str = "csv",
        start_day: int = 0,
        start_hour: int = 5,
        start_minute: int = 55,
    ) -> None:
        super().__init__()

        logger.info("Initializing BikePedModel")

        self.district = district
        self.data_crs = data_crs
        if crowding_bin_size_m is not None:
            stress_bin_size_m = crowding_bin_size_m

        self.space = District(crs=model_crs, stress_bin_size_m=stress_bin_size_m)
        self.num_commuters = num_commuters
        self.initial_car_share = float(initial_car_share)
        if not (0.0 <= self.initial_car_share <= 1.0):
            raise ValueError("initial_car_share must be in [0,1]")

        # Absolute timestamp base for CSV output (current year).
        self.base_year = datetime.now().year

        self.model_crs = model_crs
        self.output_crs = output_crs or model_crs
        self.output_format = output_format
        self._xy_transformer: pyproj.Transformer | None = None
        if self.output_crs != self.model_crs:
            self._xy_transformer = pyproj.Transformer.from_crs(
                self.model_crs,
                self.output_crs,
                always_xy=True,
            )

        # GeoJSON is effectively lon/lat (EPSG:4326) for most tooling.
        self._geojson_transformer: pyproj.Transformer | None = None
        if self.output_format == "geojson":
            self._geojson_transformer = pyproj.Transformer.from_crs(
                self.model_crs,
                "epsg:4326",
                always_xy=True,
            )

        # Clock state (5-minute ticks)
        self.day = start_day
        self.hour = start_hour
        self.minute = start_minute

        # Configure commuter behavior
        Commuter.BIKE_SPEED_M_PER_TICK = commuter_bike_speed_m_per_tick
        Commuter.CAR_SPEED_M_PER_TICK = commuter_car_speed_m_per_tick
        Commuter.MODE_CHOICE_EPSILON = commuter_mode_choice_epsilon
        Commuter.STRESS_EWMA_ALPHA = commuter_stress_ewma_alpha
        Commuter.BIKE_SWEET_SPOT = int(bike_sweet_spot)
        Commuter.OVERCROWDING_WEIGHT = float(overcrowding_weight)
        Commuter.BIKE_CAR_TRAFFIC_WEIGHT = float(bike_car_traffic_weight)
        Commuter.BIKE_SOCIAL_BENEFIT_WEIGHT = float(bike_social_benefit_weight)
        Commuter.BIKE_OVERCROWDING_EXPONENT = float(bike_overcrowding_exponent)
        threshold = int(car_congestion_threshold)
        if car_sweet_spot is not None:
            threshold = int(car_sweet_spot)
        Commuter.CAR_CONGESTION_THRESHOLD = threshold
        Commuter.CAR_SWEET_SPOT = threshold
        Commuter.CAR_OVERCROWDING_WEIGHT = float(car_overcrowding_weight)
        Commuter.CAR_BIKE_TRAFFIC_WEIGHT = float(car_bike_traffic_weight)
        Commuter.CAR_BIKE_CROWDING_THRESHOLD = int(car_bike_crowding_threshold)

        Commuter.CAR_DISTANCE_THRESHOLD_M = float(car_distance_threshold_m)
        Commuter.CAR_PROB_BELOW_THRESHOLD = float(car_prob_below_threshold)
        Commuter.CAR_PROB_MAX = float(car_prob_max)
        Commuter.CAR_PROB_RAMP_M = float(car_prob_ramp_m)

        # Output paths
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_name = str(output_name).strip()
        if not self.output_name:
            raise ValueError("output_name must not be empty")
        self._agent_log_path = self.output_dir / f"{self.output_name}.csv"
        self._agent_geojson_path = self.output_dir / f"{self.output_name}.geojson"
        self._geojson_file = None
        self._geojson_first_feature = True

        logger.info(
            "Output dir=%s | model_crs=%s | output_crs=%s | output_format=%s",
            self.output_dir,
            self.model_crs,
            self.output_crs,
            self.output_format,
        )

        logger.info(
            "Stress bin size=%.2fm | initial_car_share=%.2f | bike_sweet_spot=%d | bike_social_benefit_weight=%.2f | bike_car_traffic_weight=%.2f | bike_overcrowding_weight=%.2f | bike_overcrowding_exponent=%.2f | car_congestion_threshold=%d | car_overcrowding_weight=%.2f | car_bike_traffic_weight=%.2f | car_bike_crowding_threshold=%d",
            getattr(self.space, "stress_bin_size_m", float("nan")),
            self.initial_car_share,
            Commuter.BIKE_SWEET_SPOT,
            Commuter.BIKE_SOCIAL_BENEFIT_WEIGHT,
            Commuter.BIKE_CAR_TRAFFIC_WEIGHT,
            Commuter.OVERCROWDING_WEIGHT,
            Commuter.BIKE_OVERCROWDING_EXPONENT,
            Commuter.CAR_CONGESTION_THRESHOLD,
            Commuter.CAR_OVERCROWDING_WEIGHT,
            Commuter.CAR_BIKE_TRAFFIC_WEIGHT,
            Commuter.CAR_BIKE_CROWDING_THRESHOLD,
        )

        logger.info(
            "Car distance rule: threshold_m=%.0f prob_below=%.2f prob_max=%.2f ramp_m=%.0f",
            Commuter.CAR_DISTANCE_THRESHOLD_M,
            Commuter.CAR_PROB_BELOW_THRESHOLD,
            Commuter.CAR_PROB_MAX,
            Commuter.CAR_PROB_RAMP_M,
        )

        # Load data and initialize model state
        logger.info(
            "Loading buildings=%s carways=%s bikeways=%s",
            buildings_file,
            carways_file,
            bikeways_file,
        )
        self._load_buildings_from_file(buildings_file, crs=model_crs)
        logger.info(
            "Buildings loaded: total=%d homes=%d workplaces=%d",
            len(getattr(self.space, "_buildings", {})),
            len(self.space.homes),
            len(self.space.workplaces),
        )
        self._load_networks_from_files(
            carways_file=carways_file,
            bikeways_file=bikeways_file,
            crs=model_crs,
        )

        logger.info(
            "Car network: nodes=%d edges=%d",
            self.car_network.nx_graph.number_of_nodes(),
            self.car_network.nx_graph.number_of_edges(),
        )
        logger.info(
            "Bike network: nodes=%d edges=%d",
            self.bike_network.nx_graph.number_of_nodes(),
            self.bike_network.nx_graph.number_of_edges(),
        )

        self._set_building_entrances()
        logger.info("Building entrances assigned")
        self._create_commuters()
        logger.info("Commuters created: %d", len(self.agents_by_type.get(Commuter, [])))

        if self.output_format == "csv":
            # Initialize log file with a header (overwrite on each run so runs
            # don't get mixed together).
            header = "step,commuter_id,unix_time_ms,day,hour,minute,status,mode,x,y,output_crs\n"
            self._agent_log_path.write_text(header)
        elif self.output_format == "geojson":
            # Always overwrite: we stream a single FeatureCollection.
            logger.info("Writing GeoJSON history to %s (EPSG:4326)", self._agent_geojson_path)
            self._agent_geojson_path.write_text(
                '{"type":"FeatureCollection","features":[\n',
                encoding="utf-8",
            )
            self._geojson_file = self._agent_geojson_path.open("a", encoding="utf-8")
            self._geojson_first_feature = True
        elif self.output_format == "daily_stats":
            # Write a compact daily summary for easier downstream analysis.
            header = "day,bike_share,car_share,bike_count,car_count,c2b,b2c,net_switch\n"
            self._agent_log_path.write_text(header)
            self._last_daily_written_day: int | None = None
            self._prev_daily_modes: dict[int, str] = {}
        else:
            raise ValueError(
                f"Invalid output_format={self.output_format!r}. "
                "Expected 'csv', 'geojson', or 'daily_stats'."
            )

        logger.info("Initialization complete")

    def finalize(self) -> None:
        """Finalize outputs (e.g., close GeoJSON FeatureCollection)."""
        if self.output_format == "geojson" and self._geojson_file is not None:
            self._geojson_file.write("\n]}\n")
            self._geojson_file.close()
            self._geojson_file = None
        
    def _create_commuters(self) -> None:
        for _ in range(self.num_commuters):
            random_home = self.space.get_random_home()
            random_workplace = self.space.get_random_workplace()
            commuter = Commuter(
                model=self,
                geometry=Point(random_home.centroid),
                crs=self.space.crs,
            )
            commuter.set_home(random_home)
            commuter.set_workplace(random_workplace)
            commuter.status = "home"

            # Seed initial mode mix.
            if random.random() < self.initial_car_share:
                commuter.current_mode = "car"
            else:
                commuter.current_mode = "bike"

            self.space.add_commuter(commuter)
    
    
    def _load_buildings_from_file(self, buildings_file: str | Path, crs: str) -> None:
        buildings_df = gpd.read_file(buildings_file)
        logger.info("Read buildings rows=%d crs=%s", len(buildings_df), buildings_df.crs)

        if "unique_id" in buildings_df.columns:
            buildings_df = buildings_df.set_index("unique_id", drop=True)
        elif "Id" in buildings_df.columns:
            buildings_df = buildings_df.set_index("Id", drop=True)

        buildings_df.index.name = "unique_id"
        if buildings_df.crs is None:
            buildings_df = buildings_df.set_crs(self.data_crs, allow_override=True)
        buildings_df = buildings_df.to_crs(crs)

        if "function" not in buildings_df.columns:
            logger.warning(
                "Buildings dataset has no 'function' column. Homes/workplaces split may be empty. columns=%s",
                list(buildings_df.columns),
            )

        # If we have a function column, we only need buildings labeled as
        # homes/workplaces (function 1/0) for this PoC. Filtering here avoids
        # instantiating thousands of unused Building agents.
        if "function" in buildings_df.columns:
            function_numeric = pd.to_numeric(buildings_df["function"], errors="coerce")
            function_mask = function_numeric.isin([0, 1])
            dropped = int((~function_mask).sum())
            buildings_df = buildings_df[function_mask].copy()
            buildings_df["function"] = function_numeric[function_mask].astype(float)
            logger.info(
                "Buildings filtered to homes/workplaces: kept=%d dropped=%d",
                len(buildings_df),
                dropped,
            )

        # Defensive cleaning: some datasets contain empty/invalid geometries that
        # produce NaN centroids and break nearest-node lookup.
        before = len(buildings_df)
        buildings_df = buildings_df[buildings_df.geometry.notna()]
        buildings_df = buildings_df[~buildings_df.geometry.is_empty]

        centroids = buildings_df.geometry.centroid
        buildings_df = buildings_df[centroids.notna()]
        buildings_df["centroid"] = list(zip(centroids.x, centroids.y))
        logger.info("Buildings cleaned: kept=%d dropped=%d", len(buildings_df), before - len(buildings_df))

        # Reduce per-agent attribute assignment overhead by keeping only columns
        # used by the Building agent.
        keep_cols = ["geometry", "centroid"]
        if "name" in buildings_df.columns:
            keep_cols.append("name")
        if "function" in buildings_df.columns:
            keep_cols.append("function")
        buildings_df = buildings_df[keep_cols]

        building_creator = mg.AgentCreator(Building, model=self)
        buildings = building_creator.from_GeoDataFrame(buildings_df)
        self.space.add_buildings(buildings)

    def _load_networks_from_files(
        self,
        *,
        carways_file: str | Path,
        bikeways_file: str | Path,
        crs: str,
    ) -> None:
        car_df = gpd.read_file(carways_file)
        if car_df.crs is None:
            car_df = car_df.set_crs(self.data_crs, allow_override=True)
        car_df = car_df.to_crs(crs)
        logger.info("Read carways rows=%d crs=%s", len(car_df), car_df.crs)

        bike_df = gpd.read_file(bikeways_file)
        if bike_df.crs is None:
            bike_df = bike_df.set_crs(self.data_crs, allow_override=True)
        bike_df = bike_df.to_crs(crs)
        logger.info("Read bikeways rows=%d crs=%s", len(bike_df), bike_df.crs)

        self.car_network = DistrictWalkway(
            district=f"{self.district}_car",
            lines=car_df["geometry"],
            output_dir=str(self.output_dir),
        )
        self.bike_network = DistrictWalkway(
            district=f"{self.district}_bike",
            lines=bike_df["geometry"],
            output_dir=str(self.output_dir),
        )
        # Backward-compatible alias for older call sites.
        self.walk_network = self.car_network
        
    def _set_building_entrances(self) -> None:
        for building in (*self.space.homes, *self.space.workplaces):
            building.car_entrance_pos = self.car_network.get_nearest_node(
                building.centroid
            )
            building.bike_entrance_pos = self.bike_network.get_nearest_node(
                building.centroid
            )
            # Backward-compatible aliases/defaults.
            building.walk_entrance_pos = building.car_entrance_pos
            building.entrance_pos = building.car_entrance_pos

    def step(self) -> None:
        step = getattr(self, "_step", 0)
        commuters = list(self.agents_by_type.get(Commuter, []))

        log_interval = getattr(self, "log_interval", 1)
        if step % log_interval == 0:
            status_counts: dict[str, int] = {}
            mode_counts: dict[str, int] = {}
            for c in commuters:
                status_counts[c.status] = status_counts.get(c.status, 0) + 1
                mode_counts[c.current_mode] = mode_counts.get(c.current_mode, 0) + 1

            total = len(commuters) or 1
            bike_share = mode_counts.get("bike", 0) / total
            logger.info(
                "Step=%d day=%d %02d:%02d | bike_share=%.1f%% (%d/%d) | statuses=%s",
                step,
                self.day,
                self.hour,
                self.minute,
                bike_share * 100,
                mode_counts.get("bike", 0),
                total,
                status_counts,
            )

        self._update_clock()
        self.agents_by_type[Commuter].shuffle_do("step")
        self._write_agent_snapshot()

    def _update_clock(self) -> None:
        self.minute += 5
        if self.minute == 60:
            if self.hour == 23:
                self.hour = 0
                self.day += 1
            else:
                self.hour += 1
            self.minute = 0

    def _write_agent_snapshot(self) -> None:
        """Write per-commuter state (CSV/GeoJSON) or compact daily stats."""
        unix_time_ms = get_unix_time_ms(self)

        # Mesa doesn't guarantee a stable step counter attribute across versions.
        # Track our own.
        step = getattr(self, "_step", 0)

        commuters = self.agents_by_type.get(Commuter, [])

        if self.output_format == "csv":
            lines: list[str] = []
            for commuter in commuters:
                x = float(commuter.geometry.x)
                y = float(commuter.geometry.y)
                if self._xy_transformer is not None:
                    x, y = self._xy_transformer.transform(x, y)
                lines.append(
                    f"{step},{commuter.unique_id},{unix_time_ms},{self.day},{self.hour},{self.minute},"
                    f"{commuter.status},{commuter.current_mode},{x},{y},{self.output_crs}\n"
                )

            if lines:
                with self._agent_log_path.open("a", encoding="utf-8") as f:
                    f.writelines(lines)

        elif self.output_format == "geojson":
            if self._geojson_file is None or self._geojson_transformer is None:
                raise RuntimeError("GeoJSON output is not initialized")

            for commuter in commuters:
                x = float(commuter.geometry.x)
                y = float(commuter.geometry.y)
                lon, lat = self._geojson_transformer.transform(x, y)

                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(lon), float(lat)],
                    },
                    "properties": {
                        "step": step,
                        "commuter_id": commuter.unique_id,
                        "unix_time_ms": unix_time_ms,
                        "day": self.day,
                        "hour": self.hour,
                        "minute": self.minute,
                        "status": commuter.status,
                        "mode": commuter.current_mode,
                    },
                }

                if not self._geojson_first_feature:
                    self._geojson_file.write(",\n")
                self._geojson_first_feature = False
                self._geojson_file.write(json.dumps(feature, ensure_ascii=False))

        elif self.output_format == "daily_stats":
            if self._last_daily_written_day is None or self.day != self._last_daily_written_day:
                bike_count = sum(1 for c in commuters if c.current_mode == "bike")
                total = len(commuters)
                car_count = total - bike_count
                bike_share = (bike_count / total) if total else 0.0
                car_share = (car_count / total) if total else 0.0

                current_modes = {int(c.unique_id): str(c.current_mode) for c in commuters}
                c2b = 0
                b2c = 0
                if self._prev_daily_modes:
                    for cid, mode in current_modes.items():
                        prev = self._prev_daily_modes.get(cid)
                        if prev == "car" and mode == "bike":
                            c2b += 1
                        elif prev == "bike" and mode == "car":
                            b2c += 1

                net_switch = c2b - b2c
                row = (
                    f"{self.day},{bike_share:.6f},{car_share:.6f},"
                    f"{bike_count},{car_count},{c2b},{b2c},{net_switch}\n"
                )
                with self._agent_log_path.open("a", encoding="utf-8") as f:
                    f.write(row)

                self._prev_daily_modes = current_modes
                self._last_daily_written_day = self.day

        else:
            raise ValueError(f"Invalid output_format={self.output_format!r}")

        setattr(self, "_step", step + 1)
        
    
    
    
 
 
 

    