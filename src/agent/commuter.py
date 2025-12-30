from __future__ import annotations

import random

import mesa
import mesa_geo as mg
import numpy as np
import pyproj
from shapely.geometry import LineString, Point

from ..space.utils import UnitTransformer, redistribute_vertices
from .building import Building


class Commuter(mg.GeoAgent):
    """A commuter who chooses between biking and car trip-by-trip.

    The key new mechanism for the refactor is mode choice based on experienced
    crowding (co-presence). This version measures crowding using exact-position
    co-location (via District.get_commuters_by_pos), and updates an internal
    expected crowding for each mode.
    """

    unique_id: int
    model: mesa.Model
    geometry: Point
    crs: pyproj.CRS
    home: Building | None
    workplace: Building | None
    origin: Building | None
    destination: Building | None
    status: str  # "home", "work", "transport"
    current_mode: str  # "bike" or "car"
    my_path: list[mesa.space.FloatCoordinate]
    step_in_path: int
    start_time_h: int
    start_time_m: int
    end_time_h: int
    end_time_m: int

    # --- Model parameters (can be overridden by the model) ---
    # NOTE: naming kept for minimal diff with the earlier PoC:
    # - WALK_SPEED_M_PER_TICK is used as BIKE speed
    # - BIKE_SPEED_M_PER_TICK is used as CAR speed
    WALK_SPEED_M_PER_TICK: float
    BIKE_SPEED_M_PER_TICK: float
    MODE_CHOICE_EPSILON: float
    CROWDING_EWMA_ALPHA: float

    # Car choice: distance-based probability (in meters).
    CAR_DISTANCE_THRESHOLD_M: float
    CAR_PROB_BELOW_THRESHOLD: float
    CAR_PROB_MAX: float
    CAR_PROB_RAMP_M: float

    # When biking, car co-presence contributes to perceived crowding.
    BIKE_CAR_TRAFFIC_WEIGHT: float

    def __init__(self, model, geometry, crs) -> None:
        super().__init__(model, geometry, crs)

        self.home = None
        self.workplace = None
        self.origin = None
        self.destination = None

        self.status = "home"
        self.current_mode = "bike"

        # Start/end time setup kept similar to the original model (5-minute ticks).
        self.start_time_h = round(np.random.normal(6.5, 1))
        while self.start_time_h < 6 or self.start_time_h > 9:
            self.start_time_h = round(np.random.normal(6.5, 1))
        self.start_time_m = int(np.random.randint(0, 12) * 5)
        self.end_time_h = int(self.start_time_h + 8)
        self.end_time_m = int(self.start_time_m)

        self.my_path = []
        self.step_in_path = 0

        # Default parameters (override these from the model if desired).
        # Bike/car speeds (see NOTE above).
        self.WALK_SPEED_M_PER_TICK = 300.0
        self.BIKE_SPEED_M_PER_TICK = 600.0
        self.MODE_CHOICE_EPSILON = 0.05
        self.CROWDING_EWMA_ALPHA = 0.25

        self.CAR_DISTANCE_THRESHOLD_M = 5000.0
        self.CAR_PROB_BELOW_THRESHOLD = 0.1
        self.CAR_PROB_MAX = 0.9
        self.CAR_PROB_RAMP_M = 5000.0  # from 5km to 10km reaches max

        self.BIKE_CAR_TRAFFIC_WEIGHT = 2.0

        # Learned expectations: lower is better (less crowded).
        self._expected_crowding: dict[str, float] = {"bike": 0.0, "car": 0.0}
        self._trip_crowding_samples: list[int] = []

    def __repr__(self) -> str:
        return (
            f"Commuter(unique_id={self.unique_id}, geometry={self.geometry}, "
            f"status={self.status}, current_mode={self.current_mode})"
        )

    def __eq__(self, other) -> bool:
        return isinstance(other, Commuter) and self.unique_id == other.unique_id

    def __hash__(self) -> int:
        return hash(self.unique_id)

    def set_home(self, home: Building) -> None:
        self.home = home

    def set_workplace(self, workplace: Building) -> None:
        self.workplace = workplace

    def step(self) -> None:
        self._prepare_to_move()
        self._move()

    def advance(self) -> None:
        raise NotImplementedError

    # --- Trip lifecycle ---
    def _prepare_to_move(self) -> None:
        # Decide before departure, trip-by-trip.
        if (
            self.status == "home"
            and self.model.hour == self.start_time_h
            and self.model.minute == self.start_time_m
        ):
            if self.home is None or self.workplace is None:
                return
            self.origin = self.model.space.get_building_by_id(self.home.unique_id)
            self.destination = self.model.space.get_building_by_id(
                self.workplace.unique_id
            )
            self._choose_mode_for_next_trip()
            self.model.space.move_commuter(self, pos=self.origin.centroid)
            self._path_select()
            self.status = "transport"

        elif (
            self.status == "work"
            and self.model.hour == self.end_time_h
            and self.model.minute == self.end_time_m
        ):
            if self.home is None or self.workplace is None:
                return
            self.origin = self.model.space.get_building_by_id(
                self.workplace.unique_id
            )
            self.destination = self.model.space.get_building_by_id(self.home.unique_id)
            self._choose_mode_for_next_trip()
            self.model.space.move_commuter(self, pos=self.origin.centroid)
            self._path_select()
            self.status = "transport"

    def _move(self) -> None:
        if self.status != "transport":
            return

        if self.step_in_path < len(self.my_path):
            next_position = self.my_path[self.step_in_path]
            self._sample_crowding(next_position)
            self.model.space.move_commuter(self, next_position)
            self.step_in_path += 1
            return

        # Arrived.
        if self.destination is not None:
            self.model.space.move_commuter(self, self.destination.centroid)

        self._update_expected_crowding_from_trip()

        if self.destination == self.workplace:
            self.status = "work"
        elif self.destination == self.home:
            self.status = "home"

    # --- Mode choice ---
    def _choose_mode_for_next_trip(self) -> None:
        # Epsilon-greedy: sometimes explore.
        if np.random.uniform(0.0, 1.0) < self.MODE_CHOICE_EPSILON:
            self.current_mode = random.choice(["bike", "car"])
            return

        # Distance-driven car adoption: car becomes more likely beyond a threshold.
        dist_m = 0.0
        if self.origin is not None and self.destination is not None:
            dx = float(self.origin.centroid[0] - self.destination.centroid[0])
            dy = float(self.origin.centroid[1] - self.destination.centroid[1])
            dist_m = float((dx * dx + dy * dy) ** 0.5)

        p_car = self.CAR_PROB_BELOW_THRESHOLD
        if dist_m > self.CAR_DISTANCE_THRESHOLD_M:
            ramp = max(0.0, dist_m - self.CAR_DISTANCE_THRESHOLD_M)
            denom = max(1.0, float(self.CAR_PROB_RAMP_M))
            p_car = min(self.CAR_PROB_MAX, self.CAR_PROB_BELOW_THRESHOLD + (self.CAR_PROB_MAX - self.CAR_PROB_BELOW_THRESHOLD) * (ramp / denom))

        if np.random.uniform(0.0, 1.0) < p_car:
            self.current_mode = "car"
            return

        # Prefer the mode with lower expected crowding.
        bike_c = self._expected_crowding["bike"]
        car_c = self._expected_crowding["car"]
        if bike_c == car_c:
            # Avoid a systematic bias.
            # Keep the current mode with slightly higher probability.
            if np.random.uniform(0.0, 1.0) < 0.6:
                return
            self.current_mode = random.choice(["bike", "car"])
        else:
            self.current_mode = "bike" if bike_c < car_c else "car"

    # --- Path selection ---
    def _path_select(self) -> None:
        self.step_in_path = 0
        self._trip_crowding_samples = []

        if self.origin is None or self.destination is None:
            self.my_path = []
            return

        # Network selection is deferred to the model.
        # Expected attributes (to be added in your refactor model later):
        # - model.walk_network
        # - model.bike_network
        # PoC: reuse existing two networks.
        # - bike uses model.walk_network (former pedestrian)
        # - car uses model.bike_network (former bikeway)
        network = (
            getattr(self.model, "bike_network", None)
            if self.current_mode == "car"
            else getattr(self.model, "walk_network", None)
        )
        if network is None:
            # Minimal fail-safe: no network configured yet.
            self.my_path = []
            return

        if self.current_mode == "car":
            source = getattr(self.origin, "bike_entrance_pos", self.origin.entrance_pos)
            target = getattr(self.destination, "bike_entrance_pos", self.destination.entrance_pos)
        else:
            source = getattr(self.origin, "walk_entrance_pos", self.origin.entrance_pos)
            target = getattr(self.destination, "walk_entrance_pos", self.destination.entrance_pos)

        self.my_path = network.get_shortest_path(
            source=source,
            target=target,
        )
        self._redistribute_path_vertices(network_crs=getattr(network, "crs", None))

    def _redistribute_path_vertices(self, network_crs) -> None:
        if len(self.my_path) <= 1:
            return

        if network_crs is None:
            return

        # Choose speed based on current mode.
        # Bike uses WALK_SPEED_M_PER_TICK; car uses BIKE_SPEED_M_PER_TICK.
        speed = self.BIKE_SPEED_M_PER_TICK if self.current_mode == "car" else self.WALK_SPEED_M_PER_TICK

        unit_transformer = UnitTransformer(degree_crs=network_crs)
        original_path = LineString([Point(p) for p in self.my_path])
        path_in_meters = unit_transformer.degree2meter(original_path)
        redistributed_path_in_meters = redistribute_vertices(path_in_meters, speed)
        redistributed_path_in_degree = unit_transformer.meter2degree(
            redistributed_path_in_meters
        )
        self.my_path = list(redistributed_path_in_degree.coords)

    # --- Crowding measurement & learning ---
    def _sample_crowding(self, pos: mesa.space.FloatCoordinate) -> None:
        """Approximate crowding via binned co-location at the next point.

        For PoC we also let bike users perceive car co-presence (traffic).
        """
        colocated = self.model.space.get_commuters_by_pos(pos)
        bikes = [c for c in colocated if getattr(c, "current_mode", None) == "bike"]
        cars = [c for c in colocated if getattr(c, "current_mode", None) == "car"]

        if self.current_mode == "bike":
            perceived = len(bikes) + self.BIKE_CAR_TRAFFIC_WEIGHT * len(cars)
        else:
            perceived = len(cars)

        self._trip_crowding_samples.append(max(0, int(perceived)))

    def _update_expected_crowding_from_trip(self) -> None:
        if not self._trip_crowding_samples:
            return

        trip_mean = float(np.mean(self._trip_crowding_samples))
        prev = self._expected_crowding[self.current_mode]
        alpha = self.CROWDING_EWMA_ALPHA
        self._expected_crowding[self.current_mode] = (1 - alpha) * prev + alpha * trip_mean
    
    
