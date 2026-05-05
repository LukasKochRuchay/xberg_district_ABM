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
    stress from co-presence. This version measures local co-location via
    District.get_commuters_by_pos and updates an internal expected stress for
    each mode.
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

    BIKE_SPEED_M_PER_TICK: float = 300.0
    CAR_SPEED_M_PER_TICK: float = 600.0
    #Learning parameters. Higher epsilon means more random mode choice instead of following learned stress expectations. Higher alpha means faster updating of
    MODE_CHOICE_EPSILON: float = 0.05
    STRESS_EWMA_ALPHA: float = 0.25

    # Generic learned mode-choice features (phase 1 keeps only stress active).
    FEATURES_BY_MODE: dict[str, tuple[str, ...]] = {
        "bike": ("stress",),
        "car": ("stress",),
    }
    # Raw feature values are combined with per-mode weights.
    FEATURE_WEIGHTS: dict[str, dict[str, float]] = {
        "bike": {"stress": -1.0},
        "car": {"stress": -1.0},
    }
    # Feature scales for normalization before weighted aggregation.
    FEATURE_NORMALIZATION_SCALE: dict[str, float] = {
        "stress": 1.0,
    }

    # Car choice: distance-based probability (in meters).
    CAR_DISTANCE_THRESHOLD_M: float = 5000.0
    CAR_PROB_BELOW_THRESHOLD: float = 0.1
    CAR_PROB_MAX: float = 0.9
    CAR_PROB_RAMP_M: float = 5000.0

    # When biking, car co-presence contributes to perceived stress. Bike co-presence is beneficial up to a point, then becomes stressful when overcrowding sets in.
    BIKE_CAR_TRAFFIC_WEIGHT: float = 2.0
    BIKE_SOCIAL_BENEFIT_WEIGHT: float = 5.0
    BIKE_SWEET_SPOT: int = 5
    OVERCROWDING_WEIGHT: float = 0.0
    BIKE_OVERCROWDING_EXPONENT: float = 1.5
    
    # When driving, stress starts only once car density exceeds this threshold.
    CAR_CONGESTION_THRESHOLD: int = 3
    CAR_OVERCROWDING_WEIGHT: float = 0.3
    CAR_BIKE_TRAFFIC_WEIGHT: float = 0.8
    CAR_BIKE_CROWDING_THRESHOLD: int = 1

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

        # Commuter behavior parameters are configured on the class by BikePedModel.
        # Keeping them off the instance avoids shadowing model-provided values.

        self._expected_features: dict[str, dict[str, float]] = {
            mode: {feature: 0.0 for feature in self.FEATURES_BY_MODE[mode]}
            for mode in self.FEATURES_BY_MODE
        }
        self._trip_feature_samples: dict[str, dict[str, list[float]]] = {
            mode: {feature: [] for feature in self.FEATURES_BY_MODE[mode]}
            for mode in self.FEATURES_BY_MODE
        }

        # Phase-1 starts unbiased: all expected features initialize to zero.

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
            # FIX #6: Move stress sampling to after the agent has moved to the next position
            self.model.space.move_commuter(self, next_position)
            self._sample_stress(next_position)
            self.step_in_path += 1
            return

        # Arrived.
        if self.destination is not None:
            self.model.space.move_commuter(self, self.destination.centroid)

        self._update_expected_features_from_trip()

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

        # Prefer the mode with higher weighted utility from normalized features.
        bike_u = self._mode_utility("bike")
        car_u = self._mode_utility("car")
        if np.isclose(bike_u, car_u):
            # Break ties without inertia so agents do not get sticky by default.
            self.current_mode = random.choice(["bike", "car"])
        else:
            self.current_mode = "bike" if bike_u > car_u else "car"

    # --- Path selection ---
    def _path_select(self) -> None:
        self.step_in_path = 0
        for mode in self._trip_feature_samples:
            for feature in self._trip_feature_samples[mode]:
                self._trip_feature_samples[mode][feature].clear()

        if self.origin is None or self.destination is None:
            self.my_path = []
            return

        # Network selection is deferred to the model.
        # Expected attributes (to be added in your refactor model later):
        # - model.car_network
        # - model.bike_network
        # - bike uses model.bike_network
        # - car uses model.car_network
        network = (
            getattr(self.model, "car_network", None)
            if self.current_mode == "car"
            else getattr(self.model, "bike_network", None)
        )
        if network is None:
            # Minimal fail-safe: no network configured yet.
            self.my_path = []
            return

        if self.current_mode == "car":
            source = getattr(
                self.origin,
                "car_entrance_pos",
                self.origin.entrance_pos,
            )
            target = getattr(
                self.destination,
                "car_entrance_pos",
                self.destination.entrance_pos,
            )
        else:
            source = getattr(self.origin, "bike_entrance_pos", self.origin.entrance_pos)
            target = getattr(self.destination, "bike_entrance_pos", self.destination.entrance_pos)

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

        speed = (
            self.CAR_SPEED_M_PER_TICK
            if self.current_mode == "car"
            else self.BIKE_SPEED_M_PER_TICK
        )

        unit_transformer = UnitTransformer(degree_crs=network_crs)
        original_path = LineString([Point(p) for p in self.my_path])
        path_in_meters = unit_transformer.degree2meter(original_path)
        redistributed_path_in_meters = redistribute_vertices(path_in_meters, speed)
        redistributed_path_in_degree = unit_transformer.meter2degree(
            redistributed_path_in_meters
        )
        self.my_path = list(redistributed_path_in_degree.coords)

    # --- Stress measurement & learning ---
    def _sample_stress(self, pos: mesa.space.FloatCoordinate) -> None:
        """Approximate stress via binned co-location at the next point.

        Bikes: strong social co-presence benefit, then overcrowding penalty after
        a sweet spot plus stress from nearby cars.

        Cars: no positive effect from other cars; stress only from congestion
        (cars beyond a threshold) and from crowded bike presence.
        """
        colocated = self.model.space.get_commuters_by_pos(pos)
        # BUG FIX: Exclude self from co-location count to avoid inflating social benefit
        colocated = [c for c in colocated if c != self]
        bikes = [c for c in colocated if getattr(c, "current_mode", None) == "bike"]
        cars = [c for c in colocated if getattr(c, "current_mode", None) == "car"]

        if len(bikes) <= self.BIKE_SWEET_SPOT:
            bike_term = -self.BIKE_SOCIAL_BENEFIT_WEIGHT * len(bikes)
        else:
            excess = len(bikes) - self.BIKE_SWEET_SPOT
            bike_term = (
                -self.BIKE_SOCIAL_BENEFIT_WEIGHT * self.BIKE_SWEET_SPOT
                + self.OVERCROWDING_WEIGHT * (excess ** self.BIKE_OVERCROWDING_EXPONENT)
            )

        bike_stress = bike_term + self.BIKE_CAR_TRAFFIC_WEIGHT * len(cars)
        car_excess = max(0, len(cars) - self.CAR_CONGESTION_THRESHOLD)
        bike_excess = max(0, len(bikes) - self.CAR_BIKE_CROWDING_THRESHOLD)
        # No positive effects for cars: stress is only congestion + bike pressure + base disutility.
        car_stress = (
            car_excess * self.CAR_OVERCROWDING_WEIGHT
            + bike_excess * self.CAR_BIKE_TRAFFIC_WEIGHT
        )

        # Learn mode-specific stress feature from local context.
        self._trip_feature_samples["bike"]["stress"].append(float(bike_stress))
        self._trip_feature_samples["car"]["stress"].append(float(car_stress))

    def _update_expected_features_from_trip(self) -> None:
        alpha = self.STRESS_EWMA_ALPHA

        for mode in self.FEATURES_BY_MODE:
            for feature in self.FEATURES_BY_MODE[mode]:
                samples = self._trip_feature_samples[mode][feature]
                if not samples:
                    continue
                trip_mean = float(np.mean(samples))
                prev = self._expected_features[mode][feature]
                self._expected_features[mode][feature] = (
                    (1 - alpha) * prev + alpha * trip_mean
                )

    def _mode_utility(self, mode: str) -> float:
        expected = self._expected_features.get(mode, {})
        weights = self.FEATURE_WEIGHTS.get(mode, {})
        utility = 0.0
        for feature in self.FEATURES_BY_MODE.get(mode, ()):
            value = float(expected.get(feature, 0.0))
            normalized = self._normalize_feature_value(feature, value)
            utility += float(weights.get(feature, 0.0)) * normalized
        return utility

    def _normalize_feature_value(self, feature: str, value: float) -> float:
        scale = float(self.FEATURE_NORMALIZATION_SCALE.get(feature, 1.0))
        denom = max(1e-9, abs(scale))
        return value / denom
    
    
