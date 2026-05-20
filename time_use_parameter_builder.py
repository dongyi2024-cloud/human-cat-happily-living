"""
时间利用参数生成器

核心边界：
- 活动时长来自本地版本化时间利用数据 CSV。
- 住宅区域映射是模型假设，必须随输出单独记录。
"""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from project_paths import resolve_project_path


ACTIVITY_TO_ZONE_MAP_VERSION = "ACTIVITY_TO_ZONE_MAP_v1"

ACTIVITY_TO_ZONE_MAP = {
    "sleep": {"human_sleep": 1.0},
    "home_work": {"human_work": 0.85, "shared": 0.15},
    "outside": {"outside": 1.0},
    "household": {"shared": 0.70, "human_sleep": 0.15, "cat_feeding": 0.15},
    "care": {"shared": 0.50, "human_sleep": 0.25, "cat_feeding": 0.25},
    "leisure": {"shared": 0.65, "window": 0.20, "human_sleep": 0.15},
    "education": {"human_work": 0.75, "shared": 0.25},
}

MODEL_ASSUMPTIONS = [
    "Activity durations are derived from authoritative time-use data files.",
    "Activity order is model-defined for this version.",
    "Activity-to-zone mapping is model-defined because time-use surveys do not provide room-level residential location data.",
    "Outside is modeled as a logical state; no door or commute path is modeled.",
    "Pet care time is not directly modeled unless independent source data are provided.",
]


class TimeUseParameterBuilder:
    def __init__(
        self,
        data_dir: str = "data",
        mapping_path: str = "config/human_profile_mapping.json",
        tick_minutes: float = 1.0,
        total_ticks: int = 1440,
    ):
        self.data_dir = resolve_project_path(data_dir)
        self.mapping_path = resolve_project_path(mapping_path)
        self.tick_minutes = float(tick_minutes)
        self.total_ticks = int(total_ticks)
        self.profile_mapping = self._load_mapping(self.mapping_path)
        self.source_tables: dict[str, list[dict[str, str]]] = {}
        self.last_profile: dict[str, Any] | None = None
        self.last_source_metadata: dict[str, Any] | None = None

    def list_supported_profiles(self) -> list[str]:
        return [
            profile_id
            for profile_id, mapping in self.profile_mapping.items()
            if mapping.get("profile_status") == "data_supported"
        ]

    def build_profile(self, profile_id: str = "default_china", country: str | None = None, day_type: str = "average_day") -> dict:
        if profile_id not in self.profile_mapping:
            raise ValueError(f"未知人类画像: {profile_id}")

        mapping = deepcopy(self.profile_mapping[profile_id])
        if country and mapping.get("country") != country:
            raise ValueError(f"画像 {profile_id} 属于 {mapping.get('country')}，不能用 country={country}")

        self._validate_mapping(profile_id, mapping)
        self.load_source_tables(mapping["raw_files"])
        raw_budget = self.extract_raw_time_budget(mapping)
        activity_budget, normalization = self.derive_activity_budget(raw_budget)
        zone_budget = self.map_activity_to_zone_budget(activity_budget)
        activity_schedule = self.build_activity_schedule(activity_budget)

        profile = {
            "profile_id": profile_id,
            "display_name": mapping["display_name"],
            "source": {
                "dataset": mapping["source_dataset"],
                "tables": mapping["source_tables"],
                "raw_files": mapping["raw_files"],
                "occupation_categories": mapping["occupation_categories"],
                "day_type": day_type,
            },
            "derived_activity_budget": activity_budget,
            "zone_budget": zone_budget,
            "activity_schedule": activity_schedule,
            "assumptions": {
                "activity_to_zone_map": ACTIVITY_TO_ZONE_MAP_VERSION,
                "activity_to_zone_weights": ACTIVITY_TO_ZONE_MAP,
                "activity_order_template": "sleep -> household/care -> work/outside/education -> household/care -> leisure -> sleep/rest",
                "tick_minutes": self.tick_minutes,
                "normalization_applied": normalization["normalization_applied"],
                "normalization_scale": normalization["scale"],
                "model_assumptions": MODEL_ASSUMPTIONS,
            },
            "profile_status": mapping["profile_status"],
            "paper_ready": bool(mapping.get("paper_ready", False)),
        }
        self.last_profile = profile
        self.last_source_metadata = self.build_source_metadata(mapping, raw_budget, normalization)
        return profile

    def load_source_tables(self, raw_files: list[str]) -> dict[str, list[dict[str, str]]]:
        loaded = {}
        for raw_file in raw_files:
            path = Path(self.data_dir) / raw_file
            if not path.exists():
                raise FileNotFoundError(f"缺少时间利用数据文件: {path}")
            with path.open(newline="", encoding="utf-8") as f:
                loaded[raw_file] = list(csv.DictReader(f))
        self.source_tables.update(loaded)
        return loaded

    def extract_raw_time_budget(self, mapping: dict) -> dict[str, Any]:
        dataset = mapping["source_dataset"]
        if dataset == "ATUS_2024":
            return self._extract_atus_budget(mapping)
        if dataset == "China_Time_Use_2018":
            return self._extract_china_budget(mapping)
        raise ValueError(f"不支持的数据集: {dataset}")

    def derive_activity_budget(self, raw_time_budget: dict[str, Any]) -> tuple[dict[str, int], dict[str, Any]]:
        raw_hours = raw_time_budget["activity_hours"]
        raw_minutes = {key: float(hours) * 60.0 for key, hours in raw_hours.items()}
        target_minutes = self.total_ticks * self.tick_minutes
        total_minutes = sum(raw_minutes.values())
        if total_minutes <= 0:
            raise ValueError("活动时间总和必须大于 0")

        scale = target_minutes / total_minutes
        normalization_applied = abs(total_minutes - target_minutes) > 0.5
        scaled_ticks_float = {
            key: (minutes * scale) / self.tick_minutes
            for key, minutes in raw_minutes.items()
        }
        activity_budget = self._round_ticks_to_total(scaled_ticks_float, self.total_ticks)
        normalization = {
            "normalization_applied": normalization_applied,
            "scale": scale,
            "raw_total_minutes": total_minutes,
            "target_minutes": target_minutes,
        }
        return activity_budget, normalization

    def map_activity_to_zone_budget(self, activity_budget: dict[str, int]) -> dict[str, int]:
        zone_float: dict[str, float] = {}
        for activity, ticks in activity_budget.items():
            weights = ACTIVITY_TO_ZONE_MAP.get(activity, {})
            for zone, weight in weights.items():
                zone_float[zone] = zone_float.get(zone, 0.0) + ticks * float(weight)
        return self._round_ticks_to_total(zone_float, sum(activity_budget.values()))

    def build_activity_schedule(self, activity_budget: dict[str, int]) -> list[dict[str, Any]]:
        ordered_parts = [
            ("sleep", 0.62),
            ("household", 0.45),
            ("care", 0.45),
            ("outside", 1.0),
            ("home_work", 1.0),
            ("education", 1.0),
            ("household", 0.55),
            ("care", 0.55),
            ("leisure", 1.0),
            ("sleep", 0.38),
        ]
        consumed = {key: 0 for key in activity_budget}
        schedule = []
        for activity, share in ordered_parts:
            total = activity_budget.get(activity, 0)
            if total <= 0:
                continue
            duration = int(round(total * share))
            if share >= 1.0:
                duration = total - consumed.get(activity, 0)
            else:
                duration = min(duration, total - consumed.get(activity, 0))
            if duration <= 0:
                continue
            consumed[activity] = consumed.get(activity, 0) + duration
            schedule.append({"activity": activity, "duration": duration})

        for activity, total in activity_budget.items():
            remaining = total - consumed.get(activity, 0)
            if remaining > 0:
                schedule.append({"activity": activity, "duration": remaining})
        return schedule

    def build_source_metadata(self, mapping: dict, raw_budget: dict[str, Any], normalization: dict[str, Any]) -> dict[str, Any]:
        return {
            "dataset": mapping["source_dataset"],
            "tables": mapping["source_tables"],
            "raw_files": mapping["raw_files"],
            "occupation_categories": mapping["occupation_categories"],
            "profile_status": mapping["profile_status"],
            "paper_ready": bool(mapping.get("paper_ready", False)),
            "raw_values": raw_budget,
            "derived_formulas": raw_budget["formulas"],
            "normalization": normalization,
            "model_assumptions": MODEL_ASSUMPTIONS,
            "activity_to_zone_map_version": ACTIVITY_TO_ZONE_MAP_VERSION,
            "source_urls": raw_budget["source_urls"],
        }

    def get_activity_to_zone_mapping_metadata(self) -> dict[str, Any]:
        return {
            "version": ACTIVITY_TO_ZONE_MAP_VERSION,
            "type": "model_assumption",
            "description": "Maps source-derived activity categories to residential zones; not observed room-level survey data.",
            "mapping": ACTIVITY_TO_ZONE_MAP,
            "model_assumptions": MODEL_ASSUMPTIONS,
        }

    def _extract_atus_budget(self, mapping: dict) -> dict[str, Any]:
        categories = mapping["occupation_categories"]
        table5 = self._rows_by_category("atus_2024_table5.csv", categories)
        table7 = self._rows_by_category("atus_2024_table7.csv", categories)
        baseline = self._single_row("atus_2024_table8b.csv", mapping["baseline_category"])

        work_hours = self._weighted_average(table5, "work_hours_avg_day", "total_employed_thousands")
        home_work_share = self._weighted_average(table7, "home_work_share", "total_employed_thousands")
        travel_hours = 0.0
        activity_hours = {
            "sleep": float(baseline["sleep_hours"]),
            "home_work": work_hours * home_work_share,
            "outside": work_hours * (1.0 - home_work_share) + travel_hours,
            "household": float(baseline["household_hours"]),
            "care": float(baseline["care_hours"]),
            "leisure": float(baseline["leisure_hours"]),
            "education": float(baseline["education_hours"]),
        }
        return {
            "dataset": "ATUS_2024",
            "activity_hours": activity_hours,
            "raw_fields": {
                "work_hours": work_hours,
                "home_work_share": home_work_share,
                "travel_hours": travel_hours,
                "baseline_sleep_hours": float(baseline["sleep_hours"]),
                "baseline_household_hours": float(baseline["household_hours"]),
                "baseline_care_hours": float(baseline["care_hours"]),
                "baseline_leisure_hours": float(baseline["leisure_hours"]),
                "baseline_education_hours": float(baseline["education_hours"]),
            },
            "formulas": {
                "work_minutes": "work_hours * 60",
                "home_work_minutes": "work_hours * home_work_share * 60",
                "outside_minutes": "work_hours * (1 - home_work_share) * 60 + travel_hours * 60",
                "sleep_minutes": "sleep_hours * 60",
                "household_minutes": "household_hours * 60",
                "care_minutes": "care_hours * 60",
                "leisure_minutes": "leisure_hours * 60",
                "education_minutes": "education_hours * 60",
            },
            "source_urls": sorted({
                *(row["source_url"] for row in table5),
                *(row["source_url"] for row in table7),
                baseline["source_url"],
            }),
        }

    def _extract_china_budget(self, mapping: dict) -> dict[str, Any]:
        row = self._single_row("china_time_use_2018.csv", mapping["baseline_category"])
        activity_hours = {
            "sleep": float(row["personal_care_hours"]),
            "home_work": float(row["paid_work_hours"]),
            "outside": float(row["travel_hours"]),
            "household": float(row["unpaid_work_hours"]),
            "care": 0.0,
            "leisure": float(row["discretionary_hours"]),
            "education": float(row["education_hours"]),
        }
        return {
            "dataset": "China_Time_Use_2018",
            "activity_hours": activity_hours,
            "raw_fields": {
                "personal_care_hours": float(row["personal_care_hours"]),
                "paid_work_hours": float(row["paid_work_hours"]),
                "unpaid_work_hours": float(row["unpaid_work_hours"]),
                "discretionary_hours": float(row["discretionary_hours"]),
                "education_hours": float(row["education_hours"]),
                "travel_hours": float(row["travel_hours"]),
            },
            "formulas": {
                "sleep_minutes": "personal_care_hours * 60; source category also includes hygiene and eating",
                "home_work_minutes": "paid_work_hours * 60",
                "outside_minutes": "travel_hours * 60",
                "household_minutes": "unpaid_work_hours * 60",
                "leisure_minutes": "discretionary_hours * 60",
                "education_minutes": "education_hours * 60",
            },
            "source_urls": [row["source_url"]],
        }

    def _validate_mapping(self, profile_id: str, mapping: dict) -> None:
        required = ["display_name", "source_dataset", "source_tables", "raw_files", "occupation_categories", "profile_status"]
        missing = [key for key in required if key not in mapping]
        if missing:
            raise ValueError(f"画像 {profile_id} 缺少映射字段: {missing}")
        if mapping["profile_status"] != "data_supported" and mapping.get("paper_ready"):
            raise ValueError(f"画像 {profile_id} 缺少数据支撑，不能标记为 paper_ready")
        for raw_file in mapping["raw_files"]:
            if not (Path(self.data_dir) / raw_file).exists():
                raise FileNotFoundError(f"画像 {profile_id} 缺少源数据文件: {raw_file}")

    def _load_mapping(self, mapping_path: str | Path) -> dict:
        mapping_path = Path(mapping_path)
        if not mapping_path.exists():
            raise FileNotFoundError(f"缺少人类画像映射文件: {mapping_path}")
        with mapping_path.open(encoding="utf-8") as f:
            return json.load(f)

    def _rows_by_category(self, raw_file: str, categories: list[str]) -> list[dict[str, str]]:
        rows = self.source_tables.get(raw_file)
        if rows is None:
            self.load_source_tables([raw_file])
            rows = self.source_tables[raw_file]
        selected = [row for row in rows if row.get("category") in categories]
        if len(selected) != len(categories):
            found = {row.get("category") for row in selected}
            missing = [category for category in categories if category not in found]
            raise ValueError(f"{raw_file} 缺少分类: {missing}")
        return selected

    def _single_row(self, raw_file: str, category: str) -> dict[str, str]:
        rows = self._rows_by_category(raw_file, [category])
        return rows[0]

    def _weighted_average(self, rows: list[dict[str, str]], value_key: str, weight_key: str) -> float:
        total_weight = sum(float(row[weight_key]) for row in rows)
        if total_weight <= 0:
            raise ValueError(f"权重字段 {weight_key} 总和必须大于 0")
        return sum(float(row[value_key]) * float(row[weight_key]) for row in rows) / total_weight

    def _round_ticks_to_total(self, values: dict[str, float], total: int) -> dict[str, int]:
        floors = {key: int(value) for key, value in values.items()}
        remainder = int(total - sum(floors.values()))
        fractions = sorted(
            ((key, values[key] - floors[key]) for key in values),
            key=lambda item: item[1],
            reverse=True,
        )
        for index in range(max(0, remainder)):
            key = fractions[index % len(fractions)][0]
            floors[key] += 1
        return floors


def build_default_human_profile(total_ticks: int = 1440, tick_minutes: float = 1.0) -> dict:
    builder = TimeUseParameterBuilder(total_ticks=total_ticks, tick_minutes=tick_minutes)
    return builder.build_profile("default_china", country="CN")
