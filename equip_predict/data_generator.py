"""Generador de datos sinteticos de sensores para equipos oil & gas."""

import numpy as np
import pandas as pd
from pathlib import Path


EQUIPMENT_TYPES = {
    "centrifugal_pump": {
        "vibration_range": (0.5, 8.0),
        "temp_range": (40, 95),
        "pressure_range": (50, 400),
        "flow_range": (50, 500),
        "current_range": (10, 80),
        "rpm_range": (1000, 3600),
        "typical_life_days": (180, 720),
    },
    "reciprocating_pump": {
        "vibration_range": (2.0, 15.0),
        "temp_range": (35, 85),
        "pressure_range": (100, 1000),
        "flow_range": (10, 200),
        "current_range": (15, 100),
        "rpm_range": (200, 600),
        "typical_life_days": (240, 900),
    },
    "compressor": {
        "vibration_range": (1.0, 12.0),
        "temp_range": (50, 150),
        "pressure_range": (200, 2000),
        "flow_range": (100, 2000),
        "current_range": (20, 200),
        "rpm_range": (1500, 6000),
        "typical_life_days": (365, 1460),
    },
    "separator": {
        "vibration_range": (0.3, 4.0),
        "temp_range": (30, 70),
        "pressure_range": (50, 300),
        "flow_range": (20, 300),
        "current_range": (5, 40),
        "rpm_range": (0, 0),
        "typical_life_days": (500, 1800),
    },
}

FAILURE_MODES = [
    "bearing_wear",
    "seal_failure",
    "impeller_erosion",
    "valve_malfunction",
    "motor_overload",
]

STATUS_LABELS = {
    0: "normal",
    1: "warning",
    2: "preventive_maintenance",
    3: "corrective_maintenance",
    4: "failure_imminent",
}


class EquipDataGenerator:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def generate(self, n_units=150, n_days=365):
        records = []
        for unit_id in range(n_units):
            unit_data = self._generate_unit(unit_id, n_days)
            records.append(unit_data)
        return pd.concat(records, ignore_index=True)

    def _generate_unit(self, unit_id, n_days):
        eq_type = self.rng.choice(list(EQUIPMENT_TYPES.keys()), p=[0.35, 0.25, 0.25, 0.15])
        spec = EQUIPMENT_TYPES[eq_type]
        failure_mode = self.rng.choice(FAILURE_MODES)
        total_life = int(self.rng.uniform(*spec["typical_life_days"]))
        failure_day = int(total_life * self.rng.uniform(0.7, 1.0))

        age_factor = 0
        records = []

        for day in range(1, min(n_days, total_life) + 1):
            progress = day / total_life
            age_factor = progress ** 1.5

            vib_base = self.rng.uniform(*spec["vibration_range"])
            vib = vib_base * (1 + age_factor * 2.5) + self.rng.normal(0, vib_base * 0.1)
            vib_x = max(vib * self.rng.uniform(0.8, 1.2), 0)
            vib_y = max(vib * self.rng.uniform(0.6, 1.0), 0)
            vib_z = max(vib * self.rng.uniform(0.5, 0.9), 0)

            temp_base = self.rng.uniform(*spec["temp_range"])
            temp = temp_base + age_factor * 30 + self.rng.normal(0, 3)
            temp = max(temp, 15)

            press_base = self.rng.uniform(*spec["pressure_range"])
            pressure = press_base * (1 - age_factor * 0.3) + self.rng.normal(0, press_base * 0.05)
            pressure = max(pressure, 10)

            flow_base = self.rng.uniform(*spec["flow_range"])
            flow = flow_base * (1 - age_factor * 0.2) + self.rng.normal(0, flow_base * 0.03)
            flow = max(flow, 0)

            current_base = self.rng.uniform(*spec["current_range"])
            current = current_base * (1 + age_factor * 0.5) + self.rng.normal(0, current_base * 0.05)
            current = max(current, 0)

            rpm = self.rng.uniform(*spec["rpm_range"]) if spec["rpm_range"][1] > 0 else 0
            bearing_temp = temp + self.rng.uniform(5, 25) * (1 + age_factor)
            oil_level = max(100 - age_factor * 40 + self.rng.normal(0, 5), 0)
            seal_pressure = pressure * self.rng.uniform(0.1, 0.3) * (1 + age_factor * 0.5)
            suction_pressure = pressure * self.rng.uniform(0.3, 0.6)
            discharge_pressure = pressure * self.rng.uniform(0.8, 1.2)
            power = current * 0.48 * self.rng.uniform(0.9, 1.1) * (1 + age_factor * 0.3)

            status = self._determine_status(age_factor, day, failure_day, vib, temp, current, spec)
            rul = max(failure_day - day, 0)
            days_since_maint = int(day % self.rng.integers(30, 90))

            records.append({
                "unit_id": f"EQ-{unit_id:04d}",
                "equipment_type": eq_type,
                "failure_mode": failure_mode,
                "day": day,
                "vibration_x": round(vib_x, 3),
                "vibration_y": round(vib_y, 3),
                "vibration_z": round(vib_z, 3),
                "temperature": round(temp, 1),
                "pressure": round(pressure, 1),
                "flow_rate": round(flow, 1),
                "motor_current": round(current, 2),
                "rpm": round(rpm, 0),
                "bearing_temp": round(bearing_temp, 1),
                "oil_level": round(oil_level, 1),
                "seal_pressure": round(seal_pressure, 1),
                "suction_pressure": round(suction_pressure, 1),
                "discharge_pressure": round(discharge_pressure, 1),
                "power_consumption": round(power, 2),
                "runtime_hours": round(day * self.rng.uniform(18, 24), 1),
                "days_since_maintenance": days_since_maint,
                "status": status,
                "status_label": STATUS_LABELS[status],
                "rul_days": rul,
                "total_life_days": total_life,
                "failure_day": failure_day,
            })

        return pd.DataFrame(records)

    def _determine_status(self, age_factor, day, failure_day, vib, temp, current, spec):
        if age_factor > 0.9:
            return 4
        if age_factor > 0.75:
            return 3
        if age_factor > 0.5:
            return 2
        if age_factor > 0.3:
            return 1
        return 0

    def save(self, df, path="data/equipment_sensors.csv"):
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        return filepath


if __name__ == "__main__":
    gen = EquipDataGenerator(seed=42)
    df = gen.generate(n_units=150, n_days=365)
    path = gen.save(df)
    print(f"Dataset: {len(df)} registros, {df['unit_id'].nunique()} equipos")
    print(f"Distribucion de status:")
    for label, count in df["status_label"].value_counts().items():
        print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")
