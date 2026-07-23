#!/usr/bin/env python3
"""Script de prediccion de mantenimiento predictivo."""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from equip_predict.data_generator import EquipDataGenerator, STATUS_LABELS
from equip_predict.utils.preprocessor import EquipPreprocessor
from equip_predict.models.failure_predictor import FailurePredictor
from equip_predict.models.life_estimator import LifeEstimator


def main():
    print("=" * 60)
    print("  PREDICTION - Equipment Predictive Maintenance")
    print("=" * 60)

    gen = EquipDataGenerator(seed=99)
    df = gen.generate(n_units=10, n_days=365)

    last_records = df.groupby("unit_id").tail(1)
    print(f"  Units to predict: {last_records['unit_id'].nunique()}")

    cls_model = FailurePredictor.load("outputs/models/failure_predictor.pkl")
    rul_model = LifeEstimator.load("outputs/models/life_estimator.pkl")
    prep = EquipPreprocessor()
    prep.fit(df)

    predictions = []
    for _, row in last_records.iterrows():
        unit_df = df[df["unit_id"] == row["unit_id"]]
        X = prep.transform(unit_df)
        cls_pred = cls_model.predict(X.iloc[[-1]])[0]
        rul_pred = rul_model.predict(X.iloc[[-1]])[0]

        entry = {
            "unit_id": row["unit_id"],
            "equipment_type": row["equipment_type"],
            "current_status": STATUS_LABELS[int(cls_pred)],
            "predicted_rul_days": round(float(rul_pred), 0),
            "actual_rul_days": int(row["rul_days"]),
            "vibration_total": round(float(np.sqrt(row["vibration_x"]**2 + row["vibration_y"]**2 + row["vibration_z"]**2)), 3),
            "temperature": round(float(row["temperature"]), 1),
            "power_consumption": round(float(row["power_consumption"]), 2),
        }
        predictions.append(entry)
        print(f"  {row['unit_id']}: status={entry['current_status']:<25} "
              f"RUL={entry['predicted_rul_days']:.0f}d (actual={entry['actual_rul_days']}d) "
              f"| {row['equipment_type']}")

    output_path = Path("outputs/predictions.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"\n  Predictions saved to {output_path}")
    print("  Prediction completed successfully!")
    return predictions


if __name__ == "__main__":
    main()
