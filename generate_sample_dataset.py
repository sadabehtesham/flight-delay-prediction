import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(42)
    airlines = ["AA", "DL", "UA", "SW", "WN"]
    airports = ["JFK", "LAX", "ORD", "SFO", "ATL"]
    rows = []

    for i in range(500):
        airline = rng.choice(airlines)
        scheduled_departure = int(rng.integers(0, 2359))
        scheduled_arrival = (scheduled_departure + int(rng.integers(30, 240))) % 2400
        departure_delay = float(rng.integers(-20, 180))
        cancelled = int(rng.random() < 0.02)
        diverted = int(rng.random() < 0.01)

        rows.append(
            {
                "YEAR": 2023,
                "MONTH": int(rng.integers(1, 13)),
                "DAY": int(rng.integers(1, 29)),
                "DAY_OF_WEEK": int(rng.integers(1, 8)),
                "AIRLINE": airline,
                "FLIGHT_NUMBER": 1000 + i,
                "TAIL_NUMBER": f"TAIL{i % 100:03d}",
                "ORIGIN_AIRPORT": rng.choice(airports),
                "DESTINATION_AIRPORT": rng.choice(airports),
                "SCHEDULED_DEPARTURE": scheduled_departure,
                "DEPARTURE_TIME": scheduled_departure if rng.random() < 0.9 else np.nan,
                "DEPARTURE_DELAY": departure_delay if rng.random() < 0.9 else np.nan,
                "TAXI_OUT": int(rng.integers(5, 30)) if rng.random() < 0.9 else np.nan,
                "WHEELS_OFF": scheduled_departure + 10 if rng.random() < 0.9 else np.nan,
                "SCHEDULED_TIME": int(rng.integers(60, 240)),
                "ELAPSED_TIME": int(rng.integers(60, 240)) if rng.random() < 0.9 else np.nan,
                "AIR_TIME": int(rng.integers(60, 240)) if rng.random() < 0.9 else np.nan,
                "DISTANCE": int(rng.integers(200, 3000)),
                "WHEELS_ON": scheduled_arrival if rng.random() < 0.9 else np.nan,
                "TAXI_IN": int(rng.integers(2, 20)) if rng.random() < 0.9 else np.nan,
                "SCHEDULED_ARRIVAL": scheduled_arrival,
                "ARRIVAL_TIME": scheduled_arrival if rng.random() < 0.9 else np.nan,
                "ARRIVAL_DELAY": departure_delay if rng.random() < 0.9 else np.nan,
                "DIVERTED": diverted,
                "CANCELLED": cancelled,
                "CANCELLATION_REASON": "B" if cancelled else "",
                "AIR_SYSTEM_DELAY": int(rng.integers(0, 60)) if rng.random() < 0.9 else np.nan,
                "SECURITY_DELAY": int(rng.integers(0, 20)) if rng.random() < 0.9 else np.nan,
                "AIRLINE_DELAY": int(rng.integers(0, 90)) if rng.random() < 0.9 else np.nan,
                "LATE_AIRCRAFT_DELAY": int(rng.integers(0, 80)) if rng.random() < 0.9 else np.nan,
                "WEATHER_DELAY": int(rng.integers(0, 40)) if rng.random() < 0.9 else np.nan,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv("sample_flights.csv", index=False)
    print(f"Created sample_flights.csv with {len(df)} rows and {df.memory_usage(index=True).sum()} bytes.")


if __name__ == "__main__":
    main()
