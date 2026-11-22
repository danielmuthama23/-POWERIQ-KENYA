---
title: POWERIQ data dictionary and model definitions
category: documentation
source: POWERIQ project documentation
---
# Data dictionary

All data in the prototype is synthetic and labelled data_origin = synthetic. History covers 270 days of hourly values ending 2026-11-18 23:00 (East Africa Time). Forecasts are labelled forecast.

Demand: regional input demand in MW for ten illustrative distribution regions. Generation: plant output in MW for geothermal, hydro, wind, solar, thermal and an import link. Maintenance label: a unit fault within the next 14 days. Anomaly label: injected fault episodes in hourly telemetry. Loss label: feeders with injected non-technical loss. Held-out windows are the last 28 days for forecasting, the last 22 percent of days for maintenance, the last 40 percent of hours for anomalies, and the last 3 months for losses.
