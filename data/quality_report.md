# EnergyWise Data-Quality Report

- **Source:** `data\raw\combined_raw.parquet` (read-only)
- **Rows:** 8,301,386 across 4 building(s)
- **Buildings:** academic, facilities, lecture, library
- **Generated (UTC):** 2026-09-28 05:52:19 | **Asia/Kolkata (IST):** 2026-09-28 11:22:19 +0530
- **Units / conventions:** `power` in watts; timestamps stored in UTC, **all times below are converted to Asia/Kolkata (IST, UTC+05:30)**.
- **Plausible mains range:** voltage 184-276 V, frequency 45.0-55.0 Hz (nominal 230 V / 50 Hz).
- **Missing-interval rule:** any gap > 60 s (60 s expected cadence).

## Building: `academic`

### 1. Rows and date range

- **Total rows:** 2,269,530
- **Date range (IST):** 2013-08-10 00:00:00 +0530  ->  2017-12-31 23:59:00 +0530

### 2. Duplicate timestamps

- **Duplicate rows (extra beyond first):** 0
- **Distinct timestamps that are duplicated:** 0
- **Duplicate rows identical across all values?** n/a (no duplicates)

### 3. Missing intervals (gaps > 1 minute)

- **Number of gaps:** 489
- **Total missing minutes:** 41,670

**10 largest gaps** (start = last reading before the gap, IST):

| Gap start (IST) | Missing minutes | Span (min) |
| --- | --- | --- |
| 2014-05-20 22:34:00 +0530 | 15,283 | 15,284.0 |
| 2017-08-05 12:42:00 +0530 | 5,508 | 5,509.0 |
| 2014-02-21 12:02:00 +0530 | 4,528 | 4,529.0 |
| 2014-02-24 16:33:00 +0530 | 4,168 | 4,169.0 |
| 2014-05-31 13:18:00 +0530 | 2,888 | 2,889.0 |
| 2017-04-29 13:33:00 +0530 | 1,373 | 1,374.0 |
| 2015-01-14 14:01:00 +0530 | 1,194 | 1,195.0 |
| 2017-07-07 23:59:00 +0530 | 741 | 742.0 |
| 2017-09-12 15:31:00 +0530 | 729 | 730.0 |
| 2013-09-29 14:21:00 +0530 | 643 | 644.0 |

### 4. Impossible values

| Check | Count |
| --- | --- |
| Negative power (< 0 W) | 0 |
| Voltage below 184 V | 60 |
| Voltage above 276 V | 0 |
| Frequency below 45.0 Hz | 59 |
| Frequency above 55.0 Hz | 0 |

- **`power_factor` sign note:** 312,849 of 2,269,528 non-null values (13.78%) are **negative**. This is interpreted as a leading/lagging sign convention and is **not** counted as an error.

### 5. NA fraction per numeric column

| Column | NA fraction | NA count |
| --- | --- | --- |
| `timestamp` | 0.0000% | 0 |
| `power` | 0.0001% | 3 |
| `current` | 9.2891% | 210,818 |
| `voltage` | 0.1690% | 3,836 |
| `frequency` | 0.0001% | 3 |
| `power_factor` | 0.0001% | 2 |

## Building: `facilities`

### 1. Rows and date range

- **Total rows:** 2,048,910
- **Date range (IST):** 2013-11-15 00:00:00 +0530  ->  2017-12-31 23:59:00 +0530

### 2. Duplicate timestamps

- **Duplicate rows (extra beyond first):** 0
- **Distinct timestamps that are duplicated:** 0
- **Duplicate rows identical across all values?** n/a (no duplicates)

### 3. Missing intervals (gaps > 1 minute)

- **Number of gaps:** 3,191
- **Total missing minutes:** 122,610

**10 largest gaps** (start = last reading before the gap, IST):

| Gap start (IST) | Missing minutes | Span (min) |
| --- | --- | --- |
| 2015-09-23 13:20:00 +0530 | 63,289 | 63,290.0 |
| 2014-05-15 00:25:00 +0530 | 6,673 | 6,674.0 |
| 2013-11-22 16:38:00 +0530 | 4,198 | 4,199.0 |
| 2014-04-18 18:18:00 +0530 | 4,025 | 4,026.0 |
| 2014-05-20 22:34:00 +0530 | 3,914 | 3,915.0 |
| 2015-05-18 06:20:00 +0530 | 3,237 | 3,238.0 |
| 2014-04-29 14:16:00 +0530 | 2,909 | 2,910.0 |
| 2017-10-01 14:12:00 +0530 | 2,615 | 2,616.0 |
| 2014-05-08 02:12:00 +0530 | 2,274 | 2,275.0 |
| 2015-01-31 08:10:00 +0530 | 1,727 | 1,728.0 |

### 4. Impossible values

| Check | Count |
| --- | --- |
| Negative power (< 0 W) | 0 |
| Voltage below 184 V | 54 |
| Voltage above 276 V | 0 |
| Frequency below 45.0 Hz | 54 |
| Frequency above 55.0 Hz | 1 |

- **`power_factor` sign note:** 64,273 of 2,048,901 non-null values (3.14%) are **negative**. This is interpreted as a leading/lagging sign convention and is **not** counted as an error.

### 5. NA fraction per numeric column

| Column | NA fraction | NA count |
| --- | --- | --- |
| `timestamp` | 0.0000% | 0 |
| `power` | 0.0003% | 7 |
| `current` | 3.2395% | 66,375 |
| `voltage` | 0.2140% | 4,384 |
| `frequency` | 0.0007% | 14 |
| `power_factor` | 0.0004% | 9 |

## Building: `lecture`

### 1. Rows and date range

- **Total rows:** 2,269,309
- **Date range (IST):** 2013-08-10 00:00:00 +0530  ->  2017-12-31 23:59:00 +0530

### 2. Duplicate timestamps

- **Duplicate rows (extra beyond first):** 0
- **Distinct timestamps that are duplicated:** 0
- **Duplicate rows identical across all values?** n/a (no duplicates)

### 3. Missing intervals (gaps > 1 minute)

- **Number of gaps:** 719
- **Total missing minutes:** 41,891

**10 largest gaps** (start = last reading before the gap, IST):

| Gap start (IST) | Missing minutes | Span (min) |
| --- | --- | --- |
| 2014-05-20 22:34:00 +0530 | 15,283 | 15,284.0 |
| 2017-08-05 12:42:00 +0530 | 5,508 | 5,509.0 |
| 2014-02-21 12:02:00 +0530 | 4,528 | 4,529.0 |
| 2014-02-24 16:33:00 +0530 | 4,168 | 4,169.0 |
| 2014-05-31 13:18:00 +0530 | 2,888 | 2,889.0 |
| 2017-04-29 13:33:00 +0530 | 1,373 | 1,374.0 |
| 2015-01-14 14:01:00 +0530 | 1,194 | 1,195.0 |
| 2017-07-07 23:59:00 +0530 | 741 | 742.0 |
| 2017-09-12 15:31:00 +0530 | 729 | 730.0 |
| 2013-09-29 14:21:00 +0530 | 643 | 644.0 |

### 4. Impossible values

| Check | Count |
| --- | --- |
| Negative power (< 0 W) | 0 |
| Voltage below 184 V | 6 |
| Voltage above 276 V | 0 |
| Frequency below 45.0 Hz | 20 |
| Frequency above 55.0 Hz | 7 |

- **`power_factor` sign note:** 276 of 2,269,308 non-null values (0.01%) are **negative**. This is interpreted as a leading/lagging sign convention and is **not** counted as an error.

### 5. NA fraction per numeric column

| Column | NA fraction | NA count |
| --- | --- | --- |
| `timestamp` | 0.0000% | 0 |
| `power` | 0.0000% | 0 |
| `current` | 9.2863% | 210,734 |
| `voltage` | 89.1167% | 2,022,334 |
| `frequency` | 0.0000% | 1 |
| `power_factor` | 0.0000% | 1 |

## Building: `library`

### 1. Rows and date range

- **Total rows:** 1,713,637
- **Date range (IST):** 2013-08-10 00:00:00 +0530  ->  2017-12-31 23:59:00 +0530

### 2. Duplicate timestamps

- **Duplicate rows (extra beyond first):** 0
- **Distinct timestamps that are duplicated:** 0
- **Duplicate rows identical across all values?** n/a (no duplicates)

### 3. Missing intervals (gaps > 1 minute)

- **Number of gaps:** 639
- **Total missing minutes:** 597,563

**10 largest gaps** (start = last reading before the gap, IST):

| Gap start (IST) | Missing minutes | Span (min) |
| --- | --- | --- |
| 2015-02-05 18:34:00 +0530 | 293,445 | 293,446.0 |
| 2014-07-14 14:27:00 +0530 | 137,322 | 137,323.0 |
| 2014-11-13 17:26:00 +0530 | 89,125 | 89,126.0 |
| 2017-01-28 15:58:00 +0530 | 18,292 | 18,293.0 |
| 2017-02-12 02:25:00 +0530 | 12,495 | 12,496.0 |
| 2017-06-30 16:19:00 +0530 | 11,702 | 11,703.0 |
| 2017-08-05 12:41:00 +0530 | 5,516 | 5,517.0 |
| 2017-05-13 16:47:00 +0530 | 4,053 | 4,054.0 |
| 2017-07-16 11:09:00 +0530 | 2,817 | 2,818.0 |
| 2017-07-22 14:45:00 +0530 | 2,414 | 2,415.0 |

### 4. Impossible values

| Check | Count |
| --- | --- |
| Negative power (< 0 W) | 0 |
| Voltage below 184 V | 20 |
| Voltage above 276 V | 0 |
| Frequency below 45.0 Hz | 19 |
| Frequency above 55.0 Hz | 3 |

- **`power_factor` sign note:** 995,337 of 1,713,637 non-null values (58.08%) are **negative**. This is interpreted as a leading/lagging sign convention and is **not** counted as an error.

### 5. NA fraction per numeric column

| Column | NA fraction | NA count |
| --- | --- | --- |
| `timestamp` | 0.0000% | 0 |
| `power` | 0.0000% | 0 |
| `current` | 12.3405% | 211,471 |
| `voltage` | 0.2551% | 4,371 |
| `frequency` | 0.0000% | 0 |
| `power_factor` | 0.0000% | 0 |

## `power_factor` sign convention (summary)

A total of **1,372,735** rows have a negative `power_factor` across all buildings. Per the dataset this reflects the leading/lagging sign convention and is reported for information only -- it is **not** a data error.

## Cross-check: present-minute count vs. `data_present_status_buildings.csv`

- **Reference file:** `data\raw\energy_dataset\data_present_status_buildings.csv`
- **Method:** rows in the raw layer for each building vs. the count of minutes with status = 1 in the reference CSV.

| Building | Raw rows | Present minutes | Delta | Status |
| --- | --- | --- | --- | --- |
| academic | 2,269,530 | 2,269,527 | +3 | **MISMATCH** |
| facilities | 2,048,910 | 2,048,903 | +7 | **MISMATCH** |
| lecture | 2,269,309 | 2,269,309 | +0 | OK |
| library | 1,713,637 | 1,713,637 | +0 | OK |

> **Note:** one or more buildings show a mismatch between the raw row count and the present-status reference. Differences indicate readings that exist in the raw data but are marked absent (or vice-versa) in the status file.
