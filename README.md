# HTM-Monitor

Detects system-level anomalies in streaming data by requiring sustained agreement across groups of related signals in complex systems such as infrastructure and industrial processes.

**Two anomaly types:**
- **Regime shifts** — sustained structural changes in system behavior
- **System shocks** — short, high-coherence disruptions across signals

## Method
- one HTM model per signal
- related signals grouped
- system anomaly requires **group agreement + temporal persistence**

## Why this works
- filters transient noise by requiring temporal persistence
- captures system structure via grouped signal agreement
- converts local prediction breakdowns into stable system-level events

## Use Cases

### Power Grid Outage Detection (2020-22)

#### Overview
Monitoring core grid signals:
- demand
- net generation
- imbalance (difference between generation and demand)

Goal:
- detect known outage events in hourly streaming signals

Result:
- both labeled outages (Aug 2020, Sept 2022) are detected
- a regime shift is detected during spring 2020 (COVID onset)

Conclusion:
- the system reliably detects outages
- and surfaces broader structural changes and transient system-level disruptions

---

##### Outage -- Aug 2020
![Outage -- Aug 2020](assets/powergrid_ca_aug2020.gif)

---
##### Outage -- Sept 2022
![Outage -- Sept 2022](assets/powergrid_ca_sept2022.gif)

---

### SWaT Water Treatment Facilities Failure Detection (yy-yy)

#### Overview
(in progress)

Goal:
(to be defined) 

Result:
(to be defined)  

Conclusion:
(to be defined) 

##### -- Attack Example

---
##### -- Attack Example

---

## Getting Started

See `docs/quickstart.md` for usage instructions.
