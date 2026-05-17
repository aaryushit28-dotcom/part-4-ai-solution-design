# Part 4 – AI Solution Design for a Business Problem

## Domain: Healthcare
## Problem: AI-Powered Medical Image Triage System

---

## Overview
This project designs an end-to-end AI solution for automated medical image triage in hospitals.
A CNN-based deep learning model analyses chest X-rays and CT scans to flag urgent cases,
helping radiologists prioritise their review queue and reduce diagnostic delays.

## Repository Structure
```
part-4-ai-solution-design/
├── README.md
├── solution_report.md          ← Full 8-task analysis report
└── diagrams/
    └── solution_architecture.png   ← System architecture diagram
```

## Solution at a Glance
| Attribute | Detail |
|-----------|--------|
| Domain | Healthcare |
| Problem | Manual X-ray triage is slow; urgent cases can be missed |
| AI Task | Image Classification |
| Model | CNN + Transfer Learning (ResNet-50) |
| Expected Impact | 60% reduction in triage time, 40% faster critical case escalation |

## Reference Data Used
- `ai_usecase_reference_catalog.csv` — domain and model selection reference
- `business_kpi_sample.csv` — baseline KPI benchmarks for measuring business impact

## How to Navigate
Start with `solution_report.md` for the complete solution design.
The architecture diagram in `diagrams/` shows the end-to-end data and model pipeline.
