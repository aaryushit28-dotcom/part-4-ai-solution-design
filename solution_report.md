# AI Solution Design Report
## Healthcare – AI-Powered Medical Image Triage System

---

## Task 1: Business Domain — Healthcare

**Selected Domain:** Healthcare  
**Reference:** `ai_usecase_reference_catalog.csv` — Row: Healthcare / Medical image triage

Healthcare was selected because:
- Medical imaging is one of the fastest-growing AI application areas globally
- The problem has clear, measurable business impact (patient outcomes + operational efficiency)
- Image classification is a well-studied deep learning task with proven architectures
- The dataset type (X-rays, CT scans) is structured enough to build a reliable model

---

## Task 2: Define the Business Problem

### What problem is being solved?
Hospitals receive hundreds of radiology scans daily — chest X-rays, CT scans, MRIs.
Each image must be reviewed by a radiologist before clinical action is taken.
**The bottleneck:** radiologists manually review every scan in the order it arrives,
regardless of clinical urgency. A critical pneumonia case may wait behind a routine
check-up scan simply because it arrived later.

### Who are the users and stakeholders?
| Stakeholder | Role |
|-------------|------|
| Radiologists | Primary users — receive AI-prioritised worklist |
| Emergency physicians | Rely on fast radiology reports for critical patients |
| Hospital administrators | Measure throughput, cost, and liability |
| Patients | End beneficiaries — faster diagnosis, better outcomes |
| Insurance providers | Interested in reduced hospitalisation duration and cost |

### Current Manual Process
1. Scan is captured by a technician and uploaded to the PACS (Picture Archiving System)
2. A radiologist opens scans in FIFO (first-in, first-out) order
3. They visually examine each image, dictate a report, and mark it for physician review
4. A physician acts on the report — often hours after the scan was taken
5. Average resolution time: **~28–35 hours** (from KPI baseline data)

### Limitations of the Current Process
- **No automatic prioritisation** — critical cases are not fast-tracked
- **Human fatigue** — radiologists reviewing hundreds of scans per shift miss subtle findings
- **High error rate** — KPI baseline shows **4–11% error rate** in monthly reports
- **Low throughput** — 500+ manual processing hours per month for ~2,700 cases
- **Poor satisfaction** — customer satisfaction score baseline is only **6.4–7.6 / 10**
- **Scalability** — headcount cannot scale as fast as imaging volume grows

---

## Task 3: AI Task Type

**Selected AI Task Type: Image Classification**

### Why Image Classification?
The input is a medical image (chest X-ray or CT scan) and the output is one of a set
of predefined categories — this is precisely what image classification solves.

| Label | Description |
|-------|-------------|
| `URGENT` | Findings that require immediate clinical attention (e.g. pneumothorax, aortic dissection) |
| `PRIORITY` | Significant abnormality present, radiologist should review within 2 hours |
| `ROUTINE` | No acute abnormality; standard review queue |

The model assigns one of these three labels to each incoming scan.
This directly replaces the FIFO queue with an **AI-prioritised worklist**.

### Why not other task types?
- **Object detection** would be used if we needed to localise findings on the image
  (a useful next-phase upgrade, but not required for triage prioritisation)
- **Regression** would be used if predicting a continuous value (e.g. a severity score),
  but classification into urgency tiers is more actionable for clinical workflow
- **Sequence prediction** is not applicable as each scan is processed independently

---

## Task 4: Data Requirement Plan

### Type of Data Needed
| Data Type | Examples | Format |
|-----------|----------|--------|
| Medical images | Chest X-rays, CT scans | DICOM (.dcm) or JPEG/PNG |
| Radiologist labels | Urgency classification per image | CSV annotation file |
| Patient metadata | Age, gender, referring department | Structured CSV |
| Scan metadata | Modality, body part, scan date/time | DICOM header / CSV |

### Structured vs Unstructured
- **Unstructured:** The raw images (primary model input)
- **Structured:** Patient metadata and scan attributes (used as auxiliary features)

### Input Features
- Raw pixel values of the medical image (resized to 224×224 for ResNet)
- Patient age (numerical)
- Referring department (categorical — encoded)
- Modality type (X-ray, CT — categorical)

### Target Variable / Labels
- `urgency_label` — multiclass: `URGENT`, `PRIORITY`, `ROUTINE`
- Labels sourced from radiologist annotations on historical scans

### Data Collection Method
1. **Historical data:** De-identified scans from hospital PACS with radiologist reports
   (retrospectively labelled by senior radiologists into urgency tiers)
2. **Ongoing data:** Real-time labelling — radiologists confirm or correct AI predictions,
   which feeds back as training data (active learning loop)
3. **Public datasets:** NIH ChestX-ray14 dataset (112,000+ labelled X-rays) for pre-training

### Data Quality Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| Class imbalance (few URGENT cases) | Model biased towards ROUTINE | Oversample URGENT class; use weighted loss |
| Inconsistent labelling across radiologists | Noisy labels | Multi-radiologist consensus labels |
| Image quality variation (scanner type) | Poor generalisation | Normalise and augment during training |
| Missing patient metadata | Incomplete feature set | Impute with population medians; make metadata optional |
| DICOM privacy (patient identifiers) | Legal / compliance risk | Full de-identification pipeline before storage |

---

## Task 5: Model Recommendation

### Recommended Model: CNN with Transfer Learning (ResNet-50)

**Architecture:** ResNet-50 pre-trained on ImageNet, fine-tuned on chest X-ray data

```
Input Image (224 × 224 × 3)
        │
  [ResNet-50 Backbone]   ← Pre-trained on ImageNet (frozen initially)
        │
  [Global Average Pooling]
        │
  [Dense 256, ReLU]
        │
  [Dropout 0.4]
        │
  [Dense 3, Softmax]     ← Output: URGENT / PRIORITY / ROUTINE
```

### Why ResNet-50 with Transfer Learning?

| Reason | Explanation |
|--------|-------------|
| **Proven architecture** | ResNet-50 won ImageNet 2015; its residual connections prevent vanishing gradients in deep networks |
| **Transfer learning** | Medical imaging datasets are small; pre-trained ImageNet weights give a strong starting point without needing millions of labelled scans |
| **Computational efficiency** | ResNet-50 is significantly lighter than ResNet-152 or VGG-19 while maintaining comparable accuracy |
| **Fine-tuning flexibility** | Freeze early layers (edge/texture detectors), fine-tune deeper layers (high-level feature detectors) |
| **Clinical validation** | Variants of ResNet have been validated in published radiology AI papers (CheXNet, etc.) |

### Why not other architectures?
- **Simple CNN from scratch** — requires huge labelled dataset; would underfit on hospital-scale data
- **VGG-16** — higher parameter count with no accuracy benefit for this task
- **Transformer (ViT)** — powerful but requires even more data and compute; better as Phase 2 upgrade
- **RNN/LSTM** — designed for sequences, not 2D spatial image data

### Training Strategy
1. **Phase 1:** Freeze ResNet backbone, train only the classification head (5 epochs)
2. **Phase 2:** Unfreeze last 2 ResNet blocks, fine-tune end-to-end (15 epochs, low LR)
3. **Augmentation:** Random flip, rotation ±10°, brightness jitter (simulate scanner variation)
4. **Loss:** Weighted categorical cross-entropy (higher weight on URGENT class)
5. **Optimiser:** Adam, learning rate 1e-4

---

## Task 6: Evaluation Plan

### Technical Metrics
| Metric | Target | Why |
|--------|--------|-----|
| Overall Accuracy | > 90% | General performance indicator |
| Recall (URGENT class) | > 95% | Missing an urgent case is clinically dangerous |
| Precision (URGENT class) | > 80% | Excessive false alarms waste radiologist time |
| Macro F1-Score | > 88% | Balances performance across all three classes |
| AUC-ROC | > 0.95 | Measures discrimination ability across thresholds |

> **Clinical priority:** Recall on URGENT cases is the most critical metric.
> A false negative (missed URGENT) is far more dangerous than a false positive.

### Business Metrics
Derived from `business_kpi_sample.csv` baseline:

| KPI | Baseline (avg) | Target with AI |
|-----|----------------|----------------|
| Average resolution time | 28–35 hours | < 12 hours |
| Manual processing hours/month | ~450 hours | < 180 hours |
| Error rate | 4–11% | < 3% |
| Customer satisfaction score | 6.4–7.6 / 10 | > 8.5 / 10 |
| Monthly cases handled | ~2,800 | > 4,000 (same headcount) |

### Possible Failure Cases
- **Distribution shift:** Model trained on one scanner brand fails on a different hospital's equipment
- **Rare disease miss:** Pathologies not present in training data (e.g. novel disease) scored as ROUTINE
- **Adversarial quality:** Very low quality or corrupted images producing confident wrong predictions
- **Edge cases:** Paediatric scans with different anatomy than the adult training distribution

### Human Review and Validation Process
1. **Shadow mode deployment (Month 1–2):** Model runs alongside radiologists but outputs are not shown — compare predictions to radiologist labels silently
2. **Assisted mode (Month 3–4):** Model outputs shown to radiologists as a second opinion; they can override
3. **Active mode (Month 5+):** AI-prioritised worklist is live; radiologists review AI flags and provide feedback
4. **Ongoing:** Monthly model performance audit by clinical AI governance committee

---

## Task 7: Responsible AI Considerations

### 1. Bias in Data
**Risk:** If the training dataset is predominantly from one demographic (e.g. adult males),
the model may perform poorly on women, elderly patients, or different ethnicities.  
**Mitigation:** Audit dataset for demographic balance; stratify train/test splits by age and gender;
measure model performance separately per subgroup before deployment.

### 2. Incorrect Predictions (False Negatives)
**Risk:** A critical pneumothorax labelled ROUTINE by the model is placed at the back of the queue.
The patient deteriorates before a radiologist reviews it.  
**Mitigation:** Set recall threshold for URGENT class at 95%+; deploy with a confidence threshold —
scans below 70% confidence are automatically escalated to senior radiologist regardless of predicted class.

### 3. Privacy and Data Protection
**Risk:** Medical images contain biometric data and are protected under HIPAA / DPDP Act.
A data breach could expose patient identity and health history.  
**Mitigation:** Full de-identification of all DICOM files before training; encrypted storage;
access logs for every model inference; data processed on-premise or in a certified healthcare cloud.

### 4. Over-reliance on AI
**Risk:** Radiologists begin to "rubber stamp" AI predictions without independent judgement,
leading to automation bias — errors made by AI are no longer caught.  
**Mitigation:** Mandatory training for all radiologists on AI limitations; require radiologists
to document their own finding before viewing AI output in certain workflows;
regular blind audits where AI output is hidden.

### 5. Impact on Users (Radiologists)
**Risk:** Staff perceive AI as a threat to their jobs and resist adoption, or conversely
feel deskilled over time.  
**Mitigation:** Frame AI as a "triage assistant" not a replacement; involve radiologists in
model design and validation; maintain reporting that shows radiologist contributions,
not just AI performance.

### 6. Lack of Explainability
**Risk:** A black-box model that says "URGENT" without showing why is hard for a clinician
to trust or override intelligently.  
**Mitigation:** Integrate Grad-CAM visualisation — generate a heatmap overlay showing which
region of the image drove the prediction. Displayed alongside AI triage label in the UI.

### 7. Need for Human Oversight
**Risk:** AI system makes autonomous decisions without any human in the loop,
creating medico-legal liability.  
**Mitigation:** All AI decisions are flagged as "AI-assisted, radiologist confirmed" in the
final report. No clinical action is taken based solely on the AI output.
The AI only controls queue order, not diagnosis.

---

## Task 8: Final Solution Summary (One-Page Overview)

---

### Problem
Hospitals process 2,500–3,000 radiology scans per month using a manual FIFO review system.
Critical cases are not automatically prioritised, leading to average resolution times of
28–35 hours, error rates of 4–11%, and patient satisfaction scores below 7/10.

### Proposed AI Solution
Deploy an **AI-Powered Medical Image Triage System** using a fine-tuned ResNet-50 CNN.
The model analyses each incoming chest X-ray or CT scan and classifies it as
`URGENT`, `PRIORITY`, or `ROUTINE` — replacing the FIFO queue with an AI-ranked worklist.
Radiologists review cases in priority order rather than arrival order.

### Required Data
- De-identified chest X-rays and CT scans (DICOM format)
- Radiologist urgency labels (retrospectively annotated)
- Patient metadata: age, gender, referring department
- NIH ChestX-ray14 public dataset for transfer learning pre-training

### Model Recommendation
**ResNet-50 + Transfer Learning**  
Pre-trained on ImageNet → fine-tuned on labelled hospital scans.  
Weighted cross-entropy loss to prioritise URGENT class recall.  
Grad-CAM heatmaps for explainability.

### Expected Business Impact
| Metric | Before AI | After AI (Target) |
|--------|-----------|-------------------|
| Avg resolution time | 28–35 hrs | < 12 hrs |
| Manual hours/month | ~450 hrs | < 180 hrs |
| Error rate | 4–11% | < 3% |
| Satisfaction score | 6.4–7.6 | > 8.5 |
| Cases handled/month | ~2,800 | > 4,000 |

**Estimated ROI:** 60% reduction in triage processing time; 40% faster escalation for critical cases;
potential to serve 40% more patients with the same radiology team.

### Risks and Mitigation Plan
| Risk | Mitigation |
|------|------------|
| Missed critical case (false negative) | 95%+ recall target; confidence threshold escalation |
| Demographic bias | Subgroup performance audits before deployment |
| Patient data privacy | Full DICOM de-identification; encrypted on-premise storage |
| Automation bias | Radiologist training; blind audit programme |
| Explainability | Grad-CAM heatmap overlays displayed in UI |
| Regulatory compliance | All outputs labelled "AI-assisted"; human sign-off required |

### Deployment Roadmap
| Phase | Timeline | Activity |
|-------|----------|----------|
| 1 – Shadow Mode | Month 1–2 | Model runs silently, outputs compared to radiologist labels |
| 2 – Assisted Mode | Month 3–4 | AI suggestions shown as second opinion; radiologist overrides logged |
| 3 – Active Triage | Month 5+ | Live AI-prioritised worklist; ongoing monitoring and retraining |

---
*Report prepared for Part 4 – AI Solution Design Assessment*  
*Reference data: ai_usecase_reference_catalog.csv, business_kpi_sample.csv*
