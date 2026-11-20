# Team plan: decisions and checklist

## Decisions implemented in this code (confirm as a team)
| Open decision (briefing section 5) | Choice in this codebase | Change it by |
|---|---|---|
| Static image vs video + LSTM | **Static, no LSTM** (matches written Scope) | would need new collection + model |
| Number of classes | 12 (in `config.GESTURES`) - briefing suggests 15-20 | add rows to `GESTURES` |
| Public dataset for pipeline testing | Yes, optional (`tools/import_public.py`) | skip it |
| Poster deadline | unknown - decides how much stays "projected" | - |

The 12 gestures are simple, distinct hand shapes mapped to service phrases. If the guide wants
*sign language* signs rather than gestures, replace the vocabulary with real ISL signs.

## Collection checklist (tick per signer)
Each cell = one `collect_data.py` run covering all classes (~25-40 images/class/session).

| Signer | normal+plain | normal+cluttered | dim+plain | dim+cluttered | backlit+plain | outdoor |
|---|---|---|---|---|---|---|
| signer01 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| signer02 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| signer03 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| signer04 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| volunteer 1-3 (optional) | | | | | | |

Shoot short bursts per condition (not one long session) so a label never correlates with one
lighting. Vary hand orientation and distance during each burst.

## Phase checklist (from the briefing)
- [ ] 1 Vocabulary fixed in `config.py`
- [ ] 2 Variation matrix collected
- [ ] 3 Dataset on shared Drive with `metadata.csv`; code only in GitHub
- [ ] 4 `prepare_dataset.py` run; split is by signer; counts reviewed
- [ ] 5-6 `train.py` run; train/val gap checked
- [ ] 7 `evaluate.py` run; by-lighting numbers recorded
- [ ] 8 `demo.py` works live; confidence threshold tuned

## Suggested split of work
- Likhitha: repo owner, preprocessing + tests, poster
- Samiksha: data collection lead, metadata hygiene
- Pooja: model/training/evaluation
- Priyanka: demo app, speech output, usability testing
(Adjust freely; everyone should collect data.)

## Poster panels: fill from real outputs only
| Panel | Source |
|---|---|
| Dataset | last lines printed by `prepare_dataset.py` (classes, images, signers), sample crops from `processed/` |
| Architecture | MobileNetV2 (ImageNet) + GAP + Dropout 0.3 + softmax(N); input 224x224; fine-tune top 30 layers |
| Results | `results/results.md`, `confusion_matrix.png`, `training_curves.png` |
Do not enter estimated numbers; mark anything unmeasured as "projected".
