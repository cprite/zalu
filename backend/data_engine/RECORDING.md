# Recording guide: footage for the vest demo

Goal: enough real footage to train, validate and **independently test** a "person without safety vest" detector,
and to show an honest before/after comparison.

## What to record

| Session | Use | Length | What changes vs. the others |
|---|---|---|---|
| A | train | 2–3 min | baseline: room lights on, camera position 1 |
| B | train/val | 2–3 min | different time of day, or blinds closed, **or** a different camera angle |
| C | **independent test** | 2–3 min | a third condition (another angle or room) and, if possible, different people or clothes |

- Session C goes to Damir only. It is never used for training, validation or fal edits.
- Each session is one continuous recording (one file). Do not cut one recording into several files; the pipeline
  treats each file as an independent session, and splitting a recording would leak near-identical frames
  between train and test.

## In every session

- People **with** and **without** a vest, roughly half each. The violation class (`person_without_vest`) matters
  most, so make sure it appears often.
- Walk across the whole frame at different distances: near, middle, far.
- Sometimes two or three people at once, sometimes nobody (empty frames are useful negatives).
- Put the vest on and take it off on camera now and then, so the same person appears in both classes.
- Avoid people standing at the very edge of the frame (those labels get flagged for review).

## Camera

- Fixed position (tripod, shelf, laptop). The webcam used for the live demo is ideal, because the model will
  then be trained on the same optics it is deployed on.
- 720p or 1080p, normal frame rate. The pipeline samples about 1 frame per second, so 3 minutes gives about
  180 candidate frames before de-duplication.
- Don't move the camera during a session; move it between sessions.

## Consent and privacy

- Everyone on camera agrees to being recorded and to the footage being used for the demo.
- Record in a space where people who haven't agreed won't walk through.
- Footage stays on our machines and the GX10; don't upload it anywhere public.

## Then

```bash
# training data: sessions A and B (each file = one session)
python -m backend.data_engine run --task spec.json --media sessionA.mp4 sessionB.mp4 --out runs/vest --device mps
# independent test set (or POST /api/datasets with kind=test once the split check accepts it)
python -c "from backend.data_engine import process_test_upload; from backend.task_engine import TaskSpec; \
  print(process_test_upload(TaskSpec.model_validate_json(open('spec.json').read()), ['sessionC.mp4']).data_yaml)"
```

Review every flagged frame, and **all** test-set frames, before training or reporting metrics.
