# Sample footage for judges

Two ready-made scenarios. Each has a **train** video (what you upload on step 2) and a separate **test** video from
different scenes the model has never seen (what you upload on step 5, Verify). All videos are 1280x720, no audio.

| Scenario | Prompt (step 1, Describe) | Train (step 2, Upload) | Test (step 5, Verify) |
|---|---|---|---|
| Hard hats | `Alert me when a person on site is not wearing a hard hat` | `helmet/helmet-train.mp4` (96 s) | `helmet/helmet-test.mp4` (47 s) |
| Hi-vis vests | `Alert me when a worker is not wearing a hi-vis safety vest` | `vest/vest-train.mp4` (25 s) | `vest/vest-test.mp4` (33 s) |

## Walkthrough (~5 minutes, hard-hat scenario)

1. **Describe**: paste the prompt. ZALU turns it into a task spec (classes, what counts as the event).
2. **Upload**: attach `helmet/helmet-train.mp4`. Frames are extracted and auto-labelled by a zero-shot detector.
3. **Prepare**: review the auto-labels; fix or delete wrong boxes if you like.
4. **Train**: start training (the 30-epoch preset is enough). A YOLO11n model trains on the GX10 GPU in under a minute.
5. **Verify**: upload `helmet/helmet-test.mp4` as the independent test set and evaluate. You get the fine-tuned
   model against the zero-shot baseline on footage neither has seen.
6. **Deploy**: download the ONNX model or try live inference.

What we measured on this scenario (14 Nov 2026): F1 0.49–0.50 for the trained YOLO11n against 0.13 for the
zero-shot YOLOE baseline, 213 vs 41 FPS on the GX10. Honest caveats: the test labels are auto-generated (review them
on Prepare for a fair score), and recall on "person without a hard hat" is low; ZALU's report says so and marks the
model "not ready".

The vest scenario is deliberately hard: the test video shows workers **without** vests in scenes unlike the training
video, so you can see where a small model fails and what the report says about it.

## AI Video Search (API)

The same four videos also work as "cameras" for the video search API (`/api/video-search`, see
`backend/video_search/STATUS.md`). Example query: `worker in a black jacket and a white helmet`.

## Sources and licences

- Intel IoT DevKit sample videos (https://github.com/intel-iot-devkit/sample-videos), licensed CC BY 4.0:
  parts of `helmet-train.mp4` (worker-zone-detection, people-detection, classroom, store-aisle-detection).
  Cut, rescaled to 1280x720 and concatenated by the ZALU team.
- Mixkit free stock video (https://mixkit.co), Mixkit free licence: the remaining clips in `helmet-train.mp4`
  (ids 1436, 1442, 1459, 23010, 23407, 23550), all of `helmet-test.mp4` (23237, 46753, 23243, 23378),
  `vest-test.mp4` (46753, 23406, 23378) and `vest-train.mp4`.
  Edited (cut, rescaled, concatenated) for this demo.
