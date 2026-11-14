# ZALU

**Zero-to-model Automated Labelling Unit.** Describe what to detect, get a small vision model that runs on your camera, with its accuracy measured on your own real frames.

Imperial AI Hackathon 2026 · Track 02: Voice, Vision & Generative Media · Team MIRAZH.

## How it works

1. **Prompt:** "find people without safety goggles", "which restaurant tables are free".
2. **Data:** a few minutes of your camera footage → ~40 diverse frames, plus synthetic rare cases generated with fal.
3. **Auto-labelling:** an open-vocabulary detector (YOLOE) labels every frame from the text prompt.
4. **Training:** a small YOLO model trained on a GPU in minutes.
5. **Proof:** accuracy on held-out real frames the model never saw, before and after, plus FPS on the device.

A planner model fine-tuned with Adaption turns the prompt into classes, negative classes and test scenarios.

## Status

Work in progress (build 14-15 Nov 2026).
