"use client";

import { useEffect, useRef, useState } from "react";
import { sampleResult, type FrameResult, type FrameSample } from "./session";

type Infer = (frame: Blob) => Promise<FrameResult>;
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export function useInferenceSession(infer: Infer, simulated: boolean, classes: readonly string[], target: string) {
  const epoch = useRef(0);
  const stream = useRef<MediaStream | null>(null);
  const objectUrl = useRef<string | null>(null);
  const [samples, setSamples] = useState<FrameSample[]>([]);
  const [preview, setPreview] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function release() {
    epoch.current++;
    stream.current?.getTracks().forEach((track) => track.stop());
    stream.current = null;
  }

  useEffect(() => () => {
    release();
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
  }, []);

  function stop() {
    release();
    setRunning(false);
    setLoading(false);
  }

  function publish(result: FrameResult, frame?: Blob) {
    const at = Date.now();
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    objectUrl.current = frame ? URL.createObjectURL(frame) : null;
    setPreview(objectUrl.current);
    setSamples((current) => [...current.filter((sample) => sample.at >= at - 600000).slice(-1199),
      { at, result, eventsEnabled: !!target }]);
  }

  async function useImage(frame: Blob) {
    stop();
    const run = epoch.current;
    setLoading(true);
    setError(null);
    try {
      const result = simulated ? sampleResult(classes, samples.length, target) : await infer(frame);
      if (epoch.current === run) publish(result, simulated ? undefined : frame);
    } catch (cause) {
      if (epoch.current === run) setError((cause as Error).message);
    } finally {
      if (epoch.current === run) setLoading(false);
    }
  }

  async function start() {
    stop();
    const run = epoch.current;
    setRunning(true);
    setLoading(true);
    setError(null);
    let video: HTMLVideoElement | undefined;
    try {
      if (!simulated) {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error("Camera access needs HTTPS or localhost.");
        const media = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        if (epoch.current !== run) {
          media.getTracks().forEach((track) => track.stop());
          return;
        }
        stream.current = media;
        video = document.createElement("video");
        video.muted = true;
        video.playsInline = true;
        video.srcObject = media;
        await video.play();
      }
      const canvas = document.createElement("canvas");
      for (let index = 0; epoch.current === run; index++) {
        let frame: Blob | undefined;
        if (video) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          canvas.getContext("2d")!.drawImage(video, 0, 0);
          frame = await new Promise<Blob>((resolve, reject) => canvas.toBlob(
            (blob) => blob ? resolve(blob) : reject(new Error("Could not capture a camera frame.")), "image/jpeg", 0.85,
          ));
          if (epoch.current !== run) break;
        }
        const result = frame ? await infer(frame) : sampleResult(classes, index, target);
        if (epoch.current !== run) break;
        publish(result, frame);
        setLoading(false);
        await sleep(600);
      }
    } catch (cause) {
      if (epoch.current === run) {
        setError((cause as Error).message);
        stop();
      }
    } finally {
      if (video) video.srcObject = null;
    }
  }

  return { samples, preview, running, loading, error, start, stop, useImage };
}
