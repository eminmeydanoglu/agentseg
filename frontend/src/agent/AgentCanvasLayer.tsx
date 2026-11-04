import { useEffect, useMemo, useState } from 'react';
import { Circle, Image as KonvaImage, Layer, Rect } from 'react-konva';
import { maskToCanvas, type RLEMask } from '../utils/rle';
import { useAgentStore } from './agentStore';
import { BEAT, beatCursor, clamp01, easeInOut, planBeat } from './beatPlan';

/**
 * What the agent's hand leaves on the canvas during a box prompt: the box
 * being dragged, the +/- clicks, and SAM's proposal, in the same styles as
 * the manual tools (label color, dashed box while dragging, green/red points).
 * Once the proposal is in, the box settles onto the mask's own edges, so the
 * box always shows what is selected. A second try with the same box only
 * clicks: the box stays, the previous proposal stays up until the new one
 * replaces it. The cursor itself is drawn by AgentCursor, above the whole page.
 */

type Box = [number, number, number, number];
const lerpBox = (a: Box, b: Box, u: number): Box => [0, 1, 2, 3].map((i) => a[i] + (b[i] - a[i]) * u) as Box;

/**
 * Re-render on every animation frame from `t0` until `t0 + until` (forever
 * while `until` is null, i.e. while the beat still waits for its mask).
 */
function useBeatClock(key: number | null, t0: number, until: number | null): number {
  const [now, setNow] = useState(0);
  useEffect(() => {
    if (key == null) return;
    let raf = 0;
    const loop = (t: number) => {
      setNow(t);
      if (until == null || t - t0 < until) raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [key, t0, until]);
  return now;
}

const maskCache = new Map<string, RLEMask>();

function usePreviewMask(previewId: string | undefined, color: string, opacity: number): HTMLCanvasElement | null {
  const [loaded, setLoaded] = useState<{ id: string; rle: RLEMask } | null>(null);
  useEffect(() => {
    if (!previewId || maskCache.has(previewId)) return;
    let cancelled = false;
    fetch(`/api/video/previews/${previewId}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((data: { mask_rle?: RLEMask } | null) => {
        if (cancelled || !data?.mask_rle) return;
        maskCache.set(previewId, data.mask_rle);
        setLoaded({ id: previewId, rle: data.mask_rle });
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [previewId]);
  const rle = previewId ? maskCache.get(previewId) ?? (loaded?.id === previewId ? loaded.rle : null) : null;
  return useMemo(() => (rle ? maskToCanvas(rle, color, opacity) : null), [rle, color, opacity]);
}

interface Props {
  frame: number;
  stageScale: number;
  labelColor: string;
  maskOpacity: number;
}

export function AgentCanvasLayer({ frame, stageScale, labelColor, maskOpacity }: Props) {
  const beat = useAgentStore((s) => s.beat);
  const onFrame = beat && beat.frame === frame ? beat : null;
  const p = useMemo(() => (onFrame ? planBeat(onFrame) : null), [onFrame]);
  const maskCanvas = usePreviewMask(onFrame?.previewId, labelColor, maskOpacity);
  const prevMaskCanvas = usePreviewMask(onFrame?.prevPreviewId, labelColor, maskOpacity);

  const k = onFrame ? 1 / Math.max(0.25, onFrame.speed) : 1;
  const maskShownAt = onFrame && p && onFrame.maskAt != null
    ? Math.max(onFrame.maskAt - onFrame.t0, p.actionsEnd)
    : undefined;
  const now = useBeatClock(onFrame?.id ?? null, onFrame?.t0 ?? 0,
    maskShownAt == null ? null : maskShownAt + Math.max(BEAT.maskFade, BEAT.boxFit) * k + 50);
  if (!onFrame || !p || now === 0) return null;

  const t = now - onFrame.t0;
  const [x0, y0, x1, y1] = onFrame.bbox;
  const px = 1 / stageScale;
  const hand = beatCursor(onFrame, p, t, { x: 14 * px, y: 10 * px });

  const refine = Boolean(onFrame.refine);
  const dragging = !refine && t >= p.dragStart && t < p.dragEnd;
  const boxVisible = refine || t >= p.dragStart;
  // The box: dragged out by the hand, or (second try) the one already there;
  // when the proposal arrives it settles onto the mask's edges.
  let shown: Box = refine ? onFrame.fromBox ?? onFrame.bbox : [x0, y0, dragging ? hand.x : x1, dragging ? hand.y : y1];
  if (maskShownAt != null && onFrame.maskBox && t >= maskShownAt) {
    shown = lerpBox(shown, onFrame.maskBox, easeInOut(clamp01((t - maskShownAt) / (BEAT.boxFit * k))));
  }
  const waiting = t >= p.actionsEnd && (maskShownAt == null || t < maskShownAt);
  // While SAM works the box breathes; slow, small, and only while waiting.
  const boxOpacity = waiting ? 0.55 + 0.45 * (0.5 + 0.5 * Math.cos((t - p.actionsEnd) / 260)) : 1;
  const maskOpacityNow = maskShownAt != null ? clamp01((t - maskShownAt) / (BEAT.maskFade * k)) : 0;

  return (
    <Layer listening={false}>
      {prevMaskCanvas && maskOpacityNow < 1 && (
        <KonvaImage image={prevMaskCanvas} x={0} y={0} opacity={1 - maskOpacityNow} />
      )}
      {maskCanvas && maskOpacityNow > 0 && (
        <KonvaImage image={maskCanvas} x={0} y={0} opacity={maskOpacityNow} />
      )}

      {boxVisible && (
        <Rect
          x={Math.min(shown[0], shown[2])}
          y={Math.min(shown[1], shown[3])}
          width={Math.abs(shown[2] - shown[0])}
          height={Math.abs(shown[3] - shown[1])}
          stroke={labelColor}
          strokeWidth={2 * px}
          dash={dragging ? [5 * px, 5 * px] : undefined}
          opacity={boxOpacity}
        />
      )}

      {p.points.map((pt, i) => {
        if (t < pt.clickAt) return null;
        const pop = 0.6 + 0.4 * easeInOut(clamp01((t - pt.clickAt) / (120 * k)));
        return (
          <Circle
            key={i}
            x={pt.x}
            y={pt.y}
            radius={6 * px * pop}
            fill={pt.type === 'positive' ? '#22c55e' : '#ef4444'}
            stroke="white"
            strokeWidth={2 * px}
          />
        );
      })}
    </Layer>
  );
}
