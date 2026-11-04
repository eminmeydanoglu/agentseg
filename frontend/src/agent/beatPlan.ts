import type { BoxBeat } from './agentStore';

/** Timing of a box-prompt animation (ms at speed 1). Shared by the canvas
 * layer that draws it and the reducer that waits for it before moving on. */
export const BEAT = {
  move: 420,
  drag: 650,
  pointMove: 260,
  click: 140,
  park: 300,
  maskFade: 260,
  boxFit: 420,      // the box settling onto the mask's edges
  cursorLinger: 900,
  cursorFade: 350,
};

export interface BeatPlan {
  dragStart: number;
  dragEnd: number;
  points: { x: number; y: number; type: 'positive' | 'negative'; moveStart: number; clickAt: number }[];
  parkStart: number;
  actionsEnd: number;
}

export function planBeat(beat: Pick<BoxBeat, 'pos' | 'neg' | 'speed' | 'refine'>): BeatPlan {
  const k = 1 / Math.max(0.25, beat.speed);
  // A refinement keeps the box: the hand goes straight to the clicks.
  let t = beat.refine ? 0 : BEAT.move * k;
  const dragStart = t;
  if (!beat.refine) t += BEAT.drag * k;
  const dragEnd = t;
  const points = [
    ...beat.pos.map(([x, y]) => ({ x, y, type: 'positive' as const })),
    ...beat.neg.map(([x, y]) => ({ x, y, type: 'negative' as const })),
  ].map((p, i) => {
    const moveStart = t;
    t += (beat.refine && i === 0 ? BEAT.move : BEAT.pointMove) * k;
    const clickAt = t;
    t += BEAT.click * k;
    return { ...p, moveStart, clickAt };
  });
  const parkStart = t;
  t += BEAT.park * k;
  return { dragStart, dragEnd, points, parkStart, actionsEnd: t };
}

/** When the hand is done and the mask is on screen (display clock). */
export function beatSettlesAt(beat: BoxBeat): number {
  const plan = planBeat(beat);
  return beat.t0 + plan.actionsEnd + BEAT.maskFade / Math.max(0.25, beat.speed);
}

/** Cursor moves outside the canvas: go to a target, click, or press-drag-release. */
export const CUE = {
  move: 560,
  click: 160,
  drag: 650,
};

/** ms until a cue's click lands (or its drag is released). */
export function cueActsAt(kind: 'click' | 'drag' | 'move', speed: number): number {
  const k = 1 / Math.max(0.25, speed);
  if (kind === 'drag') return (CUE.move + CUE.click + CUE.drag) * k;
  if (kind === 'click') return (CUE.move + CUE.click) * k;
  return CUE.move * k;
}

export const easeInOut = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
export const clamp01 = (t: number) => Math.max(0, Math.min(1, t));
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

export interface BeatCursor {
  x: number;              // image coordinates
  y: number;
  pressed: boolean;
  /** 0..1 while travelling to the box corner (the start point is wherever the cursor was). */
  approach?: number;
}

/** Where the hand is `t` ms into a box prompt. */
export function beatCursor(beat: Pick<BoxBeat, 'bbox' | 'speed' | 'refine'>, p: BeatPlan, t: number, parkOffset: { x: number; y: number }): BeatCursor {
  const k = 1 / Math.max(0.25, beat.speed);
  const [x0, y0, x1, y1] = beat.bbox;
  const first = p.points[0];
  if (beat.refine && first && t < first.clickAt) {
    return { x: first.x, y: first.y, pressed: false, approach: easeInOut(clamp01(t / first.clickAt)) };
  }
  if (t < p.dragStart) return { x: x0, y: y0, pressed: false, approach: easeInOut(clamp01(t / p.dragStart)) };
  if (t < p.dragEnd) {
    const u = easeInOut(clamp01((t - p.dragStart) / (p.dragEnd - p.dragStart)));
    return { x: lerp(x0, x1, u), y: lerp(y0, y1, u), pressed: true };
  }
  let x = x1;
  let y = y1;
  let pressed = false;
  let fromX = x1;
  let fromY = y1;
  for (const pt of p.points) {
    if (t < pt.moveStart) break;
    const u = easeInOut(clamp01((t - pt.moveStart) / (pt.clickAt - pt.moveStart)));
    x = lerp(fromX, pt.x, u); y = lerp(fromY, pt.y, u);
    if (t >= pt.clickAt && t < pt.clickAt + BEAT.click * k) pressed = true;
    fromX = pt.x; fromY = pt.y;
  }
  if (t >= p.parkStart) {
    // Step aside so the mask is not hidden under the cursor.
    // (After clicks only, it steps aside from the last click instead of the box corner.)
    const u = easeInOut(clamp01((t - p.parkStart) / (p.actionsEnd - p.parkStart)));
    const [px, py] = beat.refine ? [fromX, fromY] : [x1, y1];
    x = lerp(fromX, px + parkOffset.x, u); y = lerp(fromY, py + parkOffset.y, u);
  }
  return { x, y, pressed };
}
