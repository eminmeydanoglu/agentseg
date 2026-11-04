/**
 * Screen positions for the agent's cursor. The canvas registers its stage
 * transform; everything else is found through data attributes:
 *
 * - `data-agent-anchor="<name>"`   a control (label picker, the Muse panel)
 * - `data-agent-strip` + `data-track` + `data-n`   a timeline strip
 * - `data-agent-track-row="<id>"`  a track's name in the timeline
 * - `data-agent-sheet-frame="<f>"` a tile of the overview
 */

export interface Pt { x: number; y: number }

interface CanvasSpace { el: HTMLElement; scale: number; x: number; y: number }

let canvas: CanvasSpace | null = null;

export function setCanvasSpace(space: CanvasSpace | null) {
  canvas = space;
}

export function canvasScale(): number {
  return canvas?.scale ?? 1;
}

/** Image pixel -> viewport point, or null when no canvas is on screen. */
export function imageToScreen(x: number, y: number): Pt | null {
  if (!canvas || !canvas.el.isConnected) return null;
  const rect = canvas.el.getBoundingClientRect();
  return { x: rect.left + canvas.x + x * canvas.scale, y: rect.top + canvas.y + y * canvas.scale };
}

function rectOf(selector: string): DOMRect | null {
  const el = document.querySelector(selector);
  if (!el) return null;
  const rect = el.getBoundingClientRect();
  return rect.width || rect.height ? rect : null;
}

export function anchorPoint(name: string): Pt | null {
  const rect = rectOf(`[data-agent-anchor="${name}"]`);
  return rect ? { x: rect.left + rect.width * 0.62, y: rect.top + rect.height * 0.58 } : null;
}

export function timelinePoint(frame: number, track?: number): Pt | null {
  const el = (track != null ? document.querySelector(`[data-agent-strip][data-track="${track}"]`) : null)
    ?? document.querySelector('[data-agent-strip]');
  if (!(el instanceof HTMLElement)) return null;
  const rect = el.getBoundingClientRect();
  const n = Number(el.dataset.n) || 1;
  return { x: rect.left + ((frame + 0.5) / n) * rect.width, y: rect.top + rect.height * 0.6 };
}

export function trackRowPoint(track: number): Pt | null {
  const rect = rectOf(`[data-agent-track-row="${track}"]`);
  // Just right of the name, so the name stays readable.
  return rect ? { x: rect.right - 8, y: rect.top + rect.height * 0.62 } : null;
}

export function sheetTilePoint(frame: number): Pt | null {
  const rect = rectOf(`[data-agent-sheet-frame="${frame}"]`);
  return rect ? { x: rect.left + rect.width * 0.55, y: rect.top + rect.height * 0.55 } : null;
}
