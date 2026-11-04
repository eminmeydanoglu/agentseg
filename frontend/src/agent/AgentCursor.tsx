import { useEffect, useRef } from 'react';
import { useProjectStore } from '../stores/projectStore';
import { useUIStore } from '../stores/uiStore';
import { AGENT_COLOR, useAgentStore, type CursorCue } from './agentStore';
import { beatCursor, clamp01, CUE, easeInOut, planBeat, type BeatPlan } from './beatPlan';
import { anchorPoint, canvasScale, imageToScreen, sheetTilePoint, timelinePoint, trackRowPoint, type Pt } from './cursorSpace';

/**
 * The agent's cursor, one for the whole page. It stays on screen for the
 * whole run and goes where the agent works: into the canvas for box prompts,
 * onto an overview tile or the timeline when it picks a frame, to the label
 * picker when it adds a label. Every move is backed by a real event (see the
 * cues in agentEvents.ts); it never wanders on its own.
 *
 * Positions are written straight to the DOM on each animation frame, so the
 * page does not re-render while it moves.
 */

// Classic arrow pointer, tip at (0, 0).
const CURSOR_PATH = 'M0 0 L0 16.2 L4.3 12.3 L7.1 18.7 L9.9 17.5 L7.2 11.3 L12.6 11.3 Z';
const RIPPLE_MS = 450;
const FADE_IN_MS = 280;
const LEAVE_MS = 420;

const lerpPt = (a: Pt, b: Pt, u: number): Pt => ({ x: a.x + (b.x - a.x) * u, y: a.y + (b.y - a.y) * u });

function cueTargets(cue: CursorCue): [Pt | null, Pt | null] {
  const t = cue.target;
  switch (t.kind) {
    case 'anchor': return [anchorPoint(t.anchor), null];
    case 'timeline': return [timelinePoint(t.frame, t.track), null];
    case 'timelineRange': return [timelinePoint(t.from, t.track), timelinePoint(t.to, t.track)];
    case 'sheet': return [sheetTilePoint(t.frame), null];
    case 'trackRow': return [trackRowPoint(t.track), null];
  }
}

export function AgentCursor() {
  const shown = useAgentStore((s) => s.run != null && !s.dismissed && (s.run.status === 'running' || s.cursor?.leave === true));
  const following = useUIStore((s) => s.followAgent);
  const root = useRef<HTMLDivElement>(null);
  const arrow = useRef<SVGSVGElement>(null);
  const ring = useRef<HTMLDivElement>(null);
  const tag = useRef<HTMLSpanElement>(null);
  const active = shown && following;

  useEffect(() => {
    if (!active) return;
    let raf = 0;
    let pos: Pt | null = null;
    let appearedAt = -1;
    let cueId = -1;
    let cueFrom: Pt | null = null;
    let beatId = -1;
    let beatFrom: Pt | null = null;
    let plan: BeatPlan | null = null;
    let lastBeatT0 = -Infinity;   // a cue older than the last box prompt is done: stay put
    let wasPressed = false;
    let rippleAt = -Infinity;

    const tick = (now: number) => {
      raf = requestAnimationFrame(tick);
      const el = root.current;
      if (!el || !arrow.current || !ring.current) return;
      const { beat, cursor } = useAgentStore.getState();
      const frame = useProjectStore.getState().currentImageIndex;
      if (pos == null) {
        // It comes out of the Muse panel.
        pos = anchorPoint('agent-bubble');
        appearedAt = now;
      }
      let next: Pt | null = pos;
      let pressed = false;
      let opacity = 1;

      if (beat && beat.frame === frame && (!cursor || beat.t0 >= cursor.t0)) {
        if (beat.id !== beatId) { beatId = beat.id; beatFrom = pos; plan = planBeat(beat); lastBeatT0 = beat.t0; }
        const s = canvasScale();
        const hand = beatCursor(beat, plan!, now - beat.t0, { x: 14 / s, y: 10 / s });
        const p = imageToScreen(hand.x, hand.y);
        if (p) next = hand.approach != null && beatFrom ? lerpPt(beatFrom, p, hand.approach) : p;
        pressed = hand.pressed;
      } else if (cursor && cursor.t0 >= lastBeatT0) {
        if (cursor.id !== cueId) { cueId = cursor.id; cueFrom = pos; }
        const k = 1 / Math.max(0.25, cursor.speed);
        const t = now - cursor.t0;
        const move = CUE.move * k;
        const [a, b] = cueTargets(cursor);
        if (a) {
          const from = cueFrom ?? a;
          if (t < move) next = lerpPt(from, a, easeInOut(clamp01(t / move)));
          else if (cursor.target.kind === 'timelineRange' && b) {
            // Press, drag along the frames, release.
            const u = easeInOut(clamp01((t - move - CUE.click * k) / (CUE.drag * k)));
            next = lerpPt(a, b, u);
            pressed = t < move + (CUE.click + CUE.drag) * k;
          } else {
            next = a;
            pressed = cursor.click && t < move + CUE.click * k;
          }
          if (cursor.leave) opacity = 1 - clamp01((t - move) / LEAVE_MS);
        }
      }

      if (next == null) { el.style.opacity = '0'; return; }
      pos = next;
      opacity *= clamp01((now - appearedAt) / FADE_IN_MS);
      el.style.opacity = String(opacity);
      el.style.transform = `translate3d(${next.x.toFixed(1)}px, ${next.y.toFixed(1)}px, 0)`;
      arrow.current.style.transform = pressed ? 'scale(0.86)' : 'scale(1)';
      if (tag.current) {
        // Near the bottom edge (the timeline) the name tag goes above the pointer.
        tag.current.style.top = next.y > window.innerHeight - 44 ? '-20px' : '17px';
      }

      if (pressed && !wasPressed) {
        rippleAt = now;
        ring.current.style.left = `${next.x}px`;
        ring.current.style.top = `${next.y}px`;
      }
      wasPressed = pressed;
      const r = (now - rippleAt) / RIPPLE_MS;
      if (r >= 0 && r < 1) {
        ring.current.style.opacity = String(0.55 * (1 - r));
        ring.current.style.transform = `translate(-50%, -50%) scale(${0.35 + 0.85 * easeInOut(r)})`;
      } else {
        ring.current.style.opacity = '0';
      }
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [active]);

  if (!active) return null;
  return (
    <>
      <div
        ref={ring}
        className="pointer-events-none fixed z-[60] h-7 w-7 rounded-full opacity-0"
        style={{ border: `1.5px solid ${AGENT_COLOR}` }}
        aria-hidden
      />
      <div ref={root} className="pointer-events-none fixed left-0 top-0 z-[60] opacity-0 will-change-transform" aria-hidden>
        <svg
          ref={arrow}
          width="15"
          height="21"
          viewBox="-1 -1 15 21"
          className="block origin-top-left -translate-x-px -translate-y-px transition-transform duration-100 [filter:drop-shadow(0_1px_2px_rgb(0_0_0/0.45))]"
        >
          <path d={CURSOR_PATH} fill={AGENT_COLOR} stroke="#fff" strokeWidth="1.2" strokeLinejoin="round" />
        </svg>
        <span
          ref={tag}
          className="absolute left-[13px] top-[17px] whitespace-nowrap rounded-[4px] px-[5px] py-[2px] text-[11px] font-medium leading-[14px] text-white shadow-sm"
          style={{ background: AGENT_COLOR }}
        >
          Muse
        </span>
      </div>
    </>
  );
}
