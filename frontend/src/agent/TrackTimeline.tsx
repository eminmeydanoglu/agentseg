import { useCallback, useEffect, useRef, useState } from 'react';
import { cn } from '@/lib/utils';
import { useAnnotationStore } from '../stores/annotationStore';
import { useProjectStore } from '../stores/projectStore';
import { useUIStore } from '../stores/uiStore';
import { AGENT_COLOR, STATUS_COLORS, STATUS_WORDS, useAgentStore, type TrackView } from './agentStore';
import { CUE } from './beatPlan';

/**
 * One strip per track under the canvas: every frame colored by its state
 * (keyframe, tracked, flagged, not visible, no mask). It fills in while SAM
 * tracks. Click or drag to jump to a frame. Colors only; hovering names them.
 * While the agent works and there is no track yet, a plain strip of the
 * frames stands in, so the agent's cursor has somewhere to pick frames.
 */

const EMPTY = 'hsl(240 3.7% 15.9%)';

/** The agent sweeping over frames it is about to mark (see the track.absent cue). */
function Sweep({ track, n }: { track: number; n: number }) {
  const cue = useAgentStore((s) => s.cursor);
  if (!cue || cue.target.kind !== 'timelineRange' || cue.target.track !== track) return null;
  const k = 1 / Math.max(0.25, cue.speed);
  const { from, to } = cue.target;
  return (
    <span
      key={cue.id}
      className="agent-sweep pointer-events-none absolute top-[2px] h-[14px] rounded-[2px]"
      style={{
        left: `${(from / n) * 100}%`,
        width: `${((to - from + 1) / n) * 100}%`,
        minWidth: 3,
        background: `${AGENT_COLOR}55`,
        boxShadow: `inset 0 0 0 1px ${AGENT_COLOR}`,
        ['--sweep-delay' as string]: `${(CUE.move + CUE.click) * k}ms`,
        ['--sweep-ms' as string]: `${CUE.drag * k}ms`,
      }}
    />
  );
}

function Strip({ track, current, onSeek, onHover }: {
  track: TrackView;
  current: number;
  onSeek: (frame: number) => void;
  onHover: (frame: number | null, x: number) => void;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  const dragging = useRef(false);

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !width) return;
    const dpr = window.devicePixelRatio || 1;
    const h = 10;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, h);
    const n = track.statuses.length || 1;
    const cell = width / n;
    // Runs of equal status as single rects: crisp and cheap.
    for (let i = 0; i < n;) {
      let j = i;
      while (j + 1 < n && track.statuses[j + 1] === track.statuses[i]) j++;
      const s = track.statuses[i];
      ctx.fillStyle = s ? STATUS_COLORS[s] : EMPTY;
      const x = Math.floor(i * cell);
      ctx.fillRect(x, 0, Math.max(1, Math.ceil((j + 1) * cell) - x), h);
      i = j + 1;
    }
  }, [track.statuses, width]);

  const frameAt = useCallback((clientX: number) => {
    const rect = wrap.current!.getBoundingClientRect();
    const u = Math.max(0, Math.min(0.9999, (clientX - rect.left) / rect.width));
    return Math.floor(u * track.statuses.length);
  }, [track.statuses.length]);

  const n = track.statuses.length || 1;
  return (
    <div
      ref={wrap}
      data-agent-strip
      data-track={track.id >= 0 ? track.id : undefined}
      data-n={n}
      className="relative h-[18px] flex-1 cursor-pointer select-none"
      onPointerDown={(e) => { dragging.current = true; (e.target as HTMLElement).setPointerCapture(e.pointerId); onSeek(frameAt(e.clientX)); }}
      onPointerMove={(e) => {
        const f = frameAt(e.clientX);
        onHover(f, e.clientX - wrap.current!.getBoundingClientRect().left);
        if (dragging.current) onSeek(f);
      }}
      onPointerUp={() => { dragging.current = false; }}
      onPointerLeave={() => onHover(null, 0)}
    >
      <canvas ref={ref} className="absolute inset-x-0 top-[4px] h-[10px] w-full rounded-[2px]" />
      {track.keyframes.map((k) => (
        <span
          key={k}
          className="absolute top-0 h-[3px] w-[3px] -translate-x-1/2 rounded-[1px]"
          style={{ left: `${((k + 0.5) / n) * 100}%`, background: STATUS_COLORS.keyframe }}
        />
      ))}
      <span
        className="pointer-events-none absolute top-[1px] h-[16px] w-[2px] -translate-x-1/2 rounded-full bg-white shadow-[0_0_0_1px_rgb(0_0_0/0.5)]"
        style={{ left: `${((current + 0.5) / n) * 100}%` }}
      />
      {track.id >= 0 && <Sweep track={track.id} n={n} />}
    </div>
  );
}

export function TrackTimeline() {
  const tracks = useAgentStore((s) => s.tracks);
  const hiddenLabels = useAgentStore((s) => s.replay?.hiddenLabels);
  const agentShown = useAgentStore((s) => s.run != null && !(s.dismissed && s.run.status !== 'running'));
  const labels = useAnnotationStore((s) => s.labels);
  const { currentImageIndex, setCurrentImageIndex, images } = useProjectStore();
  const [hover, setHover] = useState<{ track: number; frame: number; x: number } | null>(null);

  const withFrames = Object.values(tracks).filter((t) => t.statuses.some((s) => s != null) || t.propagating);
  // No track yet: the frames alone (id -1), only while the agent is shown.
  const list = withFrames.length || !agentShown || images.length < 2
    ? withFrames
    : [{ id: -1, n: images.length, statuses: new Array<null>(images.length).fill(null), keyframes: [], propagating: false } as TrackView];
  if (!list.length) return null;

  const seek = (frame: number) => {
    if (useAgentStore.getState().run?.status === 'running') useUIStore.getState().setFollowAgent(false);
    setCurrentImageIndex(frame);
  };

  return (
    <div className="shrink-0 border-t bg-background px-4 py-2">
      {list.map((track) => {
        const label = track.id < 0 ? undefined : labels.find((l) => l.id === track.labelId || l.name === track.label);
        const name = track.id < 0 ? `${images.length} frames` : track.label ?? label?.name ?? `Track ${track.id}`;
        const visibleName = hiddenLabels?.includes(name) ? `Track ${track.id}` : name;
        const status = hover && hover.track === track.id ? track.statuses[hover.frame] : undefined;
        return (
          <div key={track.id} className="relative flex items-center gap-3 py-0.5">
            <div
              data-agent-track-row={track.id >= 0 ? track.id : undefined}
              className={cn('flex w-36 shrink-0 items-center gap-2 truncate text-xs', track.id < 0 ? 'text-muted-foreground' : 'text-foreground/80')}
            >
              {track.id >= 0 && <span className="h-2 w-2 shrink-0 rounded-[2px]" style={{ background: label?.color ?? '#888' }} />}
              <span className="truncate">{visibleName}</span>
            </div>
            <Strip
              track={track}
              current={currentImageIndex}
              onSeek={seek}
              onHover={(frame, x) => setHover(frame == null ? null : { track: track.id, frame, x })}
            />
            {hover && hover.track === track.id && (
              <div
                className="pointer-events-none absolute bottom-full z-20 mb-1 -translate-x-1/2 whitespace-nowrap rounded-md border bg-popover px-2 py-1 text-[11px] text-popover-foreground shadow-md"
                style={{ left: `calc(9rem + 0.75rem + ${hover.x}px)` }}
              >
                Frame {hover.frame + 1}
                {status ? <span className="text-muted-foreground"> · {STATUS_WORDS[status]}</span> : null}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
