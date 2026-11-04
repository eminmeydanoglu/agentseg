import type { AgentActivityEvent } from '../hooks/useAgentActivity';
import {
  useAgentStore,
  type AgentAction,
  type AgentRun,
  type BoxBeat,
  type CursorTarget,
  type FeedItem,
  type FrameStatus,
  type IssueView,
  type Phase,
  type TrackView,
} from './agentStore';
import { beatSettlesAt, cueActsAt } from './beatPlan';

/**
 * Turns activity events (agent.*, CLI and backend) into the agent view state.
 *
 * The story is curated on purpose: the agent's own sentences about the
 * labeling, the tool call it is running, and the steps a person would care
 * about (label, mask, track, keyframes, export). Plumbing (reading the skill,
 * checking services, opening the project) stays out of sight, and so do the
 * sentences the agent says about it.
 *
 * Steps a person would do by hand with the mouse (pick a frame, add a label,
 * mark frames, remove a keyframe) are shown with the agent's cursor doing
 * them; their result lands when the cursor clicks.
 */

export interface ApplyOptions {
  now: number;          // display clock, performance.now()
  speed: number;        // animation speed factor (1 live, >1 fast replay)
  frameCount: number;   // images in the project
  live: boolean;
  /** Catching up on past events: apply end states at once, no animation, no queue. */
  hydrate?: boolean;
}

type Args = Record<string, unknown>;

interface Scratch {
  seen: Set<string>;
  segmentStep: Map<number, string>;     // frame -> running feed item id
  propagateStep: Map<number, string>;   // track -> running feed item id
  tries: Map<number, number>;           // frame -> box prompts so far
  beatId: number;
  cueId: number;
  lastThought: string;
  pendingThought: { id: string; ts: number; text: string } | null;  // waits for the call it introduces
  busyUntil: number;                    // display clock: the hand is still moving until then
  deferred: (() => void)[];
  timer: number;
  trackLabels: Map<number, string>;     // label names announced by the CLI's track.created copy
  pendingSheet: { frames: number[]; trackId?: number } | null;  // rendered, not yet read by the agent
  lastFollowAt: number;                 // display clock of the last jump while tracking
  phase: Phase;
}

const scratch: Scratch = {
  seen: new Set(),
  segmentStep: new Map(),
  propagateStep: new Map(),
  tries: new Map(),
  beatId: 0,
  cueId: 0,
  lastThought: '',
  pendingThought: null,
  busyUntil: -Infinity,
  deferred: [],
  timer: 0,
  trackLabels: new Map(),
  pendingSheet: null,
  lastFollowAt: -Infinity,
  phase: 'find',
};

/** After a box drag, let the mask settle on screen before the story moves on. */
const SETTLE_MS = 450;
/** A picked sheet tile stays outlined this long before the canvas shows that frame. */
const PICK_HOLD_MS = 650;
/** While SAM tracks, the canvas follows the newest frame at most this often. */
const FOLLOW_EVERY_MS = 140;

function schedule() {
  if (scratch.timer) return;
  scratch.timer = window.setTimeout(flushDeferred, Math.max(0, scratch.busyUntil - performance.now()));
}

function flushDeferred() {
  scratch.timer = 0;
  // A paused replay holds its queue too.
  const replay = useAgentStore.getState().replay;
  if (replay && !replay.playing && !replay.finished) {
    scratch.timer = window.setTimeout(flushDeferred, 120);
    return;
  }
  const now = performance.now();
  if (now < scratch.busyUntil) {
    scratch.timer = window.setTimeout(flushDeferred, scratch.busyUntil - now);
    return;
  }
  while (scratch.deferred.length) {
    scratch.deferred.shift()!();
    // A deferred step may start a new animation; the rest waits for that one.
    if (performance.now() < scratch.busyUntil) {
      scratch.timer = window.setTimeout(flushDeferred, scratch.busyUntil - performance.now());
      return;
    }
  }
}

export function resetAgentScratch() {
  window.clearTimeout(scratch.timer);
  scratch.timer = 0;
  scratch.deferred = [];
  scratch.busyUntil = -Infinity;
  scratch.trackLabels.clear();
  scratch.pendingSheet = null;
  scratch.pendingThought = null;
  scratch.seen.clear();
  scratch.segmentStep.clear();
  scratch.propagateStep.clear();
  scratch.tries.clear();
  scratch.lastThought = '';
  scratch.lastFollowAt = -Infinity;
  scratch.phase = 'find';
}

const PHASE_ORDER: Phase[] = ['find', 'mask', 'track', 'review', 'export'];
/** Move the run on to a later stage (never back: a fix during review stays in review). */
function advancePhase(phase: Phase) {
  if (PHASE_ORDER.indexOf(phase) > PHASE_ORDER.indexOf(scratch.phase)) scratch.phase = phase;
  if (useAgentStore.getState().phase !== scratch.phase) useAgentStore.setState({ phase: scratch.phase });
}

/** Issues touching these frames were fixed (a keyframe, or marked not visible). */
function fixIssues(trackId: number, frames: number[], note: string) {
  useAgentStore.setState((s) => ({
    issues: s.issues.map((i) => (i.trackId === trackId && frames.some((f) => f >= i.start && f <= i.end)
      ? { ...i, state: 'fixed', note: i.note ?? note } : i)),
  }));
}

const num = (v: unknown): number | undefined => (typeof v === 'number' && Number.isFinite(v) ? v : undefined);
const human = (frame: number) => frame + 1;
const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

function box(v: unknown): [number, number, number, number] | undefined {
  return Array.isArray(v) && v.length === 4 && v.every((x) => typeof x === 'number')
    ? (v as [number, number, number, number])
    : undefined;
}

function points(v: unknown): [number, number][] {
  return Array.isArray(v)
    ? v.filter((p): p is [number, number] => Array.isArray(p) && p.length === 2 && p.every((x) => typeof x === 'number'))
    : [];
}

function ranges(frames: number[]): string {
  const sorted = [...new Set(frames)].sort((a, b) => a - b);
  const out: string[] = [];
  for (let i = 0; i < sorted.length;) {
    let j = i;
    while (j + 1 < sorted.length && sorted[j + 1] === sorted[j] + 1) j++;
    out.push(i === j ? `${human(sorted[i])}` : `${human(sorted[i])}–${human(sorted[j])}`);
    i = j + 1;
  }
  return out.join(', ');
}

/** Agent text -> plain sentences (the UI is not a markdown viewer). */
export function plainText(text: string): string {
  return text
    .replace(/```[\s\S]*?```/g, '')
    .replace(/`([^`]*)`/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/__([^_]+)__/g, '$1')
    .replace(/^#{1,6}\s+/gm, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function calls(ev: AgentActivityEvent): { op: string; args: Args }[] {
  const list = ev.agentseg as { op?: string; args?: Args }[] | undefined;
  if (!Array.isArray(list)) return [];
  return list.filter((c) => c.op).map((c) => ({ op: c.op!, args: c.args ?? {} }));
}

/** CLI operations that only prepare the work. */
const PLUMBING_OPS = new Set(['status', 'up', 'down', 'doctor', 'project.open', 'project.info', 'help', 'track.list']);
/** Agent tools that neither do nor prepare work a person would see (its own todo list). */
const SILENT_TOOLS = new Set(['write_todos', 'update_plan', 'todo_write']);

type ToolKind = 'work' | 'plumbing' | 'silent';

export function toolKind(ev: AgentActivityEvent): ToolKind {
  const tool = String(ev.tool ?? '');
  if (SILENT_TOOLS.has(tool)) return 'silent';
  if (ev.view) return 'work';
  const ops = calls(ev);
  if (!ops.length) return 'plumbing';  // read_skill, ls, plain file reads
  const work = ops.some((c) => !PLUMBING_OPS.has(c.op) && !(c.op === 'labels' && typeof c.args.add !== 'string'));
  return work ? 'work' : 'plumbing';
}

/** The tool call as the bubble shows it: the CLI operation and what it is applied to. */
function toolAction(ev: AgentActivityEvent): Omit<AgentAction, 'id' | 'state'> | undefined {
  const view = ev.view as { kind?: string; frame?: number; track_id?: number } | undefined;
  if (view?.kind === 'contact_sheet') return { tool: 'read', detail: view.track_id != null ? `overview of track ${view.track_id}` : 'overview' };
  if (view?.kind === 'frame_overlay' && view.frame != null) return { tool: 'read', detail: `frame ${human(view.frame)}` };
  if (view?.kind === 'mask_preview' && view.frame != null) return { tool: 'read', detail: `mask on frame ${human(view.frame)}` };
  const call = calls(ev).find((c) => !PLUMBING_OPS.has(c.op) && !(c.op === 'labels' && typeof c.args.add !== 'string'));
  if (!call) return undefined;
  const a = call.args;
  const f = num(a.frame);
  const track = num(a.track);
  switch (call.op) {
    case 'labels': return { tool: 'labels --add', detail: String(a.add) };
    case 'sheet': return { tool: 'sheet', detail: track != null ? `track ${track}` : 'whole sequence' };
    case 'view': return { tool: 'view', detail: [f != null ? `frame ${human(f)}` : '', track != null ? `track ${track}` : ''].filter(Boolean).join(' · ') };
    case 'segment': return { tool: 'segment', detail: f != null ? `frame ${human(f)} · box` : 'box' };
    case 'track.create': return { tool: 'track create', detail: typeof a.label === 'string' ? a.label : undefined };
    case 'track.propagate': return { tool: 'track propagate', detail: track != null ? `track ${track}` : undefined };
    case 'track.refine': return { tool: 'track refine', detail: f != null ? `frame ${human(f)}` : undefined };
    case 'track.absent': return { tool: 'track absent', detail: track != null ? `track ${track}` : undefined };
    case 'track.keyframes': return { tool: a.rm != null ? 'track keyframes --rm' : 'track keyframes',
      detail: num(a.rm) != null ? `frame ${human(num(a.rm)!)}` : track != null ? `track ${track}` : undefined };
    case 'track.issues': return { tool: 'track issues', detail: track != null ? `track ${track}` : undefined };
    case 'export': return { tool: 'export', detail: typeof a.format === 'string' ? a.format.replace('yolo-seg', 'YOLO-seg') : undefined };
    default: return { tool: call.op.replace('.', ' ') };
  }
}

function emptyTrack(id: number, n: number): TrackView {
  return { id, n, statuses: new Array(n).fill(null), keyframes: [], propagating: false };
}

export function applyAgentEvent(ev: AgentActivityEvent, opts: ApplyOptions): void {
  const store = useAgentStore.getState();
  const ts = typeof ev.ts === 'number' ? ev.ts : Date.now() / 1000;
  const seq = String(ev.seq ?? ev.id ?? `${ts}:${ev.type}`);
  const t = ev.type;

  if (t === 'agent.started') {
    resetAgentScratch();
    store.reset({
      id: (ev.agent_run_id as string) ?? ev.run_id,
      model: ev.model as string | undefined,
      instruction: ev.instruction as string | undefined,
      status: 'running',
      startedAt: ts,
    });
    useAgentStore.setState({ eventClock: ts });
    return;
  }

  // Backend and CLI both announce some results; show each once.
  const dedupe = t === 'mask.proposed' ? `mask:${ev.preview_id}`
    : t === 'track.created' ? `track:${ev.track_id}`
    : undefined;
  if (dedupe) {
    if (scratch.seen.has(dedupe)) {
      // The CLI's copy carries the label. The first copy may still wait in the
      // queue, so remember the name instead of patching a track that isn't there yet.
      if (t === 'track.created' && typeof ev.label === 'string') {
        const id = num(ev.track_id);
        if (id != null) {
          scratch.trackLabels.set(id, ev.label);
          store.patch((s) => (s.tracks[id] ? { tracks: { ...s.tracks, [id]: { ...s.tracks[id], label: ev.label as string } } } : {}));
        }
      }
      return;
    }
    scratch.seen.add(dedupe);
  }

  // The mask arrives while the hand may still be drawing: the canvas needs it now.
  if (t === 'mask.proposed') {
    const f = num(ev.frame);
    const pid = typeof ev.preview_id === 'string' ? ev.preview_id : undefined;
    store.patch((s) => ({
      beat: s.beat && s.beat.frame === f && !s.beat.previewId
        ? { ...s.beat, previewId: pid, maskAt: opts.now, maskBox: box(ev.bbox) }
        : s.beat,
    }));
  }

  // Everything else is shown in order, after the current animation.
  if (!opts.hydrate && (opts.now < scratch.busyUntil || scratch.deferred.length)) {
    scratch.deferred.push(() => run({ ...opts, now: performance.now() }));
    schedule();
    return;
  }
  run(opts);

  function run(opts: ApplyOptions) {
    // Work that waits for the cursor's click, ahead of any later event.
    const soon: (() => void)[] = [];
    handle(opts, soon);
    if (soon.length) {
      scratch.deferred.unshift(...soon);
      schedule();
    }
  }

  function handle(opts: ApplyOptions, soon: (() => void)[]) {
  const push = (item: Omit<FeedItem, 'id' | 'ts'> & { id?: string }) => {
    const full: FeedItem = { id: item.id ?? seq, ts, ...(item.kind === 'step' ? { phase: scratch.phase } : {}), ...item };
    store.patch((s) => ({ feed: [...s.feed, full] }));
    return full.id;
  };
  const update = (id: string | undefined, fields: Partial<FeedItem>) => {
    if (!id) return;
    store.patch((s) => ({ feed: s.feed.map((f) => (f.id === id ? { ...f, ...fields } : f)) }));
  };
  const setActivity = (kind: 'thinking' | 'acting' | 'computing' | 'idle', text: string, progress?: { done: number; total: number }) =>
    store.patch(() => ({ activity: { kind, text, progress } }));
  /** Result of the tool call on screen, when it is the one that produced it. */
  const actionResult = (tools: string[], fields: Partial<AgentAction>) =>
    store.patch((s) => (s.action && tools.includes(s.action.tool) ? { action: { ...s.action, ...fields } } : {}));
  const track = (id: number): TrackView => useAgentStore.getState().tracks[id] ?? emptyTrack(id, opts.frameCount);
  const putTrack = (tv: TrackView) => store.patch((s) => ({ tracks: { ...s.tracks, [tv.id]: tv } }));
  /** Run `fn` once the cursor has done its part (`ms` from now); later events wait for it. */
  const later = (ms: number, fn: () => void) => {
    if (opts.hydrate) { fn(); return; }
    scratch.busyUntil = Math.max(scratch.busyUntil, opts.now + ms);
    soon.push(fn);
  };
  const cue = (target: CursorTarget, click: boolean, leave = false) => {
    if (opts.hydrate) return;
    store.patch(() => ({ cursor: { id: ++scratch.cueId, target, t0: opts.now, speed: opts.speed, click, leave } }));
  };
  const jump = (frame: number) => store.patch((s) => (opts.hydrate
    ? { focusFrame: frame, sheet: null, zoom: null }   // one jump at the end, not one per event
    : { focusFrame: frame, focusSeq: s.focusSeq + 1, sheet: null, zoom: null }));
  /**
   * Go to a frame. A frame the agent picks itself is clicked: on the overview
   * tile when the overview is open, otherwise on the timeline.
   */
  const focus = (frame: number, how: 'jump' | 'click' = 'jump', trackId?: number) => {
    if (opts.hydrate) { jump(frame); return; }
    const s = useAgentStore.getState();
    if (s.sheet && s.sheet.pick == null && s.sheet.frames.includes(frame)) {
      const click = cueActsAt('click', opts.speed);
      cue({ kind: 'sheet', frame }, true);
      store.patch((st) => ({ sheet: st.sheet ? { ...st.sheet, pick: frame, pickDelay: click } : st.sheet }));
      later(click + PICK_HOLD_MS / Math.max(0.25, opts.speed), () => jump(frame));
      return;
    }
    if (how === 'click') {
      cue({ kind: 'timeline', frame, track: trackId }, true);
      later(cueActsAt('click', opts.speed), () => jump(frame));
      return;
    }
    jump(frame);
  };
  const pushThought = (thought: { id: string; ts: number; text: string }) =>
    store.patch((s) => ({ feed: [...s.feed, { id: thought.id, ts: thought.ts, kind: 'thought', text: thought.text }] }));

  useAgentStore.setState({ eventClock: ts });

  switch (t) {
    case 'agent.thinking':
      if (useAgentStore.getState().activity.kind !== 'computing') setActivity('thinking', 'Thinking');
      return;

    case 'agent.message': {
      const text = plainText(String(ev.text ?? ev.message ?? ''));
      if (!text || text === scratch.lastThought) return;
      scratch.lastThought = text;
      const long = text.length > 320 || (text.includes('\n') && text.length > 160);
      if (long) {
        scratch.pendingThought = null;
        push({ kind: 'report', text });
        return;
      }
      // A sentence introduces the call that follows it; it is shown with that call,
      // and not at all when the call is plumbing.
      if (scratch.pendingThought) pushThought(scratch.pendingThought);
      scratch.pendingThought = { id: seq, ts, text };
      return;
    }

    case 'agent.tool.started': {
      const kind = toolKind(ev);
      if (kind === 'silent') return;
      const pending = scratch.pendingThought;
      scratch.pendingThought = null;
      if (kind === 'plumbing') {
        // Getting ready: nothing to narrate yet.
        if (!useAgentStore.getState().action) setActivity('acting', 'Getting ready');
        return;
      }
      if (pending) pushThought(pending);
      const act = toolAction(ev);
      if (act) {
        const id = typeof ev.call_id === 'string' ? ev.call_id : seq;
        store.patch(() => ({ action: { id, ...act, state: 'running' } }));
        setActivity(ev.view ? 'thinking' : 'acting', act.tool);
      }
      // Reading an image is when the agent actually looks; only then does the overview show.
      const view = ev.view as { kind?: string; frame?: number; track_id?: number; preview_id?: string } | undefined;
      if (view?.kind === 'zoom' && typeof view.frame === 'number') {
        // The close-up it reads is what the person sees too, until the agent acts on it.
        if (!opts.hydrate) store.patch(() => ({ sheet: null, zoom: { frame: view.frame!, trackId: view.track_id,
          previewId: view.preview_id, t0: opts.now } }));
        push({ kind: 'step', state: 'done', frame: view.frame,
          text: `Looked closely at frame ${human(view.frame)}`,
          meta: view.preview_id ? 'whole object?' : 'same object? whole object?' });
      } else if (!view && useAgentStore.getState().zoom) {
        store.patch(() => ({ zoom: null }));
      }
      if (view?.kind === 'contact_sheet' && scratch.pendingSheet) {
        const sheet = scratch.pendingSheet;
        scratch.pendingSheet = null;
        if (!opts.hydrate) store.patch(() => ({ sheet: { frames: sheet.frames, trackId: sheet.trackId, t0: opts.now } }));
        push({ kind: 'step', state: 'done',
          text: sheet.trackId != null ? 'Looked over the track' : 'Looked over the sequence',
          meta: plural(sheet.frames.length, 'frame'), frames: sheet.frames });
      }
      return;
    }

    case 'agent.tool.completed': {
      const id = typeof ev.call_id === 'string' ? ev.call_id : undefined;
      store.patch((s) => (s.action && s.action.state === 'running' && (!id || s.action.id === id)
        ? { action: { ...s.action, state: ev.status === 'error' ? 'failed' : 'done' } }
        : {}));
      return;
    }

    case 'label.created': {
      const name = String(ev.label ?? '');
      if (!name) return;
      advancePhase('find');
      const show = () => {
        store.patch((s) => ({
          recentLabel: { name, id: num(ev.label_id), t0: performance.now() },
          replay: s.replay ? { ...s.replay, hiddenLabels: s.replay.hiddenLabels.filter((l) => l !== name) } : null,
        }));
        push({ kind: 'step', text: `Added the label ${name}`, state: 'done' });
        // The name is typed into the picker; the story waits for it.
        if (!opts.hydrate) scratch.busyUntil = Math.max(scratch.busyUntil, performance.now() + name.length * 38 + 500);
      };
      cue({ kind: 'anchor', anchor: 'label-picker' }, true);
      later(cueActsAt('click', opts.speed), show);
      return;
    }

    case 'artifact.created': {
      if (ev.action !== 'inspect_sequence') return;
      const frames = Array.isArray(ev.frames) ? (ev.frames as number[]) : [];
      if (!frames.length) return;
      // Rendered for the agent; it is shown (and reported) once the agent opens it.
      scratch.pendingSheet = { frames, trackId: num(ev.track_id) };
      actionResult(['sheet'], { result: plural(frames.length, 'frame'), state: 'done' });
      return;
    }

    case 'frame.started': {
      // `agentseg view N`: the agent chose this frame to look at.
      const f = num(ev.frame);
      if (f == null) return;
      focus(f, 'click', num(ev.track_id));
      // With a track it is a check of the mask there; without, a look at the frame itself.
      push({ kind: 'step', state: 'done', text: `${num(ev.track_id) != null ? 'Checked' : 'Looked at'} frame ${human(f)}`, frame: f });
      return;
    }

    case 'segment.started': {
      const f = num(ev.frame);
      const b = box(ev.bbox);
      if (f == null || !b) return;
      advancePhase('mask');
      focus(f);
      const tries = (scratch.tries.get(f) ?? 0) + 1;
      scratch.tries.set(f, tries);
      // The same box again on this frame: the agent is only adding clicks to its last proposal.
      const prev = useAgentStore.getState().beat;
      const refine = prev != null && prev.frame === f && b.every((v, i) => Math.abs(v - prev.bbox[i]) <= 4);
      const beat: BoxBeat = { id: ++scratch.beatId, frame: f, bbox: b, pos: points(ev.pos), neg: points(ev.neg),
        t0: opts.now, speed: opts.speed,
        ...(refine ? { refine, fromBox: prev.maskBox ?? prev.bbox, prevPreviewId: prev.previewId } : {}) };
      if (!opts.hydrate) {
        store.patch(() => ({ beat, sheet: null }));
        scratch.busyUntil = beatSettlesAt(beat) + SETTLE_MS / Math.max(0.25, opts.speed);
      }
      const clicks = [beat.pos.length ? plural(beat.pos.length, '+ point') : '', beat.neg.length ? plural(beat.neg.length, '− point') : '']
        .filter(Boolean);
      // A second try keeps its box (the canvas shows that); only the new clicks are listed.
      actionResult(['segment'], { detail: [`frame ${human(f)}`, (refine && clicks.length ? clicks : ['box', ...clicks]).join(', ')].join(' · ') });
      setActivity('acting', `Drawing a box on frame ${human(f)}`);
      scratch.segmentStep.set(f, push({ kind: 'step', state: 'running', text: `Mask proposal on frame ${human(f)}`,
        meta: tries > 1 ? `try ${tries}` : undefined, frame: f }));
      return;
    }

    case 'mask.proposed': {
      const f = num(ev.frame);
      const pid = typeof ev.preview_id === 'string' ? ev.preview_id : undefined;
      if (f == null) return;
      store.patch((s) => ({
        beat: s.beat && s.beat.frame === f && !s.beat.previewId
          ? { ...s.beat, previewId: pid, maskAt: opts.now, maskBox: box(ev.bbox) }
          : s.beat,
      }));
      const score = num(ev.score);
      const stepId = scratch.segmentStep.get(f);
      const tries = scratch.tries.get(f) ?? 1;
      update(stepId, { state: 'done', meta: [tries > 1 ? `try ${tries}` : '', score != null ? `score ${score.toFixed(2)}` : '']
        .filter(Boolean).join(' · ') || undefined });
      scratch.segmentStep.delete(f);
      if (score != null) actionResult(['segment', 'track refine'], { result: `score ${score.toFixed(2)}`, state: 'done' });
      setActivity('thinking', `Checking the mask on frame ${human(f)}`);
      return;
    }

    case 'track.created': {
      const id = num(ev.track_id);
      if (id == null) return;
      const f = num(ev.frame);
      const tv = track(id);
      const statuses = [...tv.statuses];
      if (f != null) statuses[f] = 'keyframe';
      putTrack({ ...tv, statuses, keyframes: f != null ? [f] : [],
        label: (ev.label as string) ?? scratch.trackLabels.get(id) ?? tv.label,
        labelId: num(ev.label_id) ?? tv.labelId });
      store.patch(() => ({ beat: null }));
      actionResult(['track create'], { result: `track ${id}`, state: 'done' });
      advancePhase('mask');
      push({ kind: 'step', state: 'done', text: 'Created the track',
        meta: typeof ev.label === 'string' ? ev.label : undefined, frame: f });
      return;
    }

    case 'track.started': {
      if (ev.action !== 'propagate') return;
      const id = num(ev.track_id);
      if (id == null) return;
      advancePhase('track');
      putTrack({ ...track(id), propagating: true, progress: undefined });
      store.patch(() => ({ beat: null }));
      cue({ kind: 'trackRow', track: id }, false);
      setActivity('computing', 'Tracking through the sequence');
      scratch.propagateStep.set(id, push({ kind: 'step', state: 'running', text: 'Tracking through the sequence' }));
      return;
    }

    case 'track.frames': {
      const id = num(ev.track_id);
      if (id == null) return;
      const tv = track(id);
      const statuses = [...tv.statuses];
      const batch = ((ev.frames as [number, boolean][]) ?? []).filter((p) => Array.isArray(p) && typeof p[0] === 'number');
      for (const [f, visible] of batch) {
        if (f < 0 || f >= statuses.length) continue;
        if (statuses[f] === 'keyframe' || statuses[f] === 'absent') continue;
        statuses[f] = visible ? 'ok' : 'empty';
      }
      const p = ev.progress as { done?: number; total?: number } | undefined;
      const progress = p?.done != null && p?.total ? { done: p.done, total: p.total } : tv.progress;
      putTrack({ ...tv, statuses, progress, propagating: true });
      if (progress) {
        setActivity('computing', 'Tracking through the sequence', progress);
        actionResult(['track propagate'], { result: `${progress.done} / ${progress.total}` });
      }
      // The canvas rides along with SAM: the newest frame, with its new mask.
      const newest = batch.at(-1)?.[0];
      if (!opts.hydrate && newest != null && opts.now - scratch.lastFollowAt >= FOLLOW_EVERY_MS / Math.max(0.25, opts.speed)) {
        scratch.lastFollowAt = opts.now;
        jump(newest);
      }
      return;
    }

    case 'track.finished': {
      const id = num(ev.track_id);
      if (id == null) return;
      const tv = track(id);
      const statuses = [...tv.statuses];
      const issues = (ev.issues as { id?: number; start?: number; end?: number; worst_frame?: number; kind?: string }[]) ?? [];
      const flagged: number[] = [];
      for (const issue of issues) {
        if (issue.start == null || issue.end == null) continue;
        for (let f = issue.start; f <= issue.end && f < statuses.length; f++) {
          if (statuses[f] === 'ok') { statuses[f] = 'review'; flagged.push(f); }
        }
      }
      putTrack({ ...tv, statuses, propagating: false, progress: undefined });
      const views: IssueView[] = issues.filter((i) => i.start != null && i.end != null).map((i) => ({
        key: `${id}:${i.id ?? i.start}`, trackId: id, id: i.id, start: i.start!, end: i.end!,
        worst: i.worst_frame ?? i.start!, kind: i.kind, state: 'open' }));
      store.patch((s) => ({ issues: [...s.issues.filter((i) => i.trackId !== id), ...views] }));
      advancePhase('review');
      const meta = issues.length ? `${plural(issues.length, 'spot')} to review` : 'no issues';
      update(scratch.propagateStep.get(id), { state: 'done', meta });
      actionResult(['track propagate'], { result: meta, state: 'done' });
      scratch.propagateStep.delete(id);
      setActivity('thinking', issues.length ? 'Reviewing the flagged frames' : 'Checking the result');
      return;
    }

    case 'track.lost': {
      // SAM stopped finding the object, so tracking in that direction stopped: the agent takes over.
      const id = num(ev.track_id);
      const from = num(ev.frame);
      if (id == null || from == null) return;
      const untracked = Array.isArray(ev.untracked) ? (ev.untracked as number[]) : [];
      if (untracked.length === 2) {
        const tv = track(id);
        const statuses = [...tv.statuses];
        const [a, b] = [Math.min(...untracked), Math.max(...untracked)];
        for (let f = a; f <= b && f < statuses.length; f++) {
          if (statuses[f] !== 'keyframe' && statuses[f] !== 'absent') statuses[f] = null;
        }
        putTrack({ ...tv, statuses });
      }
      push({ kind: 'step', state: 'done', text: `Lost the object at frame ${human(from)}: stopped tracking there`, frame: from });
      if (!opts.hydrate) jump(from);
      return;
    }

    case 'issue.reviewed': {
      // The agent looked at these spots and the masks are right: no longer "review".
      const id = num(ev.track_id);
      if (id == null) return;
      const tv = track(id);
      const statuses = [...tv.statuses];
      const reviewed = (ev.ranges as [number, number][]) ?? [];
      for (const [a, b] of reviewed) {
        for (let f = a; f <= b && f < statuses.length; f++) if (statuses[f] === 'review') statuses[f] = 'ok';
      }
      putTrack({ ...tv, statuses });
      const note = typeof ev.note === 'string' && ev.note ? ev.note : undefined;
      const ids = Array.isArray(ev.issue_ids) ? (ev.issue_ids as number[]) : null;
      store.patch((s) => ({
        issues: s.issues.map((i) => (i.trackId === id && i.state === 'open'
          && (ids && i.id != null ? ids.includes(i.id) : reviewed.some(([a, b]) => a <= i.end && b >= i.start))
          ? { ...i, state: 'reviewed', note: note ?? i.note } : i)),
      }));
      if (reviewed.length) {
        push({ kind: 'step', state: 'done', text: `Checked ${reviewed.length === 1 ? 'frames' : 'spots'} ${reviewed.map(([a, b]) => (a === b ? `${human(a)}` : `${human(a)}–${human(b)}`)).join(', ')}`,
          meta: 'mask is right', frame: reviewed[0][0] });
      }
      actionResult(['track review'], { result: String(ev.message ?? 'checked'), state: 'done' });
      return;
    }

    case 'mask.refine_started': {
      const f = num(ev.frame);
      if (f == null) return;
      focus(f);
      const b = box(ev.bbox);
      // Promoting the proposal already on screen draws nothing new.
      const cur = useAgentStore.getState().beat;
      const promotes = cur != null && cur.frame === f && cur.previewId != null && cur.previewId === ev.preview_id;
      if (b && !opts.hydrate && !promotes) {
        const beat: BoxBeat = { id: ++scratch.beatId, frame: f, bbox: b, pos: points(ev.pos), neg: points(ev.neg),
          t0: opts.now, speed: opts.speed };
        store.patch(() => ({ beat }));
        scratch.busyUntil = beatSettlesAt(beat) + SETTLE_MS / Math.max(0.25, opts.speed);
      }
      setActivity('acting', `Adding a keyframe on frame ${human(f)}`);
      return;
    }

    case 'keyframe.added': {
      const id = num(ev.track_id);
      const f = num(ev.frame);
      if (id == null || f == null) return;
      const tv = track(id);
      const statuses = [...tv.statuses];
      statuses[f] = 'keyframe';
      putTrack({ ...tv, statuses, keyframes: [...new Set([...tv.keyframes, f])].sort((a, b) => a - b) });
      store.patch(() => ({ beat: null }));
      actionResult(['track refine'], { result: 'keyframe added', state: 'done' });
      fixIssues(id, [f], `keyframe on frame ${human(f)}`);
      push({ kind: 'step', state: 'done', text: `Added a keyframe on frame ${human(f)}`, frame: f });
      return;
    }

    case 'keyframe.removed': {
      const id = num(ev.track_id);
      const f = num(ev.frame);
      if (id == null || f == null) return;
      cue({ kind: 'timeline', frame: f, track: id }, true);
      later(cueActsAt('click', opts.speed), () => {
        const tv = track(id);
        const statuses = [...tv.statuses];
        if (statuses[f] === 'keyframe') statuses[f] = null;
        putTrack({ ...tv, statuses, keyframes: tv.keyframes.filter((k) => k !== f) });
        actionResult(['track keyframes --rm'], { result: 'removed', state: 'done' });
        push({ kind: 'step', state: 'done', text: `Removed the keyframe on frame ${human(f)}`, frame: f });
      });
      return;
    }

    case 'track.absent': {
      const id = num(ev.track_id);
      const frames = Array.isArray(ev.frames) ? (ev.frames as number[]) : [];
      if (id == null || !frames.length) return;
      cue({ kind: 'timelineRange', from: Math.min(...frames), to: Math.max(...frames), track: id }, false);
      later(cueActsAt('drag', opts.speed), () => {
        const tv = track(id);
        const statuses = [...tv.statuses];
        for (const f of frames) if (f >= 0 && f < statuses.length) statuses[f] = 'absent';
        putTrack({ ...tv, statuses });
        const text = `${frames.length === 1 ? 'frame' : 'frames'} ${ranges(frames)}`;
        actionResult(['track absent'], { result: text, state: 'done' });
        fixIssues(id, frames, 'marked not visible');
        push({ kind: 'step', state: 'done', text: `Marked ${text} as not visible`, frame: Math.min(...frames) });
      });
      return;
    }

    case 'export.completed': {
      const format = typeof ev.format === 'string' ? ev.format.replace('yolo-seg', 'YOLO-seg') : undefined;
      advancePhase('export');
      push({ kind: 'step', state: 'done', text: 'Exported the labels', meta: format });
      actionResult(['export'], { result: 'written', state: 'done' });
      setActivity('thinking', 'Writing the report');
      return;
    }

    case 'command.failed': {
      const run = useAgentStore.getState().run;
      if (!run || run.status !== 'running') return;
      const message = String(ev.error ?? ev.message ?? 'Command failed');
      if (/interrupted/i.test(message)) return;
      push({ kind: 'error', text: message.length > 160 ? `${message.slice(0, 159)}…` : message });
      return;
    }

    case 'agent.finished':
    case 'agent.failed': {
      const failed = t === 'agent.failed';
      const stopped = ev.status === 'stopped';
      if (scratch.pendingThought) { pushThought(scratch.pendingThought); scratch.pendingThought = null; }
      store.patch((s) => ({
        run: s.run ? { ...s.run, status: failed ? 'failed' : stopped ? 'stopped' : 'done', endedAt: ts,
          usage: (ev.usage as AgentRun['usage']) ?? s.run.usage } : s.run,
        beat: null,
        sheet: null,
        activity: { kind: 'idle', text: failed ? 'Stopped with an error' : stopped ? 'Stopped' : 'Finished' },
        action: s.action?.state === 'running' ? { ...s.action, state: 'failed' } : s.action,
      }));
      // The cursor goes back to the panel and fades out.
      cue({ kind: 'anchor', anchor: 'agent-bubble' }, false, true);
      if (failed) push({ kind: 'error', text: String(ev.error ?? ev.message ?? 'The agent stopped with an error') });
      // Mark steps that never got an answer.
      store.patch((s) => ({ feed: s.feed.map((f) => (f.state === 'running' ? { ...f, state: 'failed' } : f)) }));
      return;
    }

    default:
      return;
  }
  }
}

/** Frame statuses implied by a set of events, for hydrating a page opened mid-run. */
export function statusesFromServer(statuses: (string | null)[]): (FrameStatus | null)[] {
  return statuses.map((s) => (s === 'keyframe' || s === 'ok' || s === 'review' || s === 'absent' ? s : null));
}
