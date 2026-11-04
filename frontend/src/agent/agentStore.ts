import { create } from 'zustand';

/**
 * What the labeling agent is doing, derived from activity events.
 *
 * Live runs feed it from the SSE stream; recorded runs feed it from a replay
 * clock (see replay.ts). Every visual the UI shows for the agent is backed by
 * one of these fields, and every field is set from a real event.
 */

export type FrameStatus = 'keyframe' | 'ok' | 'review' | 'absent' | 'empty';
export type RunStatus = 'running' | 'done' | 'failed' | 'stopped';
export type ActivityKind = 'thinking' | 'acting' | 'computing' | 'idle';

/** The stages of a labeling run, in order; the run log groups its steps by them. */
export type Phase = 'find' | 'mask' | 'track' | 'review' | 'export';
export const PHASES: Phase[] = ['find', 'mask', 'track', 'review', 'export'];
export const PHASE_NAMES: Record<Phase, string> = {
  find: 'Find the object',
  mask: 'First mask',
  track: 'Track',
  review: 'Review issues',
  export: 'Export',
};

export interface FeedItem {
  id: string;
  kind: 'thought' | 'step' | 'report' | 'error';
  text: string;
  meta?: string;
  state?: 'running' | 'done' | 'failed';
  ts: number;
  phase?: Phase;          // set on steps; a sentence belongs to the step it introduces
  frame?: number;         // the frame the step is about (the run log links to it)
  frames?: number[];      // an overview's frames
}

/** A spot SAM flagged after tracking, and what the agent did about it. */
export interface IssueView {
  key: string;
  trackId: number;
  id?: number;
  start: number;
  end: number;
  worst: number;
  kind?: string;
  state: 'open' | 'reviewed' | 'fixed';
  note?: string;
}

/** One box prompt on the canvas: cursor drags the box, clicks the points, SAM answers. */
export interface BoxBeat {
  id: number;
  frame: number;
  bbox: [number, number, number, number];
  pos: [number, number][];
  neg: [number, number][];
  t0: number;            // display clock (performance.now) when the beat started
  speed: number;         // animation speed (replays can run faster)
  maskAt?: number;       // when the mask proposal arrived
  previewId?: string;
  maskBox?: [number, number, number, number];  // the proposal's own extent; the box settles onto it
  /** Same box again on the same frame: only the clicks are new. The box is not redrawn,
   * and the previous proposal stays up until the new one arrives. */
  refine?: boolean;
  fromBox?: [number, number, number, number];
  prevPreviewId?: string;
}

export interface TrackView {
  id: number;
  label?: string;
  labelId?: number;
  n: number;
  statuses: (FrameStatus | null)[];
  keyframes: number[];
  propagating: boolean;
  progress?: { done: number; total: number };
}

export interface SheetView {
  frames: number[];
  trackId?: number;
  pick?: number;          // tile the agent went on to work with
  pickDelay?: number;     // ms until the cursor clicks that tile
  t0: number;
}

/** A close-up the agent opened (`view --zoom`, or the zoom `segment` writes). */
export interface ZoomView {
  frame: number;
  trackId?: number;
  previewId?: string;
  t0: number;
}

/** The tool call the agent is running now (or ran last), as the bubble shows it. */
export interface AgentAction {
  id: string;
  tool: string;           // the CLI operation, e.g. "segment", "track.propagate"
  detail?: string;        // what it is applied to, e.g. "frame 67"
  result?: string;        // what came back, e.g. "score 0.96"
  state: 'running' | 'done' | 'failed';
}

/** Where the agent's cursor goes next, outside the canvas box prompts. */
export type CursorTarget =
  | { kind: 'anchor'; anchor: string }                                  // [data-agent-anchor=...]
  | { kind: 'timeline'; frame: number; track?: number }
  | { kind: 'timelineRange'; from: number; to: number; track?: number } // press, drag, release
  | { kind: 'sheet'; frame: number }
  | { kind: 'trackRow'; track: number };

export interface CursorCue {
  id: number;
  target: CursorTarget;
  t0: number;             // display clock
  speed: number;
  click: boolean;
  leave?: boolean;        // fade out after arriving (end of the run)
}

export interface AgentRun {
  id?: string;
  model?: string;
  instruction?: string;
  status: RunStatus;
  startedAt: number;      // event clock (seconds)
  endedAt?: number;
  usage?: { calls?: number; input_tokens?: number; output_tokens?: number };
}

export interface ReplayState {
  id: string;
  speed: number;
  playing: boolean;
  position: number;       // seconds into the (compressed) replay
  duration: number;
  finished: boolean;
  hiddenLabels: string[]; // labels the recording creates, hidden until it does
  recordedTracks: number[]; // tracks the recording creates; other tracks show as they are
  synthetic: boolean;     // hand-made demo recording, not a real agent run
}

interface AgentState {
  run: AgentRun | null;
  activity: { kind: ActivityKind; text: string; progress?: { done: number; total: number } };
  feed: FeedItem[];
  beat: BoxBeat | null;
  focusFrame: number | null;
  focusSeq: number;       // bumps on every focus request, so re-focusing the same frame still follows
  sheet: SheetView | null;
  zoom: ZoomView | null;
  tracks: Record<number, TrackView>;
  recentLabel: { name: string; id?: number; t0: number } | null;
  replay: ReplayState | null;
  eventClock: number;     // ts of the last applied event (seconds)
  dismissed: boolean;
  action: AgentAction | null;
  cursor: CursorCue | null;
  logOpen: boolean;
  phase: Phase | null;
  issues: IssueView[];

  reset: (run: AgentRun | null) => void;
  patch: (fn: (s: AgentState) => Partial<AgentState>) => void;
  dismiss: () => void;
  setLogOpen: (open: boolean) => void;
}

const IDLE = { kind: 'idle' as ActivityKind, text: '' };

export const useAgentStore = create<AgentState>()((set) => ({
  run: null,
  activity: IDLE,
  feed: [],
  beat: null,
  focusFrame: null,
  focusSeq: 0,
  sheet: null,
  zoom: null,
  tracks: {},
  recentLabel: null,
  replay: null,
  eventClock: 0,
  dismissed: false,
  action: null,
  cursor: null,
  logOpen: false,
  phase: null,
  issues: [],

  reset: (run) => set((s) => ({
    run,
    activity: run ? { kind: 'thinking', text: 'Starting' } : IDLE,
    feed: [],
    beat: null,
    focusFrame: null,
    sheet: null,
    zoom: null,
    tracks: {},
    recentLabel: null,
    dismissed: false,
    action: null,
    cursor: null,
    phase: null,
    issues: [],
    replay: s.replay,
  })),
  patch: (fn) => set((s) => fn(s)),
  dismiss: () => set({ dismissed: true }),
  setLogOpen: (logOpen) => set({ logOpen }),
}));

/** True while one of the agent's tracks is filling in (the canvas follows it). */
export function agentPropagating(): boolean {
  return Object.values(useAgentStore.getState().tracks).some((t) => t.propagating);
}

/** "muse-spark-1.3-contributor" -> "Muse Spark 1.3" */
export function modelDisplayName(model?: string): string {
  if (!model) return 'Muse';
  const core = model.replace(/-(contributor|preview|latest)$/i, '');
  return core
    .split('-')
    .map((part) => (/^\d/.test(part) ? part : part.charAt(0).toUpperCase() + part.slice(1)))
    .join(' ');
}

export const AGENT_COLOR = '#a594ff';

export const STATUS_COLORS: Record<FrameStatus, string> = {
  keyframe: '#e8c547',
  ok: '#3fa66b',
  review: '#e0823d',
  absent: '#5c6370',
  empty: '#d9534f',
};

export const STATUS_WORDS: Record<FrameStatus, string> = {
  keyframe: 'keyframe',
  ok: 'tracked',
  review: 'flagged for review',
  absent: 'not visible',
  empty: 'no mask',
};
