import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { applyAgentEvent, resetAgentScratch } from '../src/agent/agentEvents';
import { useAgentStore } from '../src/agent/agentStore';
import type { AgentActivityEvent } from '../src/hooks/useAgentActivity';

const N = 20;
let seq = 0;
const ev = (type: string, fields: Partial<AgentActivityEvent> = {}): AgentActivityEvent =>
  ({ type, seq: ++seq, ts: 1000 + seq, ...fields });
const apply = (e: AgentActivityEvent, now: number, hydrate = false) =>
  applyAgentEvent(e, { now, speed: 1, frameCount: N, live: true, hydrate });

beforeEach(() => {
  vi.useFakeTimers();
  vi.spyOn(performance, 'now').mockImplementation(() => Date.now());
  seq = 0;
  resetAgentScratch();
  useAgentStore.setState({ replay: null });
  useAgentStore.getState().reset(null);
  apply(ev('agent.started', { model: 'muse-spark-1.3', instruction: 'label the board' }), Date.now());
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

it('waits for the box animation, then shows the mask result in order', () => {
  const t0 = Date.now();
  apply(ev('segment.started', { frame: 3, bbox: [10, 10, 50, 40], pos: [], neg: [] }), t0);
  apply(ev('mask.proposed', { frame: 3, preview_id: 'p1', score: 0.96 }), t0 + 100);
  // The canvas gets the mask at once; the feed still says the step is running.
  expect(useAgentStore.getState().beat?.previewId).toBe('p1');
  expect(useAgentStore.getState().feed.at(-1)).toMatchObject({ state: 'running' });
  expect(useAgentStore.getState().activity.text).toBe('Drawing a box on frame 4');
  vi.advanceTimersByTime(3000);
  expect(useAgentStore.getState().feed.at(-1)).toMatchObject({ state: 'done', meta: 'score 0.96' });
});

it('a duplicate track.created behind the animation does not create a broken track', () => {
  const t0 = Date.now();
  apply(ev('segment.started', { frame: 3, bbox: [10, 10, 50, 40] }), t0);
  apply(ev('track.created', { track_id: 2, frame: 3, source: 'backend' }), t0 + 50);   // queued
  apply(ev('track.created', { track_id: 2, frame: 3, label: 'board' }), t0 + 60);      // CLI copy
  expect(useAgentStore.getState().tracks[2]).toBeUndefined();
  vi.advanceTimersByTime(3000);
  const track = useAgentStore.getState().tracks[2];
  expect(track.statuses).toHaveLength(N);
  expect(track.statuses[3]).toBe('keyframe');
  expect(track.label).toBe('board');
});

it('only says the agent looked at a sheet once it opens the image', () => {
  const now = Date.now();
  apply(ev('artifact.created', { action: 'inspect_sequence', frames: [0, 5, 10] }), now);
  expect(useAgentStore.getState().feed).toHaveLength(0);
  expect(useAgentStore.getState().sheet).toBeNull();
  apply(ev('agent.tool.started', { tool: 'read_file', view: { kind: 'contact_sheet' } }), now + 10);
  expect(useAgentStore.getState().sheet?.frames).toEqual([0, 5, 10]);
  expect(useAgentStore.getState().feed.at(-1)).toMatchObject({ text: 'Looked over the sequence', meta: '3 frames' });
});

it('catching up applies end states without animating or queueing', () => {
  const now = Date.now();
  apply(ev('segment.started', { frame: 3, bbox: [10, 10, 50, 40] }), now, true);
  apply(ev('mask.proposed', { frame: 3, preview_id: 'p1', score: 0.9 }), now, true);
  apply(ev('track.created', { track_id: 1, frame: 3, label: 'board' }), now, true);
  const state = useAgentStore.getState();
  expect(state.beat).toBeNull();
  expect(state.tracks[1].statuses[3]).toBe('keyframe');
  expect(state.feed.map((f) => f.text)).toEqual(['Mask proposal on frame 4', 'Created the track']);
});

it('marks the frames of reported issues for review when tracking ends', () => {
  const now = Date.now();
  apply(ev('track.created', { track_id: 1, frame: 0, label: 'board' }), now);
  apply(ev('track.started', { action: 'propagate', track_id: 1 }), now);
  apply(ev('track.frames', { track_id: 1, frames: Array.from({ length: N }, (_, f) => [f, f !== 7]),
    progress: { done: N, total: N } }), now);
  apply(ev('track.finished', { track_id: 1, issues: [{ start: 4, end: 5 }] }), now);
  const statuses = useAgentStore.getState().tracks[1].statuses;
  expect(statuses.slice(0, 8)).toEqual(['keyframe', 'ok', 'ok', 'ok', 'review', 'review', 'ok', 'empty']);
});

it('keeps sentences about plumbing out of the story', () => {
  const now = Date.now();
  apply(ev('agent.message', { text: 'I will check the services first.' }), now);
  apply(ev('agent.tool.started', { tool: 'bash', agentseg: [{ op: 'status', args: {} }] }), now);
  apply(ev('agent.message', { text: 'I will list the labels.' }), now);
  apply(ev('agent.tool.started', { tool: 'bash', agentseg: [{ op: 'labels', args: {} }] }), now);
  apply(ev('agent.message', { text: 'Frame 5 shows the board, so I will look closer.' }), now);
  apply(ev('agent.tool.started', { tool: 'write_todos' }), now);   // its own todo list: neither
  apply(ev('agent.tool.started', { tool: 'bash', call_id: 'c1', agentseg: [{ op: 'view', args: { frame: 4 } }] }), now);
  const state = useAgentStore.getState();
  expect(state.feed.map((f) => f.text)).toEqual(['Frame 5 shows the board, so I will look closer.']);
  expect(state.action).toMatchObject({ id: 'c1', tool: 'view', detail: 'frame 5', state: 'running' });
  apply(ev('agent.tool.completed', { tool: 'bash', call_id: 'c1' }), now);
  expect(useAgentStore.getState().action?.state).toBe('done');
});

it('a label lands in the picker when the cursor clicks it', () => {
  useAgentStore.setState({ replay: { id: 'r', speed: 1, playing: true, position: 0, duration: 10, finished: false,
    hiddenLabels: ['board'], recordedTracks: [], synthetic: false } });
  apply(ev('label.created', { label: 'board', label_id: 3 }), Date.now());
  let state = useAgentStore.getState();
  expect(state.cursor?.target).toEqual({ kind: 'anchor', anchor: 'label-picker' });
  expect(state.recentLabel).toBeNull();
  expect(state.replay?.hiddenLabels).toEqual(['board']);
  vi.advanceTimersByTime(1000);
  state = useAgentStore.getState();
  expect(state.recentLabel?.name).toBe('board');
  expect(state.replay?.hiddenLabels).toEqual([]);
});

it('picks a frame from the open overview by clicking its tile', () => {
  const now = Date.now();
  apply(ev('artifact.created', { action: 'inspect_sequence', frames: [0, 5, 10] }), now);
  apply(ev('agent.tool.started', { tool: 'read_file', view: { kind: 'contact_sheet' } }), now);
  apply(ev('frame.started', { frame: 5 }), now);
  let state = useAgentStore.getState();
  expect(state.cursor?.target).toEqual({ kind: 'sheet', frame: 5 });
  expect(state.sheet?.pick).toBe(5);
  expect(state.focusFrame).toBeNull();
  vi.advanceTimersByTime(2000);
  state = useAgentStore.getState();
  expect(state.sheet).toBeNull();
  expect(state.focusFrame).toBe(5);
  // Without the overview, the frame is picked on the timeline.
  apply(ev('frame.started', { frame: 9 }), Date.now());
  expect(useAgentStore.getState().cursor?.target).toEqual({ kind: 'timeline', frame: 9, track: undefined });
});

it('the canvas follows the newest frame while the track fills in', () => {
  const now = Date.now();
  apply(ev('track.created', { track_id: 1, frame: 0, label: 'board' }), now);
  apply(ev('track.started', { action: 'propagate', track_id: 1 }), now);
  apply(ev('track.frames', { track_id: 1, frames: [[1, true], [2, true]], progress: { done: 2, total: N } }), now);
  expect(useAgentStore.getState().focusFrame).toBe(2);
  apply(ev('track.frames', { track_id: 1, frames: [[3, true]], progress: { done: 3, total: N } }), now + 20);
  expect(useAgentStore.getState().focusFrame).toBe(2);   // throttled
  apply(ev('track.frames', { track_id: 1, frames: [[4, true]], progress: { done: 4, total: N } }), now + 200);
  expect(useAgentStore.getState().focusFrame).toBe(4);
});

it('a second try with the same box only adds clicks, and promoting it draws nothing new', () => {
  const t0 = Date.now();
  apply(ev('segment.started', { frame: 3, bbox: [10, 10, 50, 40], pos: [], neg: [] }), t0);
  apply(ev('mask.proposed', { frame: 3, preview_id: 'p1', score: 0.9, bbox: [12, 11, 48, 40] }), t0 + 50);
  vi.advanceTimersByTime(3000);
  apply(ev('segment.started', { frame: 3, bbox: [10, 10, 50, 40], pos: [], neg: [[20, 38]] }), Date.now());
  let beat = useAgentStore.getState().beat!;
  expect(beat).toMatchObject({ refine: true, fromBox: [12, 11, 48, 40], prevPreviewId: 'p1' });
  apply(ev('mask.proposed', { frame: 3, preview_id: 'p2', score: 0.9, bbox: [12, 11, 48, 30] }), Date.now());
  vi.advanceTimersByTime(3000);
  beat = useAgentStore.getState().beat!;
  expect(beat.maskBox).toEqual([12, 11, 48, 30]);
  apply(ev('mask.refine_started', { frame: 3, preview_id: 'p2', bbox: [10, 10, 50, 40] }), Date.now());
  expect(useAgentStore.getState().beat!.id).toBe(beat.id);
  // A different box is a new prompt, drawn again.
  apply(ev('segment.started', { frame: 3, bbox: [5, 5, 30, 30] }), Date.now());
  expect(useAgentStore.getState().beat!.refine).toBeUndefined();
});
