import type { AgentActivityEvent } from '../hooks/useAgentActivity';
import { useProjectStore } from '../stores/projectStore';
import { applyAgentEvent, resetAgentScratch, toolKind } from './agentEvents';
import { useAgentStore } from './agentStore';

/**
 * Plays a recorded agent run (GET /api/agent/recordings/{id}) through the same
 * reducer as live events. Long idle gaps (model thinking) are shortened so a
 * replay keeps its rhythm; everything else keeps its recorded timing.
 *
 * Open the UI with `?replay=<run id>` (optionally `&speed=2`). A recording
 * from another project opens that project first.
 */

const MAX_GAP_S = 2.4;          // a thinking pause never lasts longer than this
const SETUP_GAP_S = 0.6;        // before the real work starts (skill, services, project) nothing is shown
const PROPAGATION_SCALE = 0.5;  // tracking frames arrive faster than recorded

interface Scheduled { at: number; event: AgentActivityEvent }

let timeline: Scheduled[] = [];
let cursor = 0;
let raf = 0;
let lastTick = 0;
let generation = 0;             // bumps on every start/stop; stale fetches and loops check it

function compress(events: AgentActivityEvent[]): Scheduled[] {
  const out: Scheduled[] = [];
  let at = 0;
  let prev: number | undefined;
  let working = false;
  for (const event of events) {
    const ts = typeof event.ts === 'number' ? event.ts : prev ?? 0;
    if (event.type === 'agent.tool.started' && toolKind(event) === 'work') working = true;
    if (prev != null) {
      let gap = Math.max(0, ts - prev);
      if (event.type === 'track.frames' || event.type === 'track.progress') gap *= PROPAGATION_SCALE;
      at += Math.min(gap, working ? MAX_GAP_S : SETUP_GAP_S);
    }
    prev = ts;
    out.push({ at, event });
  }
  return out;
}

function loop(gen: number) {
  const tick = (now: number) => {
    if (gen !== generation) return;
    const replay = useAgentStore.getState().replay;
    if (!replay) return;
    if (!replay.playing) { lastTick = now; raf = requestAnimationFrame(tick); return; }
    const dt = Math.min(0.25, (now - lastTick) / 1000) * replay.speed;
    lastTick = now;
    const position = replay.position + dt;
    const frameCount = useProjectStore.getState().images.length;
    while (cursor < timeline.length && timeline[cursor].at <= position) {
      applyAgentEvent(timeline[cursor].event, { now, speed: replay.speed, frameCount, live: false });
      cursor++;
    }
    const finished = cursor >= timeline.length;
    useAgentStore.setState((s) => ({
      replay: s.replay ? { ...s.replay, position: Math.min(position, s.replay.duration), finished,
        playing: finished ? false : s.replay.playing } : null,
    }));
    if (!finished) raf = requestAnimationFrame(tick);
  };
  cancelAnimationFrame(raf);
  lastTick = performance.now();
  raf = requestAnimationFrame(tick);
}

export async function startReplay(id: string, speed = 1): Promise<void> {
  stopReplay();
  const gen = generation;
  const response = await fetch(`/api/agent/recordings/${encodeURIComponent(id)}`);
  if (gen !== generation) return;
  if (!response.ok) throw new Error(`Recording ${id} not found`);
  const data = await response.json() as { run?: { project_root_dir?: string; synthetic?: boolean }; events: AgentActivityEvent[] };
  if (gen !== generation) return;

  // Frame numbers and track ids only mean something in the recording's own project.
  const root = data.run?.project_root_dir ?? data.events.find((e) => e.project_root_dir)?.project_root_dir;
  const project = useProjectStore.getState().project;
  if (root && root !== project?.root_dir) {
    // The backend has one open project for everyone: switching it would pull a live
    // agent run out of its own project.
    const runs = await fetch('/api/agent/runs').then((r) => r.json()).catch(() => ({ runs: [] }));
    if (gen !== generation) return;
    if ((runs.runs ?? []).some((r: { status?: string }) => r.status === 'running')) {
      throw new Error('An agent run is live on this backend. The replay would switch its project, '
        + 'so it waits: open the replay again once the run has finished.');
    }
    await useProjectStore.getState().openProject(root);
    return;  // the project change starts the replay again
  }

  const events = data.events.filter((e) => e.type !== 'video_job.frame');
  const start = events.findIndex((e) => e.type === 'agent.started');
  timeline = compress(start > 0 ? events.slice(start) : events);
  cursor = 0;
  // Hide only what this run made: labels it created and tracks it created.
  const hiddenLabels = [...new Set(events.filter((e) => e.type === 'label.created')
    .map((e) => e.label).filter((l): l is string => typeof l === 'string'))];
  const recordedTracks = [...new Set(events.filter((e) => e.type === 'track.created')
    .map((e) => e.track_id).filter((t): t is number => typeof t === 'number'))];
  resetAgentScratch();
  useAgentStore.getState().reset(null);
  useAgentStore.setState({
    replay: { id, speed, playing: true, position: 0, duration: timeline.at(-1)?.at ?? 0, finished: false,
      hiddenLabels, recordedTracks, synthetic: Boolean(data.run?.synthetic) },
  });
  loop(gen);
}

export function setReplaySpeed(speed: number) {
  useAgentStore.setState((s) => ({ replay: s.replay ? { ...s.replay, speed } : null }));
}

export function toggleReplay() {
  const replay = useAgentStore.getState().replay;
  if (!replay) return;
  if (replay.finished) { void startReplay(replay.id, replay.speed); return; }
  useAgentStore.setState({ replay: { ...replay, playing: !replay.playing } });
}

export function restartReplay() {
  const replay = useAgentStore.getState().replay;
  if (replay) void startReplay(replay.id, replay.speed);
}

export function stopReplay() {
  generation++;
  cancelAnimationFrame(raf);
  timeline = [];
  cursor = 0;
  resetAgentScratch();
  if (useAgentStore.getState().replay) useAgentStore.setState({ replay: null });
}

export function replayParams(): { id: string; speed: number } | null {
  const params = new URLSearchParams(window.location.search);
  const id = params.get('replay');
  if (!id) return null;
  const speed = Number(params.get('speed') ?? 1);
  return { id, speed: Number.isFinite(speed) && speed > 0 ? speed : 1 };
}
