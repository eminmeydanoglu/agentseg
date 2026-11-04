import { useCallback, useEffect } from 'react';
import { getOptimizedImageUrl } from '../api/client';
import type { AgentActivityEvent } from '../hooks/useAgentActivity';
import { preloadImage } from '../utils/imageCache';
import { useAnnotationStore } from '../stores/annotationStore';
import { useProjectStore } from '../stores/projectStore';
import { useUIStore } from '../stores/uiStore';
import type { Annotation, Label } from '../types';
import { applyAgentEvent, resetAgentScratch, statusesFromServer } from './agentEvents';
import { agentPropagating, useAgentStore, type TrackView } from './agentStore';
import { replayParams, startReplay, stopReplay } from './replay';

/**
 * Glue between the agent view and the rest of the app:
 * - live activity events -> agent store (and label / track refreshes)
 * - `?replay=<id>` -> play a recorded run instead
 * - catch up on a run that started before this page was opened
 * - Follow: the canvas goes where the agent works until the person navigates away
 */

let lastSeq = 0;
let followTarget: number | null = null;
let projectGen = 0;                         // bumps on project change; async results check it
let hydrating = false;
let buffered: AgentActivityEvent[] = [];    // live events that arrive while catching up

/** Warm frames before the canvas shows them: the next ones while SAM tracks, or one frame (direction 0). */
const RIDE_AHEAD = 10;
function warmAhead(frame: number, direction: number) {
  const { images, project } = useProjectStore.getState();
  const ids: number[] = [];
  // direction 0: just this frame.
  for (let i = direction === 0 ? 0 : 1; i <= (direction === 0 ? 0 : RIDE_AHEAD); i++) {
    const image = images[frame + i * direction];
    if (!image) break;
    ids.push(image.id);
    void preloadImage(getOptimizedImageUrl(image.id, 2048, 85, project?.root_dir)).catch(() => {});
  }
  if (ids.length) useAnnotationStore.getState().prefetchAnnotations(ids);
}

const REFRESH_TRACKS_ON = new Set(['track.finished', 'track.absent', 'issue.reviewed', 'keyframe.removed', 'track.created']);
let refreshTimer = 0;

async function loadServerTracks() {
  if (useAgentStore.getState().replay) return;
  const gen = projectGen;
  try {
    const list = await (await fetch('/api/video/tracks')).json() as { id: number; label_id: number; keyframes: number[] }[];
    if (!Array.isArray(list)) return;
    const loaded: Record<number, TrackView> = {};
    await Promise.all(list.map(async (t) => {
      const data = await (await fetch(`/api/video/tracks/${t.id}/frames`)).json() as { n: number; statuses: (string | null)[] };
      if (!Array.isArray(data?.statuses)) return;
      loaded[t.id] = { id: t.id, labelId: t.label_id, n: data.n, statuses: statusesFromServer(data.statuses),
        keyframes: t.keyframes ?? [], propagating: false };
    }));
    if (gen !== projectGen || useAgentStore.getState().replay) return;
    useAgentStore.setState((s) => {
      const tracks: Record<number, TrackView> = {};
      for (const [id, tv] of Object.entries(loaded)) {
        const prev = s.tracks[Number(id)];
        // Never overwrite a track that is filling in live.
        tracks[Number(id)] = prev?.propagating ? prev : { ...tv, label: prev?.label };
      }
      return { tracks };
    });
  } catch {
    // Older backends have no SAM 3 track endpoints; the timeline just stays hidden.
  }
}

function scheduleTrackRefresh() {
  window.clearTimeout(refreshTimer);
  refreshTimer = window.setTimeout(() => { void loadServerTracks(); }, 400);
}

function applyLive(event: AgentActivityEvent) {
  const seq = typeof event.seq === 'number' ? event.seq : undefined;
  if (seq != null) {
    if (seq <= lastSeq) return;
    lastSeq = seq;
  }
  if (event.type === 'agent.started') useUIStore.getState().setFollowAgent(true);
  applyAgentEvent(event, {
    now: performance.now(),
    speed: 1,
    frameCount: useProjectStore.getState().images.length,
    live: true,
  });
  if (REFRESH_TRACKS_ON.has(event.type)) scheduleTrackRefresh();
}

export function handleLiveAgentEvent(event: AgentActivityEvent) {
  if (useAgentStore.getState().replay) return;
  if (hydrating) { buffered.push(event); return; }
  applyLive(event);
}

async function hydrateLiveRun() {
  const gen = projectGen;
  hydrating = true;
  buffered = [];
  try {
    const run = await (await fetch('/api/agent/runs/latest')).json() as { id?: string; status?: string; finished_at?: number };
    if (!run?.id) return;
    const recent = run.status === 'running' || (run.finished_at != null && Date.now() / 1000 - run.finished_at < 30 * 60);
    if (!recent) return;
    const data = await (await fetch(`/api/agent/runs/${run.id}/events`)).json() as { events?: AgentActivityEvent[] };
    if (gen !== projectGen || useAgentStore.getState().replay) return;
    const frameCount = useProjectStore.getState().images.length;
    for (const event of data.events ?? []) {
      const seq = typeof event.seq === 'number' ? event.seq : 0;
      if (seq && seq <= lastSeq) continue;
      lastSeq = Math.max(lastSeq, seq);
      // End states only: no cursor, no queue, one jump to the agent's frame below.
      applyAgentEvent(event, { now: performance.now(), speed: 1, frameCount, live: true, hydrate: true });
    }
    useAgentStore.setState((s) => ({ beat: null, sheet: null,
      focusSeq: s.run?.status === 'running' && s.focusFrame != null ? s.focusSeq + 1 : s.focusSeq }));
    scheduleTrackRefresh();
  } catch {
    // No agent API (older backend): nothing to catch up on.
  } finally {
    if (gen === projectGen) {
      hydrating = false;
      const pending = buffered;
      buffered = [];
      for (const event of pending) applyLive(event);  // anything already caught up is skipped by seq
    }
  }
}

export function useAgentRuntime() {
  const project = useProjectStore((s) => s.project);
  const imageCount = useProjectStore((s) => s.images.length);
  const projectKey = project ? `${project.root_dir}:${project.id}` : '';

  // Replay or catch up once the project and its frames are known.
  useEffect(() => {
    if (!projectKey) {
      // A replay link on a backend with no project open: the replay opens its own project.
      const replay = replayParams();
      if (replay) void startReplay(replay.id, replay.speed).catch((err) => {
        useUIStore.getState().addToast(String(err.message ?? err), 'error');
      });
      return;
    }
    if (!imageCount) return;
    projectGen++;
    hydrating = false;
    buffered = [];
    window.clearTimeout(refreshTimer);
    resetAgentScratch();
    const replay = replayParams();
    if (replay) {
      useUIStore.getState().setFollowAgent(true);
      void startReplay(replay.id, replay.speed).catch((err) => {
        useUIStore.getState().addToast(String(err.message ?? err), 'error');
      });
      return () => stopReplay();
    }
    lastSeq = 0;
    useAgentStore.getState().reset(null);
    void loadServerTracks();
    void hydrateLiveRun();
    return undefined;
  }, [projectKey, imageCount]);

  // Follow: go to the agent's frame.
  useEffect(() => useAgentStore.subscribe((s, prev) => {
    if (s.focusSeq === prev.focusSeq || s.focusFrame == null) return;
    if (!useUIStore.getState().followAgent) return;
    const { images, currentImageIndex, setCurrentImageIndex } = useProjectStore.getState();
    if (s.focusFrame < 0 || s.focusFrame >= images.length) return;
    followTarget = s.focusFrame;
    if (agentPropagating() && prev.focusFrame != null && prev.focusFrame !== s.focusFrame) {
      warmAhead(s.focusFrame, Math.sign(s.focusFrame - prev.focusFrame));
    }
    if (currentImageIndex !== s.focusFrame) setCurrentImageIndex(s.focusFrame);
  }), []);

  // The cursor is on its way to a frame: load it now, so it is there when the click lands.
  useEffect(() => useAgentStore.subscribe((s, prev) => {
    if (!s.cursor || s.cursor === prev.cursor) return;
    const t = s.cursor.target;
    const frame = t.kind === 'timeline' || t.kind === 'sheet' ? t.frame : undefined;
    if (frame != null) warmAhead(frame, 0);
  }), []);

  // Any other navigation during a run means the person wants to look around.
  useEffect(() => useProjectStore.subscribe((s, prev) => {
    if (s.currentImageIndex === prev.currentImageIndex) return;
    const run = useAgentStore.getState().run;
    if (run?.status !== 'running' || !useUIStore.getState().followAgent) return;
    if (s.currentImageIndex !== followTarget) useUIStore.getState().setFollowAgent(false);
  }), []);

  // Turning Follow back on jumps to where the agent is.
  useEffect(() => useUIStore.subscribe((s, prev) => {
    if (!s.followAgent || prev.followAgent) return;
    const frame = useAgentStore.getState().focusFrame;
    if (frame == null) return;
    followTarget = frame;
    useProjectStore.getState().setCurrentImageIndex(frame);
  }), []);

  // A label the agent added shows up in the label picker, selected, like one a person adds.
  useEffect(() => useAgentStore.subscribe((s, prev) => {
    if (!s.recentLabel || s.recentLabel === prev.recentLabel) return;
    const name = s.recentLabel.name;
    const annotations = useAnnotationStore.getState();
    const select = () => {
      const label = useAnnotationStore.getState().labels.find((l) => l.name === name);
      if (label) useAnnotationStore.getState().selectLabel(label.id);
    };
    if (s.replay) select();
    else void annotations.loadLabels().then(select).catch(() => {});
  }), []);
}

/** During a replay, hide what the recording has not created yet. */
export function useReplayVisibility() {
  const replaying = useAgentStore((s) => s.replay != null);
  const tracks = useAgentStore((s) => s.tracks);
  const hiddenLabels = useAgentStore((s) => s.replay?.hiddenLabels);
  const recordedTracks = useAgentStore((s) => s.replay?.recordedTracks);

  const annotationVisible = useCallback((ann: Annotation, frame: number) => {
    if (!replaying || ann.track_id == null || !recordedTracks?.includes(ann.track_id)) return true;
    const status = tracks[ann.track_id]?.statuses[frame];
    return status === 'keyframe' || status === 'ok' || status === 'review';
  }, [replaying, tracks, recordedTracks]);

  const labelVisible = useCallback((label: Label) => !hiddenLabels?.includes(label.name), [hiddenLabels]);

  return { replaying, annotationVisible, labelVisible };
}
