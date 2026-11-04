import { useCallback, useEffect, useRef, useState } from 'react';
import { useUIStore } from '../stores/uiStore';

export interface AgentActivityEvent {
  seq?: number;
  id?: string | number;
  run_id?: string;
  project_id?: number;
  project_root_dir?: string;
  source?: string;
  type: string;
  action?: string;
  status?: string;
  message?: string;
  detail?: string;
  frame?: number;
  image_index?: number;
  image_id?: number;
  track_id?: number;
  bbox?: [number, number, number, number] | null;
  render_url?: string | null;
  visible?: boolean;
  progress?: { done?: number; total?: number; current?: number };
  counts?: Record<string, number>;
  artifact?: { url?: string; kind?: string; label?: string } | null;
  error?: string | null;
  ts?: number;
  [key: string]: unknown;
}

const MAX_EVENTS = 120;

function eventKey(event: AgentActivityEvent): string {
  return String(event.seq ?? event.id ?? `${event.ts ?? 0}:${event.type}:${event.message ?? ''}`);
}

function parseEvent(value: unknown): AgentActivityEvent | null {
  if (!value || typeof value !== 'object') return null;
  const event = value as Partial<AgentActivityEvent>;
  if (typeof event.type !== 'string') return null;
  return event as AgentActivityEvent;
}

function mergeEvents(current: AgentActivityEvent[], incoming: AgentActivityEvent[]): AgentActivityEvent[] {
  const map = new Map(current.map((event) => [eventKey(event), event]));
  for (const event of incoming) map.set(eventKey(event), event);
  return Array.from(map.values())
    .sort((a, b) => (a.seq ?? a.ts ?? 0) - (b.seq ?? b.ts ?? 0))
    .slice(-MAX_EVENTS);
}

export function useAgentActivity(projectId: number | undefined, onFollowFrame?: (event: AgentActivityEvent) => void, projectRootDir?: string) {
  const projectKey = `${projectRootDir ?? ''}:${projectId ?? ''}`;
  const [history, setHistory] = useState<{ key: string; events: AgentActivityEvent[] }>({ key: projectKey, events: [] });
  const events = history.key === projectKey ? history.events : [];
  const [connected, setConnected] = useState(false);
  const followAgent = useUIStore((state) => state.followAgent);
  const setFollowAgent = useUIStore((state) => state.setFollowAgent);
  const lastSeq = useRef(0);
  const onFollowFrameRef = useRef(onFollowFrame);

  useEffect(() => {
    onFollowFrameRef.current = onFollowFrame;
  }, [onFollowFrame]);

  useEffect(() => {
    lastSeq.current = 0;
    if (!projectId) {
      return;
    }

    let cancelled = false;
    const loadHistory = async () => {
      try {
        const response = await fetch(`/api/activity/events?project_id=${projectId}&limit=${MAX_EVENTS}`);
        if (!response.ok) return;
        const payload: unknown = await response.json();
        const list = Array.isArray(payload)
          ? payload
          : (payload && typeof payload === 'object' && Array.isArray((payload as { events?: unknown[] }).events)
            ? (payload as { events: unknown[] }).events
            : []);
        const parsed = list.map(parseEvent).filter((event): event is AgentActivityEvent => Boolean(event));
        if (!cancelled) {
          setHistory({ key: projectKey, events: parsed });
          lastSeq.current = parsed.reduce((max, event) => Math.max(max, event.seq ?? 0), lastSeq.current);
        }
      } catch {
        // The activity view remains useful when an older backend has no activity endpoint.
      }
    };

    let stream: EventSource | null = null;
    const handleMessage = (message: MessageEvent<string>) => {
      try {
        const event = parseEvent(JSON.parse(message.data));
        if (!event || cancelled) return;
        if (event.project_root_dir && event.project_root_dir !== projectRootDir) return;
        lastSeq.current = Math.max(lastSeq.current, event.seq ?? 0);
        setHistory((current) => ({ key: projectKey, events: mergeEvents(current.key === projectKey ? current.events : [], [event]) }));
        onFollowFrameRef.current?.(event);
      } catch {
        // Ignore malformed keepalive or extension events.
      }
    };
    // The backend names its SSE event `activity`; listen to both named and
    // unnamed messages so this client remains compatible with simple proxies.
    // Open after loading history so old frame events aren't replayed as live
    // work, moving the canvas or resurrecting a stale preview on every mount.
    void loadHistory().then(() => {
      if (cancelled) return;
      stream = new EventSource(`/api/activity/stream?project_id=${projectId}&after=${lastSeq.current}`);
      stream.onopen = () => { if (!cancelled) setConnected(true); };
      stream.onerror = () => { if (!cancelled) setConnected(false); };
      stream.addEventListener('activity', handleMessage);
      stream.onmessage = handleMessage;
    });

    return () => {
      cancelled = true;
      stream?.removeEventListener('activity', handleMessage);
      stream?.close();
      setConnected(false);
    };
  }, [projectId, projectRootDir, projectKey]);

  const clear = useCallback(() => setHistory({ key: projectKey, events: [] }), [projectKey]);
  return { events, connected, followAgent, setFollowAgent, clear };
}
