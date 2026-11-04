import { useAgentStore, type FeedItem } from './agentStore';

/** State the agent views read. */

/** True when the agent view should be on screen at all. */
export function useAgentVisible(): boolean {
  return useAgentStore((s) => s.run != null && !(s.dismissed && s.run.status !== 'running'));
}

/** The sentence the views show: the newest thing the agent said (the report's first paragraph at the end). */
export function currentLine(feed: FeedItem[]): FeedItem | undefined {
  for (let i = feed.length - 1; i >= 0; i--) {
    const item = feed[i];
    if (item.kind === 'thought' || item.kind === 'report' || item.kind === 'error') return item;
  }
  return undefined;
}

/** Everything a live view needs, read once. */
export function useAgentNow() {
  const run = useAgentStore((s) => s.run);
  const feed = useAgentStore((s) => s.feed);
  const action = useAgentStore((s) => s.action);
  const activity = useAgentStore((s) => s.activity);
  const phase = useAgentStore((s) => s.phase);
  const running = run?.status === 'running';
  const line = currentLine(feed);
  // Between calls the model is thinking: its last sentence has been acted on.
  const thinking = running && activity.kind === 'thinking' && action?.state !== 'running';
  const text = line?.kind === 'report' ? line.text.split(/\n\s*\n/)[0] : line?.text;
  const progress = activity.kind === 'computing' ? activity.progress : undefined;
  return { run, feed, action, activity, phase, running, line, thinking, text, progress };
}

