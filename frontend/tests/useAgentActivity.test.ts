import { act, cleanup, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { useAgentActivity } from '../src/hooks/useAgentActivity';

class FakeEventSource {
  static instances: FakeEventSource[] = [];
  url: string;
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onmessage: ((event: MessageEvent<string>) => void) | null = null;
  close = vi.fn();
  addEventListener = vi.fn();
  removeEventListener = vi.fn();
  constructor(url: string) { this.url = url; FakeEventSource.instances.push(this); }
}

beforeEach(() => {
  FakeEventSource.instances = [];
  vi.stubGlobal('EventSource', FakeEventSource);
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({ events: [{ seq: 50, type: 'video_job.frame', frame: 10 }] }) }));
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it('does not replay historical frames as live work and reconnects on DB path changes', async () => {
  const follow = vi.fn();
  const { result, rerender } = renderHook(({ root }) => useAgentActivity(1, follow, root), { initialProps: { root: '/old' } });
  await waitFor(() => expect(FakeEventSource.instances).toHaveLength(1));
  expect(FakeEventSource.instances[0].url).toContain('after=50');
  expect(follow).not.toHaveBeenCalled();
  expect(result.current.events).toHaveLength(1);

  vi.mocked(fetch).mockResolvedValue({ ok: true, json: async () => ({ events: [] }) } as Response);
  rerender({ root: '/new' });
  expect(result.current.events).toHaveLength(0);
  await waitFor(() => expect(FakeEventSource.instances).toHaveLength(2));
  expect(FakeEventSource.instances[0].close).toHaveBeenCalled();
  expect(FakeEventSource.instances[1].url).toContain('after=0');

  act(() => FakeEventSource.instances[1].onmessage?.(new MessageEvent('message', { data: JSON.stringify({ seq: 1, type: 'video_job.frame', project_root_dir: '/old' }) })));
  expect(follow).not.toHaveBeenCalled();
  act(() => FakeEventSource.instances[1].onmessage?.(new MessageEvent('message', { data: JSON.stringify({ seq: 2, type: 'video_job.frame', project_root_dir: '/new' }) })));
  expect(follow).toHaveBeenCalledOnce();
});
