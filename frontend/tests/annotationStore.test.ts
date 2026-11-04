import { beforeEach, describe, expect, it, vi } from 'vitest';
import { useAnnotationStore } from '../src/stores/annotationStore';
import * as api from '../src/api/client';
import type { Annotation } from '../src/types';

vi.mock('../src/api/client', () => ({ listAnnotations: vi.fn(), listLabels: vi.fn() }));

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => { resolve = r; });
  return { promise, resolve };
}

const annotation = (id: number, imageId = 1) => ({ id, image_id: imageId, label_id: 1 }) as Annotation;

beforeEach(() => {
  vi.resetAllMocks();
  useAnnotationStore.getState().resetForProject();
});

describe('annotation request ownership', () => {
  it('does not restore an old project request after reset, even when IDs repeat', async () => {
    const old = deferred<Annotation[]>();
    vi.mocked(api.listAnnotations).mockReturnValueOnce(old.promise).mockResolvedValue([annotation(20)]);
    const oldLoad = useAnnotationStore.getState().loadAnnotations(1);
    useAnnotationStore.getState().resetForProject();
    await useAnnotationStore.getState().loadAnnotations(1);
    old.resolve([annotation(10)]);
    await oldLoad;
    expect(useAnnotationStore.getState().annotations).toEqual([annotation(20)]);
    await useAnnotationStore.getState().loadAnnotations(1);
    expect(useAnnotationStore.getState().annotations).toEqual([annotation(20)]);
  });

  it('does not repopulate invalidated cache from an old prefetch', async () => {
    const old = deferred<Annotation[]>();
    vi.mocked(api.listAnnotations).mockReturnValueOnce(old.promise).mockResolvedValue([annotation(20)]);
    useAnnotationStore.getState().prefetchAnnotations([1]);
    useAnnotationStore.getState().invalidateAnnotations(1);
    await useAnnotationStore.getState().loadAnnotations(1);
    old.resolve([annotation(10)]);
    await old.promise;
    await useAnnotationStore.getState().loadAnnotations(1);
    expect(useAnnotationStore.getState().annotations).toEqual([annotation(20)]);
  });

  it('ignores a slower load of a previously selected frame without requiring a signal', async () => {
    const old = deferred<Annotation[]>();
    vi.mocked(api.listAnnotations).mockReturnValueOnce(old.promise).mockResolvedValue([annotation(20, 2)]);
    const oldLoad = useAnnotationStore.getState().loadAnnotations(1);
    await useAnnotationStore.getState().loadAnnotations(2);
    old.resolve([annotation(10)]);
    await oldLoad;
    expect(useAnnotationStore.getState().annotations).toEqual([annotation(20, 2)]);
  });

  it('does not share an already aborted request with the next load', async () => {
    const old = deferred<Annotation[]>();
    vi.mocked(api.listAnnotations).mockReturnValueOnce(old.promise).mockResolvedValue([annotation(20)]);
    const controller = new AbortController();
    const oldLoad = useAnnotationStore.getState().loadAnnotations(1, controller.signal);
    controller.abort();
    const newLoad = useAnnotationStore.getState().loadAnnotations(1);
    old.resolve([annotation(10)]);
    await Promise.all([oldLoad, newLoad]);
    expect(useAnnotationStore.getState().annotations).toEqual([annotation(20)]);
  });
});
