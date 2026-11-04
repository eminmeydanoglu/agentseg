import { beforeEach, expect, it, vi } from 'vitest';
import { refreshAgentAnnotations } from '../src/utils/agentAnnotations';
import { useAnnotationStore } from '../src/stores/annotationStore';
import { useProjectStore } from '../src/stores/projectStore';
import { useUIStore } from '../src/stores/uiStore';
import type { ImageRecord } from '../src/types';

beforeEach(() => {
  useProjectStore.setState({ images: [{ id: 10, order_index: 0 }, { id: 11, order_index: 1 }] as ImageRecord[], currentImageIndex: 0 });
  useUIStore.getState().setFollowAgent(false);
});

it('refreshes visible persisted labels after CLI writes, even with Follow off', () => {
  const invalidate = vi.spyOn(useAnnotationStore.getState(), 'invalidateAnnotations').mockImplementation(() => {});
  const load = vi.spyOn(useAnnotationStore.getState(), 'loadAnnotations').mockResolvedValue();
  refreshAgentAnnotations({ type: 'video_job.frame', frame: 0 });
  expect(invalidate).toHaveBeenCalledWith(10);
  expect(load).toHaveBeenCalledWith(10);
});

it('invalidates off-screen prefetches without replacing the active frame', () => {
  const invalidate = vi.spyOn(useAnnotationStore.getState(), 'invalidateAnnotations').mockImplementation(() => {});
  const load = vi.spyOn(useAnnotationStore.getState(), 'loadAnnotations').mockResolvedValue();
  refreshAgentAnnotations({ type: 'video_job.frame', frame: 1 });
  expect(invalidate).toHaveBeenCalledWith(11);
  expect(load).not.toHaveBeenCalled();
});

it('invalidates and reloads the active frame after a completed propagation', () => {
  const invalidate = vi.spyOn(useAnnotationStore.getState(), 'invalidateAnnotations').mockImplementation(() => {});
  const load = vi.spyOn(useAnnotationStore.getState(), 'loadAnnotations').mockResolvedValue();
  refreshAgentAnnotations({ type: 'video_job.finished' });
  expect(invalidate).toHaveBeenCalledWith();
  expect(load).toHaveBeenCalledWith(10);
});
