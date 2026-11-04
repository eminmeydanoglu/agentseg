import { useAgentStore } from '../src/agent/agentStore';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import * as api from '../src/api/client';
import { useProjectStore } from '../src/stores/projectStore';
import { useAnnotationStore } from '../src/stores/annotationStore';
import type { Annotation, ImageRecord, Project } from '../src/types';

vi.mock('../src/api/client', () => ({
  getCurrentProject: vi.fn(), openProject: vi.fn(), listImages: vi.fn(), getSetting: vi.fn(),
}));

const project = (root: string): Project => ({ id: 1, name: root, root_dir: root, image_count: 1 });
const image = (path: string): ImageRecord => ({ id: 1, project_id: 1, path, width: 480, height: 268, order_index: 0 });

beforeEach(() => {
  vi.resetAllMocks();
  useProjectStore.setState({ project: null, images: [], currentImageIndex: 0, isLoading: false });
  useAnnotationStore.getState().resetForProject();
  vi.mocked(api.getSetting).mockResolvedValue('0');
});

describe('backend-authoritative project sync', () => {
  it('adopts a CLI project with the same numeric ID without reopening its DB', async () => {
    useProjectStore.setState({ project: project('/old'), images: [image('/old/0.jpg')] });
    useAnnotationStore.setState({ annotations: [{ id: 1, image_id: 1 }] as Annotation[] });
    useAgentStore.setState({ beat: { id: 1, frame: 0, bbox: [1, 2, 3, 4], pos: [], neg: [], t0: 0, speed: 1 } });
    vi.mocked(api.getCurrentProject).mockResolvedValue(project('/new'));
    vi.mocked(api.listImages).mockResolvedValue([image('/new/0.jpg')]);
    expect(await useProjectStore.getState().syncCurrentProject()).toBe(true);
    expect(useProjectStore.getState().images[0].path).toBe('/new/0.jpg');
    expect(useProjectStore.getState().project?.root_dir).toBe('/new');
    expect(useAnnotationStore.getState().annotations).toEqual([]);
    expect(useAgentStore.getState().beat).toBeNull();
    expect(api.openProject).not.toHaveBeenCalled();
  });

  it('does not reset or refetch an unchanged project on every poll', async () => {
    useProjectStore.setState({ project: project('/same'), images: [image('/same/0.jpg')] });
    vi.mocked(api.getCurrentProject).mockResolvedValue(project('/same'));
    expect(await useProjectStore.getState().syncCurrentProject()).toBe(false);
    expect(api.listImages).not.toHaveBeenCalled();
  });

  it('attaches to an existing backend project on startup, without POST /open', async () => {
    vi.mocked(api.getCurrentProject).mockResolvedValue(project('/running'));
    vi.mocked(api.listImages).mockResolvedValue([image('/running/0.jpg')]);
    expect(await useProjectStore.getState().tryOpenLastProject()).toBe(true);
    expect(api.openProject).not.toHaveBeenCalled();
    expect(useProjectStore.getState().project?.root_dir).toBe('/running');
  });

  it('clears the editor when the CLI closes its project', async () => {
    useProjectStore.setState({ project: project('/old'), images: [image('/old/0.jpg')] });
    vi.mocked(api.getCurrentProject).mockResolvedValue(null);
    await useProjectStore.getState().syncCurrentProject();
    expect(useProjectStore.getState().project).toBeNull();
    expect(useProjectStore.getState().images).toEqual([]);
  });
});
