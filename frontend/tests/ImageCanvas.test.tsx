import React from 'react';
import { act, cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ImageCanvas } from '../src/components/Canvas/ImageCanvas';
import { useProjectStore } from '../src/stores/projectStore';
import { useAnnotationStore } from '../src/stores/annotationStore';
import { getCachedImage, preloadImage } from '../src/utils/imageCache';
import { getOptimizedImageUrl } from '../src/api/client';
import type { ImageRecord, Project } from '../src/types';

vi.mock('../src/utils/imageCache', () => ({ getCachedImage: vi.fn(), preloadImage: vi.fn() }));
vi.mock('react-konva', () => ({
  Stage: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  Layer: ({ children, visible = true }: React.PropsWithChildren<{ visible?: boolean }>) => <div hidden={!visible}>{children}</div>,
  Image: ({ image }: { image: HTMLImageElement }) => <img data-testid="bitmap" src={image.src} />,
  Rect: () => <span data-testid="bbox" />,
  Circle: () => null, Line: () => null, Transformer: () => null,
}));

const project = (root: string): Project => ({ id: 1, name: root, root_dir: root, image_count: 1 });
const image = (id = 1): ImageRecord => ({ id, project_id: 1, path: `/images/${id}.jpg`, width: 480, height: 268, order_index: id-1 });
const bitmap = (src: string) => Object.assign(new Image(), { src, width: 480, height: 268 });

beforeEach(() => {
  vi.resetAllMocks();
  vi.useFakeTimers();
  vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} });
  useAnnotationStore.getState().resetForProject();
  useProjectStore.setState({ project: project('/old'), images: [image()], currentImageIndex: 0 });
});
afterEach(() => { cleanup(); vi.useRealTimers(); vi.unstubAllGlobals(); });

describe('bitmap ownership', () => {
  it('uses the DB path, not its reusable project ID, as the image cache key', () => {
    const old = bitmap('/old.jpg');
    const next = bitmap('/new.jpg');
    vi.mocked(getCachedImage).mockImplementation((src) => src.includes(encodeURIComponent('/new')) ? next : old);
    render(<ImageCanvas />);
    act(() => { useProjectStore.setState({ project: project('/new'), images: [image()] }); });
    expect(screen.getByTestId('bitmap').getAttribute('src')).toContain('/new.jpg');
    expect(getCachedImage).toHaveBeenLastCalledWith(getOptimizedImageUrl(1, 2048, 85, '/new'));
  });

  it('hides the previous photo and overlays while the next bitmap is pending', () => {
    const first = getOptimizedImageUrl(1, 2048, 85, '/old');
    vi.mocked(getCachedImage).mockImplementation((src) => (src === first ? bitmap('/old.jpg') : null));
    vi.mocked(preloadImage).mockReturnValue(new Promise(() => {}));
    render(<ImageCanvas />);
    expect(screen.getByTestId('bitmap')).toBeTruthy();
    act(() => { useProjectStore.setState({ images: [image(2)] }); });
    expect(screen.queryByTestId('bitmap')).toBeNull();
  });

  it('escapes project paths in cache URLs', () => {
    const url = getOptimizedImageUrl(1, 2048, 85, '/data/project & v=wrong');
    expect(new URL(url, 'http://localhost').searchParams.get('project_root_dir')).toBe('/data/project & v=wrong');
  });
});
