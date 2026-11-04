import { create } from 'zustand';
import type { Project, ImageRecord } from '../types';
import * as api from '../api/client';
import { useUIStore } from './uiStore';
import { useAgentStore } from '../agent/agentStore';
import { useAnnotationStore } from './annotationStore';

interface ProjectState {
  // State
  project: Project | null;
  images: ImageRecord[];
  currentImageIndex: number;
  isLoading: boolean;
  error: string | null;

  // Computed
  currentImage: ImageRecord | null;
  imageCount: number;

  // Actions
  createProject: (
    projectDir: string,
    imageDir: string,
    name: string
  ) => Promise<Project>;
  openProject: (projectDir: string) => Promise<void>;
  closeProject: () => Promise<void>;
  loadImages: () => Promise<void>;
  setCurrentImageIndex: (index: number) => void;
  nextImage: () => void;
  prevImage: () => void;
  tryOpenLastProject: () => Promise<boolean>;
  syncCurrentProject: () => Promise<boolean>;
  resyncImages: () => Promise<{ added: number; removed: number }>;
}

export const useProjectStore = create<ProjectState>((set, get) => {
  let projectVersion = 0;
  let syncRequest: Promise<boolean> | null = null;

  const activateProject = async (project: Project | null, version: number) => {
    if (version !== projectVersion) return;
    useAnnotationStore.getState().resetForProject();
    // The agent view belongs to the old project's frames; the new one catches up on its own.
    useAgentStore.setState({ beat: null, sheet: null, focusFrame: null, tracks: {} });
    set({ project, images: [], currentImageIndex: 0, error: null, isLoading: Boolean(project) });
    if (!project) {
      useUIStore.getState().setLastProjectPath(null);
      return;
    }
    try {
      const [images, savedIndex] = await Promise.all([
        api.listImages(),
        api.getSetting('last_image_index').catch(() => null),
      ]);
      if (version !== projectVersion) return;
      const index = parseInt(savedIndex ?? '0', 10) || 0;
      set({ images, currentImageIndex: index >= 0 && index < images.length ? index : 0 });
      useUIStore.getState().setLastProjectPath(project.root_dir);
    } finally {
      if (version === projectVersion) set({ isLoading: false });
    }
  };

  return ({
  // Initial state
  project: null,
  images: [],
  currentImageIndex: 0,
  isLoading: false,
  error: null,

  // Computed getters
  get currentImage() {
    const { images, currentImageIndex } = get();
    return images[currentImageIndex] ?? null;
  },

  get imageCount() {
    return get().images.length;
  },

  // Actions
  createProject: async (projectDir, imageDir, name) => {
    const version = ++projectVersion;
    set({ isLoading: true, error: null });
    try {
      const project = await api.createProject(projectDir, imageDir, name);
      await activateProject(project, version);
      return project;
    } catch (err) {
      if (version === projectVersion) set({ error: err instanceof Error ? err.message : 'Failed to create project' });
      throw err;
    } finally {
      if (version === projectVersion) set({ isLoading: false });
    }
  },

  openProject: async (projectDir) => {
    const version = ++projectVersion;
    set({ isLoading: true, error: null });
    try {
      const project = await api.openProject(projectDir);
      await activateProject(project, version);
    } catch (err) {
      if (version === projectVersion) set({ error: err instanceof Error ? err.message : 'Failed to open project' });
      throw err;
    } finally {
      if (version === projectVersion) set({ isLoading: false });
    }
  },

  closeProject: async () => {
    const version = ++projectVersion;
    try {
      await api.closeProject();
    } finally {
      await activateProject(null, version);
    }
  },

  loadImages: async () => {
    const version = projectVersion;
    try {
      const images = await api.listImages();
      if (version === projectVersion) set({ images });
    } catch (err) {
      if (version === projectVersion) set({ error: err instanceof Error ? err.message : 'Failed to load images' });
    }
  },

  setCurrentImageIndex: (index) => {
    const { images } = get();
    if (index >= 0 && index < images.length) {
      set({ currentImageIndex: index });
    }
  },

  nextImage: () => {
    const { currentImageIndex, images } = get();
    if (currentImageIndex < images.length - 1) {
      const newIndex = currentImageIndex + 1;
      set({ currentImageIndex: newIndex });
    }
  },

  prevImage: () => {
    const { currentImageIndex } = get();
    if (currentImageIndex > 0) {
      const newIndex = currentImageIndex - 1;
      set({ currentImageIndex: newIndex });
    }
  },

  // Try to open last project from localStorage
  tryOpenLastProject: async () => {
    // The backend's current project is authoritative. This matters when a
    // project was created by the CLI while the browser still has an older
    // lastProjectPath persisted locally.
    try {
      await get().syncCurrentProject();
      if (get().project) return true;
    } catch {
      // Fall through to the persisted path when the backend has no project.
    }
    const lastPath = useUIStore.getState().lastProjectPath;
    if (!lastPath) return false;
    
    try {
      await get().openProject(lastPath);
      return true;
    } catch {
      // Clear invalid last project path
      useUIStore.getState().setLastProjectPath(null);
      return false;
    }
  },

  // Adopt CLI changes without POST /open: reopening would tear down SAM jobs.
  // root_dir is essential here: every independent project DB can have id=1.
  syncCurrentProject: () => {
    if (syncRequest) return syncRequest;
    if (get().isLoading) return Promise.resolve(false);
    const version = projectVersion;
    syncRequest = (async () => {
      const current = await api.getCurrentProject();
      if (version !== projectVersion) return false;
      if (current?.root_dir === get().project?.root_dir && current?.id === get().project?.id &&
        (!current || current.image_count === get().images.length)) return false;
      await activateProject(current, ++projectVersion);
      return true;
    })().finally(() => { syncRequest = null; });
    return syncRequest;
  },

  resyncImages: async () => {
    const { project } = get();
    if (!project) throw new Error('No project loaded');

    try {
      const result = await api.resyncImages();
      useAnnotationStore.getState().invalidateAnnotations();

      // Reload images to get updated list
      await get().loadImages();

      // Reset index if current is out of bounds
      const { images, currentImageIndex } = get();
      if (currentImageIndex >= images.length) {
        set({ currentImageIndex: Math.max(0, images.length - 1) });
      }

      return { added: result.added, removed: result.removed };
    } catch (err) {
      set({ error: err instanceof Error ? err.message : 'Failed to resync images' });
      throw err;
    }
  },
  });
});
