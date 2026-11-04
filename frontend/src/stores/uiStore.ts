import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { InteractionMode } from '../types';
import { toast } from 'sonner';

export type ReviewFilter = 'all' | 'pending' | 'approved' | 'rejected';

const STORAGE_KEY = 'agentseg-ui-settings';

interface UIState {
  // Persisted settings
  lastProjectPath: string | null;
  maskOpacity: number;
  followAgent: boolean; // Keep the canvas on the frame the agent is working on

  // Session state (not persisted)
  mode: InteractionMode;
  reviewModeEnabled: boolean;
  reviewFilter: ReviewFilter;

  // Modal states
  showCreateProjectModal: boolean;
  showOpenProjectModal: boolean;
  showExportModal: boolean;
  showSettingsModal: boolean;

  statusMessage: string;

  setMode: (mode: InteractionMode) => void;
  setReviewMode: (enabled: boolean) => void;
  setReviewFilter: (filter: ReviewFilter) => void;
  setMaskOpacity: (opacity: number) => void;
  setLastProjectPath: (path: string | null) => void;
  setFollowAgent: (enabled: boolean) => void;
  setShowCreateProjectModal: (show: boolean) => void;
  setShowOpenProjectModal: (show: boolean) => void;
  setShowExportModal: (show: boolean) => void;
  setShowSettingsModal: (show: boolean) => void;
  setStatusMessage: (message: string) => void;
  addToast: (message: string, type?: 'success' | 'error' | 'warning' | 'info', duration?: number) => void;
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      lastProjectPath: null,
      maskOpacity: 0.5,
      followAgent: true,

      mode: 'view',
      reviewModeEnabled: false,
      reviewFilter: 'all',
      showCreateProjectModal: false,
      showOpenProjectModal: false,
      showExportModal: false,
      showSettingsModal: false,
      statusMessage: 'Ready',

      setMode: (mode) => set({ mode }),
      setReviewMode: (enabled) => set({ reviewModeEnabled: enabled, reviewFilter: enabled ? 'pending' : 'all' }),
      setReviewFilter: (filter) => set({ reviewFilter: filter }),
      setMaskOpacity: (opacity) => set({ maskOpacity: Math.max(0, Math.min(1, opacity)) }),
      setLastProjectPath: (path) => set({ lastProjectPath: path }),
      setFollowAgent: (enabled) => set({ followAgent: enabled }),
      setShowCreateProjectModal: (show) => set({ showCreateProjectModal: show }),
      setShowOpenProjectModal: (show) => set({ showOpenProjectModal: show }),
      setShowExportModal: (show) => set({ showExportModal: show }),
      setShowSettingsModal: (show) => set({ showSettingsModal: show }),
      setStatusMessage: (message) => set({ statusMessage: message }),

      addToast: (message, type = 'info', duration) => {
        const options = duration ? { duration } : undefined;
        if (type === 'success') toast.success(message, options);
        else if (type === 'error') toast.error(message, options);
        else if (type === 'warning') toast.warning(message, options);
        else toast.info(message, options);
      },
    }),
    {
      name: STORAGE_KEY,
      partialize: (state) => ({
        lastProjectPath: state.lastProjectPath,
        maskOpacity: state.maskOpacity,
        followAgent: state.followAgent,
      }),
    }
  )
);
