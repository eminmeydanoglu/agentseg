import { useEffect, useCallback } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Header } from './components/Layout/Header';
import { Sidebar } from './components/Layout/Sidebar';
import { ImageCanvas } from './components/Canvas/ImageCanvas';
import { CreateProjectModal } from './components/Modals/CreateProjectModal';
import { OpenProjectModal } from './components/Modals/OpenProjectModal';
import { ExportModal } from './components/Modals/ExportModal';
import { SettingsModal } from './components/Modals/SettingsModal';
import { Toaster } from "@/components/ui/sonner";
import { useProjectStore } from './stores/projectStore';
import { useAnnotationStore } from './stores/annotationStore';
import { useUIStore } from './stores/uiStore';
import { preloadImage } from './utils/imageCache';
import { useAgentActivity, type AgentActivityEvent } from './hooks/useAgentActivity';
import { refreshAgentAnnotations } from './utils/agentAnnotations';
import { handleLiveAgentEvent, useAgentRuntime } from './agent/useAgentRuntime';
import { AgentCursor } from './agent/AgentCursor';
import { agentPropagating } from './agent/agentStore';
import { SheetOverlay } from './agent/SheetOverlay';
import { ZoomOverlay } from './agent/ZoomOverlay';
import { TrackTimeline } from './agent/TrackTimeline';
import { AgentCaption } from './agent/AgentCaption';
import { RunLog } from './agent/RunLog';
import * as api from './api/client';

const queryClient = new QueryClient();
const PREFETCH_LOOKAHEAD_IMAGES = 10;

const NAVIGATION_LOAD_DEBOUNCE_MS = 90;
const LAST_IMAGE_INDEX_SAVE_DEBOUNCE_MS = 180;

function AppContent() {
  const { images, currentImageIndex, project, tryOpenLastProject } = useProjectStore();
  const {
    loadAnnotations,
    loadLabels,
    clearAnnotations,
  } = useAnnotationStore();
  const { setStatusMessage } = useUIStore();

  const currentImage = images[currentImageIndex];
  const currentImageId = currentImage?.id;
  const projectRootDir = project?.root_dir;

  // Activity events drive the agent view (status bar, canvas cursor, feed,
  // timeline). CLI writes also invalidate cached annotations and labels.
  const handleAgentEvent = useCallback((event: AgentActivityEvent) => {
    const project = useProjectStore.getState().project;
    if (event.project_root_dir && event.project_root_dir !== project?.root_dir) return;
    refreshAgentAnnotations(event);
    handleLiveAgentEvent(event);
  }, []);

  useAgentRuntime();
  useAgentActivity(project?.id, handleAgentEvent, project?.root_dir);

  // The CLI can switch the backend's active DB while this tab stays open.
  // Read and adopt that state; never reopen the DB just to refresh the browser.
  useEffect(() => {
    const sync = () => { void useProjectStore.getState().syncCurrentProject().catch(() => {}); };
    const timer = window.setInterval(sync, 1000);
    window.addEventListener('focus', sync);
    return () => {
      window.clearInterval(timer);
      window.removeEventListener('focus', sync);
    };
  }, []);

  // On app load: try to reconnect to the last project
  useEffect(() => {
    const init = async () => {
      setStatusMessage('Initializing...');

      const opened = await tryOpenLastProject();
      if (opened) {
        setStatusMessage('Reopened last project');
      } else {
        setStatusMessage('Ready - Open or create a project');
      }
    };

    init();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Mount-only: reads fresh state from stores

  // Load annotations when image changes
  useEffect(() => {
    if (!currentImage) {
      clearAnnotations();
      return;
    }

    // While the agent's track fills in, the canvas steps through frames fast:
    // keep the last masks up until the next ones are there, instead of blinking.
    const riding = agentPropagating();
    if (!riding) clearAnnotations();

    const abortController = new AbortController();
    const loadDelay = riding ? 0 : NAVIGATION_LOAD_DEBOUNCE_MS;

    const timeoutId = window.setTimeout(() => {
      loadAnnotations(currentImage.id, abortController.signal)
        .catch((err) => {
          if (abortController.signal.aborted) return;
          console.error('Failed to load annotations:', err);
        });
    }, loadDelay);

    return () => {
      abortController.abort();
      window.clearTimeout(timeoutId);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentImage?.id, project?.root_dir]);

  // Warm browser/API caches for the next few frames so navigation feels instant.
  useEffect(() => {
    if (!project || images.length === 0) return;

    const upcomingImages = images.slice(
      currentImageIndex + 1,
      currentImageIndex + 1 + PREFETCH_LOOKAHEAD_IMAGES
    );
    if (upcomingImages.length === 0) return;

    useAnnotationStore.getState().prefetchAnnotations(upcomingImages.map((img) => img.id));

    for (const img of upcomingImages) {
      const src = api.getOptimizedImageUrl(img.id, 2048, 85, project.root_dir);
      void preloadImage(src).catch(() => {
        // Prefetch is best-effort; the active image loader will surface real errors.
      });
    }
  }, [project, images, currentImageIndex]);

  // Persist last visited image index after navigation settles.
  useEffect(() => {
    if (!projectRootDir || currentImageId == null) return;

    const timeoutId = window.setTimeout(() => {
      api.setSetting('last_image_index', String(currentImageIndex)).catch(() => {});
    }, LAST_IMAGE_INDEX_SAVE_DEBOUNCE_MS);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [projectRootDir, currentImageId, currentImageIndex]);

  // Load labels when project changes
  useEffect(() => {
    if (project) {
      loadLabels();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project?.root_dir]);

  // Handle segmentation
  const handleSegment = useCallback(async () => {
    const { selectedAnnotationId: selectedAnn, annotations, refinePoints: points } = useAnnotationStore.getState();
    const { images: currentImages, currentImageIndex: currentIdx } = useProjectStore.getState();
    const currentImg = currentImages[currentIdx];

    if (!selectedAnn || !currentImg) return;

    const annotation = annotations.find((a) => a.id === selectedAnn);
    if (!annotation?.bbox) return;

    useUIStore.getState().setStatusMessage('Running segmentation...');
    try {
      const posPoints = points
        .filter((p) => p.type === 'positive')
        .map((p) => [p.x, p.y] as [number, number]);
      const negPoints = points
        .filter((p) => p.type === 'negative')
        .map((p) => [p.x, p.y] as [number, number]);

      const result = await api.segment(
        currentImg.id,
        annotation.bbox,
        posPoints.length > 0 ? posPoints : undefined,
        negPoints.length > 0 ? negPoints : undefined
      );

      // Update annotation with mask and polygon
      await useAnnotationStore.getState().updateAnnotation(selectedAnn, {
        bbox: result.bbox,
        mask_rle: result.mask_rle,
        polygon: result.polygon,
      });

      // Don't clear points after refinement - let them accumulate for iterative refinement
      // User can press Escape to clear manually, or points clear when selecting different annotation
      useUIStore.getState().addToast(`Segmentation complete (score: ${(result.score * 100).toFixed(0)}%)`, 'success');
      useUIStore.getState().setMode('refine');
    } catch (err) {
      const detail = (err as { response?: { data?: { detail?: string } } }).response?.data?.detail;
      useUIStore.getState().addToast(detail ? `Segmentation failed: ${detail}` : 'Segmentation failed', 'error');
      console.error('Segmentation error:', err);
    }
  }, []); // No dependencies - reads fresh state from stores

  // Approve selected annotation
  const handleApprove = useCallback(async () => {
    const selectedAnn = useAnnotationStore.getState().selectedAnnotationId;
    if (!selectedAnn) return;
    await useAnnotationStore.getState().updateAnnotation(selectedAnn, {
      status: 'approved',
    });
    useUIStore.getState().addToast('Annotation approved', 'success', 1500);
  }, []);

  // Reject selected annotation
  const handleReject = useCallback(async () => {
    const selectedAnn = useAnnotationStore.getState().selectedAnnotationId;
    if (!selectedAnn) return;
    await useAnnotationStore.getState().updateAnnotation(selectedAnn, {
      status: 'rejected',
    });
    useUIStore.getState().addToast('Annotation rejected', 'info', 1500);
  }, []);

  // Jump to next pending image
  const handleNextPending = useCallback(async () => {
    try {
      const result = await api.getImagesWithStatus('pending');
      if (result.image_indices.length === 0) {
        useUIStore.getState().addToast('No pending annotations found', 'info');
        return;
      }

      // Find next pending image after current index
      const currentIdx = useProjectStore.getState().currentImageIndex;
      const nextIdx = result.image_indices.find(idx => idx > currentIdx);
      if (nextIdx !== undefined) {
        useProjectStore.getState().setCurrentImageIndex(nextIdx);
        useUIStore.getState().addToast(`Jumped to image ${nextIdx + 1} (pending)`, 'success', 2000);
      } else {
        // Wrap around to first pending
        useProjectStore.getState().setCurrentImageIndex(result.image_indices[0]);
        useUIStore.getState().addToast(`Wrapped to image ${result.image_indices[0] + 1} (pending)`, 'success', 2000);
      }
    } catch (err) {
      console.error('Failed to get pending images:', err);
    }
  }, []);

  // Jump to next unlabeled image (no annotations at all)
  const handleNextUnlabeled = useCallback(async () => {
    try {
      const { project } = useProjectStore.getState();
      if (!project) {
        useUIStore.getState().addToast('No project open', 'warning');
        return;
      }

      const result = await api.findImagesMissingAnnotations(project.id);
      if (result.image_indices.length === 0) {
        useUIStore.getState().addToast('All images have annotations!', 'success');
        return;
      }

      // Find next unlabeled image after current index
      const currentIdx = useProjectStore.getState().currentImageIndex;
      const nextIdx = result.image_indices.find(idx => idx > currentIdx);
      if (nextIdx !== undefined) {
        useProjectStore.getState().setCurrentImageIndex(nextIdx);
        useUIStore.getState().addToast(`Jumped to image ${nextIdx + 1} (unlabeled) - ${result.total_missing} total`, 'info', 2000);
      } else {
        // Wrap around to first unlabeled
        useProjectStore.getState().setCurrentImageIndex(result.image_indices[0]);
        useUIStore.getState().addToast(`Wrapped to image ${result.image_indices[0] + 1} (unlabeled) - ${result.total_missing} total`, 'info', 2000);
      }
    } catch (err) {
      console.error('Failed to get unlabeled images:', err);
    }
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      // Read fresh state from stores
      const { reviewModeEnabled: reviewEnabled, mode: currentMode, setMode: setModeAction, setReviewMode: setReview } = useUIStore.getState();
      const { selectedAnnotationId: selectedAnn, refinePoints: points, clearRefinePoints: clearPoints } = useAnnotationStore.getState();
      const { nextImage: goNext, prevImage: goPrev } = useProjectStore.getState();

      switch (e.key.toLowerCase()) {
        case 'a':
        case 'arrowleft':
          goPrev();
          break;
        case 'd':
        case 'arrowright':
          goNext();
          break;
        case 'v':
          setModeAction('view');
          break;
        case 'b':
          setModeAction('draw');
          break;
        case 'r':
          setModeAction('refine');
          break;
        case 'q':
          setReview(!reviewEnabled);
          break;
        case '1':
          // Approve selected annotation while reviewing.
          if (reviewEnabled && selectedAnn) {
            handleApprove();
          }
          break;
        case 'n':
        case '2':
          // Reject (N for No, 2 for quick access)
          if (reviewEnabled && selectedAnn) {
            handleReject();
          }
          break;
        case ']':
          // Jump to next pending image
          handleNextPending();
          break;
        case '[':
          // Jump to next unlabeled image (no annotations)
          handleNextUnlabeled();
          break;
        case 's':
          if (currentMode === 'refine' || selectedAnn) {
            handleSegment();
          }
          break;
        case 'enter':
          if (currentMode === 'refine' && points.length > 0) {
            handleSegment();
          }
          break;
        case 'escape':
          clearPoints();
          break;
        case 'delete':
        case 'backspace':
          if (selectedAnn) {
            useAnnotationStore.getState().deleteAnnotation(selectedAnn);
          }
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Empty deps - all callbacks are now stable and read fresh state

  return (
    <div className="h-screen flex flex-col">
      <Header />
      <div className="flex-1 min-h-0 flex overflow-hidden relative">
        <div className="flex-1 min-w-0 flex flex-col">
          <div className="relative flex-1 min-h-0 flex">
            <ImageCanvas />
            <SheetOverlay />
            <ZoomOverlay />
            <AgentCaption />
          </div>
          <TrackTimeline />
        </div>
        <Sidebar />
        <RunLog />
      </div>

      <AgentCursor />

      {/* Modals */}
      <CreateProjectModal />
      <OpenProjectModal />
      <ExportModal />
      <SettingsModal />

      {/* Toast Notifications */}
      <Toaster />
    </div>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppContent />
    </QueryClientProvider>
  );
}

export default App;
