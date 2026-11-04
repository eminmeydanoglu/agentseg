import type { AgentActivityEvent } from '../hooks/useAgentActivity';
import { useAnnotationStore } from '../stores/annotationStore';
import { useProjectStore } from '../stores/projectStore';

/** CLI writes must invalidate prefetched labels even when Follow agent is off. */
export function refreshAgentAnnotations(event: AgentActivityEvent) {
  const { images, currentImageIndex } = useProjectStore.getState();
  const annotations = useAnnotationStore.getState();
  const allFrames = ['video_job.started', 'video_job.finished', 'video_job.stopped',
    'video_job.error', 'track.finished', 'track.absent'].includes(event.type) ||
    (event.type === 'command.completed' && event.action === 'track');
  const oneFrame = ['video_job.frame', 'track.progress', 'track.created'].includes(event.type);
  if (!allFrames && !oneFrame) return;

  const frame = typeof event.frame === 'number' ? event.frame : event.image_index;
  const imageId = event.image_id ?? (frame == null ? undefined : images[frame]?.id);
  if (allFrames) annotations.invalidateAnnotations();
  else if (imageId != null) annotations.invalidateAnnotations(imageId);

  const current = images[currentImageIndex];
  if (current && (allFrames || current.id === imageId)) {
    void annotations.loadAnnotations(current.id).catch(() => {});
  }
  if (event.type === 'track.created') void annotations.loadLabels();
}
