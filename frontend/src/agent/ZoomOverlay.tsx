import { useUIStore } from '../stores/uiStore';
import { AGENT_COLOR, useAgentStore } from './agentStore';

/**
 * The close-up the agent is reading (`agentseg view --zoom`, or the zoom
 * `segment` writes): the nearest keyframe for "same object?", the frame plain,
 * and the frame with the mask outline for "whole object?". It is the same
 * image the agent sees, rendered by the backend.
 *
 * Like the overview, it covers the canvas only while the person follows the
 * agent, and a click puts the frame back.
 */

export function ZoomOverlay() {
  const zoom = useAgentStore((s) => s.zoom);
  const following = useUIStore((s) => s.followAgent);
  if (!zoom || !following) return null;

  const params = new URLSearchParams({ frame: String(zoom.frame) });
  if (zoom.trackId != null) params.set('track_id', String(zoom.trackId));
  if (zoom.previewId) params.set('preview_id', zoom.previewId);
  const panels = zoom.trackId != null
    ? ['keyframe: what it looks like', 'this frame', 'this frame + mask outline']
    : ['this frame', 'this frame + mask outline'];

  return (
    <div
      className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 bg-background px-6 pt-6 pb-[168px] animate-in fade-in duration-200"
      onPointerDown={() => useAgentStore.setState({ zoom: null })}
      title="The close-up Muse is reading. Click to return to the frame."
    >
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <span className="h-2 w-2 rounded-full" style={{ background: AGENT_COLOR }} />
        Close-up of frame {zoom.frame + 1}: {panels.join('  ·  ')}
      </div>
      <img
        key={`${zoom.frame}:${zoom.trackId}:${zoom.previewId}:${zoom.t0}`}
        src={`/api/video/render/zoom?${params}`}
        alt=""
        className="max-h-full max-w-full rounded-[3px] object-contain"
        style={{ boxShadow: `0 0 0 2px ${AGENT_COLOR}` }}
        draggable={false}
      />
    </div>
  );
}
