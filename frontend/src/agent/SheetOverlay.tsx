import { getOptimizedImageUrl } from '../api/client';
import { useProjectStore } from '../stores/projectStore';
import { useUIStore } from '../stores/uiStore';
import { AGENT_COLOR, STATUS_COLORS, useAgentStore } from './agentStore';

/**
 * The overview the agent is reading (`agentseg sheet`): the same frames, laid
 * out on the canvas. When the agent goes on to work on one of them, its cursor
 * clicks that tile, the tile is outlined for a moment, and the canvas shows
 * that frame (agentEvents.ts times both).
 *
 * It only covers the canvas while the person follows the agent; a click
 * puts the frame back so manual tools are never blocked.
 */

export function SheetOverlay() {
  const sheet = useAgentStore((s) => s.sheet);
  const tracks = useAgentStore((s) => s.tracks);
  const { images, project } = useProjectStore();
  const following = useUIStore((s) => s.followAgent);

  if (!sheet || !following) return null;
  const shown = sheet;
  const statuses = shown.trackId != null ? tracks[shown.trackId]?.statuses : undefined;
  const cols = shown.frames.length > 16 ? 6 : shown.frames.length > 9 ? 4 : 3;
  const aspect = images[0] ? images[0].width / images[0].height : 16 / 9;

  return (
    <div
      // The bottom padding keeps the tiles clear of Muse's caption.
      className="absolute inset-0 z-10 flex items-center justify-center bg-background px-6 pt-6 pb-[168px] animate-in fade-in duration-200"
      onPointerDown={() => useAgentStore.setState({ sheet: null })}
      title="The overview Muse is reading. Click to return to the frame."
    >
      <div
        className="grid w-full gap-1.5"
        style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`, maxWidth: `calc((100vh - 380px) * ${aspect} * ${cols} / ${Math.ceil(shown.frames.length / cols)})` }}
      >
        {shown.frames.map((f) => {
          const image = images[f];
          if (!image) return null;
          const picked = shown.pick === f;
          const dim = shown.pick != null && !picked;
          const status = statuses?.[f];
          return (
            <div
              key={f}
              data-agent-sheet-frame={f}
              className="relative overflow-hidden rounded-[3px] bg-muted transition-[opacity,box-shadow] duration-200"
              style={{
                aspectRatio: `${aspect}`,
                opacity: dim ? 0.35 : 1,
                boxShadow: picked ? `0 0 0 2px ${AGENT_COLOR}` : '0 0 0 0 transparent',
                // The outline lands with the cursor's click.
                transitionDelay: shown.pick != null ? `${shown.pickDelay ?? 0}ms` : '0ms',
              }}
            >
              <img
                src={getOptimizedImageUrl(image.id, 320, 70, project?.root_dir)}
                alt=""
                className="h-full w-full object-cover"
                draggable={false}
              />
              <span className="absolute bottom-1 left-1.5 font-mono text-[10px] leading-none text-white/85 [text-shadow:0_1px_2px_rgb(0_0_0/0.8)]">
                {f + 1}
              </span>
              {status && (
                <span className="absolute inset-x-0 bottom-0 h-[3px]" style={{ background: STATUS_COLORS[status] }} />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
