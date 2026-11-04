import { cn } from '@/lib/utils';
import { AGENT_COLOR, PHASE_NAMES, modelDisplayName } from './agentStore';
import { ActionLine, Controls, PhaseSteps, ProgressLine, Words } from './agentParts';
import { useAgentNow, useAgentVisible } from './agentView';

/**
 * Muse while it works: the agent speaks in a card floating over the bottom of the
 * canvas, like subtitles under what it is doing. Big sentence, tool call under
 * it, stage and controls along the top edge.
 */
export function AgentCaption() {
  const visible = useAgentVisible();
  const { run, action, phase, running, line, thinking, text, progress } = useAgentNow();
  if (!visible || !run) return null;
  return (
    <div className="pointer-events-none absolute inset-x-0 bottom-4 z-20 flex justify-center px-6">
      <div
        data-agent-anchor="agent-bubble"
        className="pointer-events-auto relative w-full max-w-[760px] overflow-hidden rounded-xl border bg-background/85 shadow-2xl backdrop-blur-md"
        style={{ borderColor: `${AGENT_COLOR}44` }}
      >
        <div className="flex items-center gap-3 border-b border-white/5 px-4 py-1.5">
          <span className={cn('h-2 w-2 rounded-full', running && 'agent-pulse')} style={{ background: running ? AGENT_COLOR : 'hsl(240 5% 45%)' }} />
          <span className="text-[12px] font-semibold" style={{ color: AGENT_COLOR }}>{modelDisplayName(run.model)}</span>
          <PhaseSteps phase={phase} done={!running} compact />
          <span className="text-[11px] text-muted-foreground">{running ? (phase ? PHASE_NAMES[phase] : 'Starting') : 'Finished'}</span>
          <Controls className="ml-auto" />
        </div>
        <div className="px-5 pb-3.5 pt-3">
          {text ? (
            <Words key={line!.id} text={text}
              className={cn('line-clamp-3 text-[16px] leading-6',
                line!.kind === 'error' ? 'text-red-300/90' : thinking ? 'text-foreground/60' : 'text-foreground')}
            />
          ) : (
            <p className={cn('text-[16px] leading-6', running ? 'agent-shimmer' : 'text-muted-foreground')}>{running ? 'Getting ready' : 'Finished'}</p>
          )}
          {running && <ActionLine action={action} thinking={thinking} className="mt-2 text-[12px]" />}
        </div>
        <ProgressLine progress={progress} className="absolute inset-x-0 bottom-0" />
      </div>
    </div>
  );
}
