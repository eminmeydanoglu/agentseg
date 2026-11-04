import { cn } from '@/lib/utils';
import { useUIStore } from '../stores/uiStore';
import {
  AGENT_COLOR,
  PHASES,
  PHASE_NAMES,
  useAgentStore,
  type AgentAction,
  type Phase,
} from './agentStore';
import { setReplaySpeed, toggleReplay } from './replay';

/**
 * Pieces every agent view shares: the sentence, the tool call, the stage of
 * the run and the controls.
 */

export function Words({ text, className }: { text: string; className?: string }) {
  // Words settle in one after another, like the sentence is being said.
  return (
    <p className={className}>
      {text.split(/(\s+)/).map((part, i) => (/^\s+$/.test(part)
        ? part
        : <span key={i} className="agent-word" style={{ animationDelay: `${Math.min(i, 60) * 14}ms` }}>{part}</span>))}
    </p>
  );
}

export function ActionLine({ action, thinking, className }: { action: AgentAction | null; thinking: boolean; className?: string }) {
  if (!action) return null;
  const running = action.state === 'running';
  return (
    <div key={action.id} className={cn('agent-rise flex min-w-0 items-baseline gap-2 font-mono text-[11px] leading-4', className)}>
      <span className="shrink-0" style={{ color: AGENT_COLOR, opacity: thinking && !running ? 0.6 : 1 }}>{action.tool}</span>
      {action.detail && (
        <span className={cn('min-w-0 truncate', running ? 'agent-shimmer' : 'text-muted-foreground')}>{action.detail}</span>
      )}
      {action.result && (
        <span key={action.result} className={cn('ml-auto shrink-0 tabular-nums',
          action.state === 'failed' ? 'text-red-300/80' : 'text-foreground/70')}
        >
          {action.result}
        </span>
      )}
    </div>
  );
}

/** Five stages, the current one lit; finished stages are filled. */
export function PhaseSteps({ phase, done, compact }: { phase: Phase | null; done: boolean; compact?: boolean }) {
  const at = phase ? PHASES.indexOf(phase) : -1;
  return (
    <ol className={cn('flex items-center', compact ? 'gap-1' : 'gap-1.5')}>
      {PHASES.map((p, i) => {
        const past = done || i < at;
        const now = !done && i === at;
        return (
          <li key={p} className="flex items-center gap-1.5" title={PHASE_NAMES[p]}>
            {i > 0 && <span className={cn('h-px', compact ? 'w-2' : 'w-3', past || now ? 'bg-foreground/30' : 'bg-foreground/10')} />}
            <span
              className={cn('rounded-full transition-colors', compact ? 'h-1.5 w-1.5' : 'h-2 w-2', now && 'agent-pulse')}
              style={{ background: now ? AGENT_COLOR : past ? `${AGENT_COLOR}88` : 'hsl(240 4% 26%)' }}
            />
            {!compact && (
              <span className={cn('whitespace-nowrap text-[11px]', now ? 'text-foreground' : past ? 'text-muted-foreground' : 'text-muted-foreground/50')}>
                {PHASE_NAMES[p]}
              </span>
            )}
          </li>
        );
      })}
    </ol>
  );
}

export function Controls({ className }: { className?: string }) {
  const run = useAgentStore((s) => s.run);
  const replay = useAgentStore((s) => s.replay);
  const logOpen = useAgentStore((s) => s.logOpen);
  const { followAgent, setFollowAgent } = useUIStore();
  if (!run) return null;
  const running = run.status === 'running';
  const button = 'rounded px-1.5 py-0.5 hover:bg-muted/60 hover:text-foreground';
  return (
    <div className={cn('flex shrink-0 items-center gap-0.5 text-[11px] font-normal normal-case tracking-normal text-muted-foreground', className)}>
      {replay && (
        <>
          <span className="mr-1" title={replay.synthetic ? 'A hand-made demo recording built from a real run, not a real agent run' : 'A recorded run'}>
            {replay.synthetic ? 'Synthetic replay' : 'Replay'}
          </span>
          {[1, 2, 4].map((speed) => (
            <button key={speed} type="button" onClick={() => setReplaySpeed(speed)}
              className={cn(button, 'tabular-nums', replay.speed === speed && 'bg-muted text-foreground')}
            >
              {speed}×
            </button>
          ))}
          <button type="button" onClick={toggleReplay} className={button}>
            {replay.finished ? 'Again' : replay.playing ? 'Pause' : 'Play'}
          </button>
        </>
      )}
      {running && !followAgent && (
        <button type="button" onClick={() => setFollowAgent(true)} className={cn(button, 'bg-muted text-foreground')}
          title="Go back to the frame Muse works on"
        >
          Follow
        </button>
      )}
      <button type="button" onClick={() => useAgentStore.getState().setLogOpen(!logOpen)}
        className={cn(button, logOpen && 'bg-muted text-foreground')} title="Every step of the run, by stage"
      >
        Run log
      </button>
      {running && !replay && run.id && (
        <button type="button" className={cn(button, 'text-red-300/90 hover:text-red-200')}
          onClick={() => { void fetch(`/api/agent/runs/${run.id}/stop`, { method: 'POST' }); }}
        >
          Stop
        </button>
      )}
      {!running && !replay && (
        <button type="button" onClick={() => useAgentStore.getState().dismiss()} className={button}>Close</button>
      )}
    </div>
  );
}

export function ProgressLine({ progress, className }: { progress?: { done: number; total: number }; className?: string }) {
  if (!progress) return null;
  return (
    <div className={cn('h-[2px] overflow-hidden bg-foreground/10', className)}>
      <div className="h-full transition-[width] duration-200" style={{ width: `${(progress.done / progress.total) * 100}%`, background: AGENT_COLOR }} />
    </div>
  );
}
