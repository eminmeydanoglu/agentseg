import { useLayoutEffect, useMemo, useRef, useState } from 'react';
import { X } from 'lucide-react';
import { cn } from '@/lib/utils';
import { getOptimizedImageUrl } from '../api/client';
import { useProjectStore } from '../stores/projectStore';
import { useUIStore } from '../stores/uiStore';
import {
  AGENT_COLOR,
  PHASES,
  PHASE_NAMES,
  modelDisplayName,
  useAgentStore,
  type FeedItem,
  type IssueView,
  type Phase,
} from './agentStore';

/**
 * The run log: a full-height panel on the right with everything the agent did,
 * grouped by stage (find the object, first mask, track, review issues,
 * export). Sentences are the agent's own; every step about a frame has a
 * thumbnail that jumps there. The review stage lists each flagged spot with
 * what became of it.
 */

const human = (f: number) => f + 1;

function clock(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

function goTo(frame: number) {
  const { images, setCurrentImageIndex } = useProjectStore.getState();
  if (frame < 0 || frame >= images.length) return;
  useUIStore.getState().setFollowAgent(false);
  setCurrentImageIndex(frame);
}

function Thumb({ frame, size = 'md', current }: { frame: number; size?: 'sm' | 'md' | 'lg'; current: boolean }) {
  const image = useProjectStore((s) => s.images[frame]);
  const root = useProjectStore((s) => s.project?.root_dir);
  const cls = size === 'sm' ? 'h-9 w-14' : size === 'lg' ? 'h-[60px] w-[96px]' : 'h-11 w-[70px]';
  return (
    <button type="button" onClick={() => goTo(frame)} title={`Go to frame ${human(frame)}`}
      className={cn('group relative shrink-0 overflow-hidden rounded-[4px] bg-muted ring-1 ring-white/10 transition hover:ring-2',
        cls, current && 'ring-2')}
      style={current ? { ['--tw-ring-color' as string]: AGENT_COLOR } : undefined}
    >
      {image && <img src={getOptimizedImageUrl(image.id, 192, 70, root)} alt="" loading="lazy" className="h-full w-full object-cover" />}
      <span className="absolute bottom-0 right-0 rounded-tl bg-black/70 px-1 font-mono text-[9.5px] leading-[14px] text-white/90">{human(frame)}</span>
    </button>
  );
}

interface Group { phase: Phase; items: FeedItem[] }

/** Steps carry their stage; a sentence goes with the step it introduces. */
function groupByPhase(feed: FeedItem[], current: Phase | null): Group[] {
  const groups = new Map<Phase, FeedItem[]>();
  let waiting: FeedItem[] = [];
  const add = (phase: Phase, items: FeedItem[]) => groups.set(phase, [...(groups.get(phase) ?? []), ...items]);
  for (const item of feed) {
    if (item.kind === 'report') continue;
    if (item.kind === 'step' && item.phase) { add(item.phase, [...waiting, item]); waiting = []; } else waiting.push(item);
  }
  if (waiting.length) add(current ?? 'find', waiting);
  return PHASES.filter((p) => groups.has(p)).map((phase) => ({ phase, items: groups.get(phase)! }));
}

/** "id 3, 414-418 lost: masks correct, distant small board" -> {3: "masks correct, distant small board"} */
function reportNotes(report?: string): Map<number, string> {
  const notes = new Map<number, string>();
  for (const line of report?.split('\n') ?? []) {
    const m = line.match(/\bid\s+(\d+)\b[^:]*:\s*(.+)$/i);
    if (m) notes.set(Number(m[1]), m[2].trim());
  }
  return notes;
}

const ISSUE_STATE: Record<IssueView['state'], { word: string; color: string }> = {
  open: { word: 'open', color: '#e0823d' },
  reviewed: { word: 'checked · mask is right', color: '#3fa66b' },
  fixed: { word: 'fixed', color: AGENT_COLOR },
};

function Issues({ issues, notes, done, frame }: { issues: IssueView[]; notes: Map<number, string>; done: boolean; frame: number }) {
  if (!issues.length) return null;
  const noteOf = (issue: IssueView) => issue.note ?? (issue.id != null ? notes.get(issue.id) : undefined);
  // A finished run with a note in its report: the agent looked and left it as is.
  const stateOf = (issue: IssueView) => (issue.state === 'open' && done && noteOf(issue) ? 'reviewed' : issue.state);
  return (
    <div className="mb-3 rounded-lg border bg-muted/15">
      <div className="flex items-center justify-between border-b px-3 py-1.5 text-[11px] text-muted-foreground">
        <span>{issues.length} {issues.length === 1 ? 'spot' : 'spots'} SAM flagged</span>
        <span>{issues.filter((i) => stateOf(i) !== 'open').length} resolved</span>
      </div>
      <ul className="divide-y divide-white/5">
        {issues.map((issue) => {
          const note = noteOf(issue);
          const state = stateOf(issue);
          const n = issue.end - issue.start + 1;
          return (
            <li key={issue.key} className="flex gap-3 px-3 py-2.5">
              <Thumb frame={issue.worst} size="lg" current={frame === issue.worst} />
              <div className="min-w-0 flex-1">
                <div className="flex items-baseline gap-2 text-[13px]">
                  <button type="button" className="font-medium hover:underline" onClick={() => goTo(issue.start)}>
                    {n === 1 ? `Frame ${human(issue.start)}` : `Frames ${human(issue.start)}–${human(issue.end)}`}
                  </button>
                  <span className="text-[11.5px] text-muted-foreground">{[issue.kind, n > 1 ? `${n} frames` : ''].filter(Boolean).join(' · ')}</span>
                </div>
                <div className="mt-0.5 flex items-center gap-1.5 text-[11.5px]" style={{ color: ISSUE_STATE[state].color }}>
                  <span className="h-1.5 w-1.5 rounded-full" style={{ background: ISSUE_STATE[state].color }} />
                  {ISSUE_STATE[state].word}
                </div>
                {note && <p className="mt-1 text-[12.5px] leading-[18px] text-foreground/80">{note}</p>}
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function Item({ item, t0, frame }: { item: FeedItem; t0: number; frame: number }) {
  const at = <span className="w-9 shrink-0 pt-[3px] text-right font-mono text-[10.5px] tabular-nums text-muted-foreground/60">{clock(item.ts - t0)}</span>;
  if (item.kind === 'thought') {
    return (
      <li className="flex gap-3 py-1.5">
        {at}
        <p className="min-w-0 flex-1 border-l-2 pl-3 text-[13.5px] leading-[20px] text-foreground/95" style={{ borderColor: `${AGENT_COLOR}88` }}>{item.text}</p>
      </li>
    );
  }
  if (item.kind === 'error') {
    return <li className="flex gap-3 py-1.5">{at}<p className="min-w-0 flex-1 text-[12.5px] leading-5 text-red-300/85">{item.text}</p></li>;
  }
  return (
    <li className="flex items-center gap-3 py-1.5">
      {at}
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline gap-2 text-[12.5px] leading-5">
          <span className={cn('min-w-0 truncate',
            item.state === 'running' ? 'agent-shimmer' : item.state === 'failed' ? 'text-muted-foreground/60 line-through' : 'text-foreground/80')}
          >
            {item.text}
          </span>
          {item.meta && <span className="ml-auto shrink-0 tabular-nums text-[11.5px] text-muted-foreground">{item.meta}</span>}
        </div>
        {item.frames && item.frames.length > 0 && (
          <div className="mt-1.5 flex gap-1 overflow-x-auto pb-1">
            {item.frames.map((f) => <Thumb key={f} frame={f} size="sm" current={frame === f} />)}
          </div>
        )}
      </div>
      {item.frame != null && <Thumb frame={item.frame} current={frame === item.frame} />}
    </li>
  );
}

/** The agent's closing report: its first paragraph, the rest on request. */
function Report({ text }: { text: string }) {
  const [full, setFull] = useState(false);
  const [head, ...rest] = text.split(/\n\s*\n/);
  const body = rest.join('\n\n').split('\n').map((l) => l.replace(/^\s*[-*]\s+/, '– ')).join('\n');
  return (
    <section className="mt-4 rounded-lg border px-4 py-3" style={{ borderColor: `${AGENT_COLOR}44`, background: `${AGENT_COLOR}0a` }}>
      <h3 className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">Report</h3>
      <p className="text-[13.5px] leading-5 text-foreground">{head}</p>
      {body && full && <p className="mt-2 whitespace-pre-line text-[12.5px] leading-[19px] text-foreground/85">{body}</p>}
      {body && (
        <button type="button" onClick={() => setFull(!full)} className="mt-1.5 text-[11.5px] text-muted-foreground hover:text-foreground">
          {full ? 'Less' : 'Full report'}
        </button>
      )}
    </section>
  );
}

export function RunLog() {
  const open = useAgentStore((s) => s.logOpen);
  const run = useAgentStore((s) => s.run);
  const feed = useAgentStore((s) => s.feed);
  const phase = useAgentStore((s) => s.phase);
  const issues = useAgentStore((s) => s.issues);
  const clockNow = useAgentStore((s) => s.eventClock);
  const frame = useProjectStore((s) => s.currentImageIndex);
  const scroller = useRef<HTMLDivElement>(null);
  const stick = useRef(true);

  const groups = useMemo(() => {
    const out = groupByPhase(feed, phase);
    // The flagged spots have a home even before the agent takes a step about them.
    if (issues.length && !out.some((g) => g.phase === 'review')) {
      out.splice(out.filter((g) => PHASES.indexOf(g.phase) < PHASES.indexOf('review')).length, 0, { phase: 'review', items: [] });
    }
    return out;
  }, [feed, phase, issues.length]);
  const report = useMemo(() => [...feed].reverse().find((f) => f.kind === 'report')?.text, [feed]);
  const notes = useMemo(() => reportNotes(report), [report]);

  useLayoutEffect(() => {
    const el = scroller.current;
    if (el && stick.current && run?.status === 'running') el.scrollTop = el.scrollHeight;
  }, [feed, run?.status]);

  if (!open || !run) return null;
  const running = run.status === 'running';
  const t0 = run.startedAt;
  const elapsed = (run.endedAt ?? clockNow) - t0;
  const done = !running;

  return (
    <aside className="agent-drawer absolute inset-y-0 right-0 z-40 flex w-[min(480px,100%)] flex-col border-l bg-background shadow-2xl">
      <header className="flex shrink-0 items-start gap-3 border-b px-5 py-3.5">
        <div className="min-w-0 flex-1">
          <div className="flex items-baseline gap-2">
            <h2 className="text-[15px] font-semibold">Run log</h2>
            <span className="text-[13px]" style={{ color: AGENT_COLOR }}>{modelDisplayName(run.model)}</span>
          </div>
          <div className="mt-0.5 text-[11.5px] text-muted-foreground">
            {[running ? 'running' : run.status, clock(elapsed),
              run.usage?.calls != null ? `${run.usage.calls} calls` : '',
              issues.length ? `${issues.length} flagged spots` : ''].filter(Boolean).join(' · ')}
          </div>
          {run.instruction && <p className="mt-1.5 line-clamp-2 text-[12px] text-foreground/70" title={run.instruction}>“{run.instruction}”</p>}
        </div>
        <button type="button" onClick={() => useAgentStore.getState().setLogOpen(false)}
          className="rounded p-1 text-muted-foreground hover:bg-muted hover:text-foreground" title="Close the run log"
        >
          <X className="h-4 w-4" />
        </button>
      </header>
      <div
        ref={scroller}
        className="min-h-0 flex-1 overflow-y-auto px-5 pb-6"
        onScroll={(e) => {
          const el = e.currentTarget;
          stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 32;
        }}
      >
        {report && done && <Report text={report} />}
        <ol>
          {groups.map((group) => {
            const index = PHASES.indexOf(group.phase);
            const current = running && group.phase === phase;
            const steps = group.items.filter((i) => i.kind === 'step');
            return (
              <li key={group.phase} className="mt-5">
                <div className="sticky top-0 z-10 -mx-5 mb-1 flex items-center gap-2.5 bg-background/95 px-5 py-1.5 backdrop-blur">
                  <span className={cn('flex h-5 w-5 items-center justify-center rounded-full text-[10.5px] font-semibold', current && 'agent-pulse')}
                    style={{ background: current ? AGENT_COLOR : `${AGENT_COLOR}33`, color: current ? '#14121f' : AGENT_COLOR }}
                  >
                    {index + 1}
                  </span>
                  <h3 className="text-[13px] font-semibold">{PHASE_NAMES[group.phase]}</h3>
                  <span className="text-[11px] text-muted-foreground">{steps.length} {steps.length === 1 ? 'step' : 'steps'}</span>
                  {group.items.length > 0 && (
                    <span className="ml-auto font-mono text-[10.5px] text-muted-foreground/70">
                      {clock(group.items[0].ts - t0)}–{clock(group.items.at(-1)!.ts - t0)}
                    </span>
                  )}
                </div>
                {group.phase === 'review' && <Issues issues={issues} notes={notes} done={done} frame={frame} />}
                <ul>
                  {group.items.map((item) => <Item key={item.id} item={item} t0={t0} frame={frame} />)}
                </ul>
              </li>
            );
          })}
        </ol>
        {!groups.length && <p className="mt-6 text-[13px] text-muted-foreground">Nothing yet.</p>}
      </div>
    </aside>
  );
}
