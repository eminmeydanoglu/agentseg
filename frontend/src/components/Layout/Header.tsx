import { useUIStore } from '../../stores/uiStore';
import { useProjectStore } from '../../stores/projectStore';
import type { InteractionMode } from '../../types';
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import {
  Hand,
  Pencil,
  MousePointer2,
  Sparkles,
  CheckCircle2,
  Settings,
  FilePlus,
  FolderOpen,
  Download,
} from 'lucide-react';

/** Projects the CLI creates are all "Untitled Project"; their folder says more. */
function projectTitle(project: { name: string; root_dir: string }): string {
  if (project.name && project.name !== 'Untitled Project') return project.name;
  const parts = project.root_dir.split('/').filter(Boolean);
  const last = parts.at(-1) ?? project.name;
  return /^(agentseg|autoseg)_project$/.test(last) && parts.length > 1 ? parts.at(-2)! : last;
}

export function Header() {
  const {
    mode,
    setMode,
    reviewModeEnabled,
    setReviewMode,
    setShowCreateProjectModal,
    setShowOpenProjectModal,
    setShowExportModal,
    setShowSettingsModal,
  } = useUIStore();

  const { project } = useProjectStore();

  // Trigger segmentation via keyboard event simulation
  const triggerSegmentation = () => {
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 's', bubbles: true }));
  };

  const tools: { mode: InteractionMode; label: string; shortcut: string; icon: React.ComponentType<{ className?: string }> }[] = [
    { mode: 'view', label: 'View', shortcut: 'V', icon: Hand },
    { mode: 'draw', label: 'Draw', shortcut: 'B', icon: Pencil },
    { mode: 'refine', label: 'Refine', shortcut: 'R', icon: MousePointer2 },
  ];

  return (
    <header className="h-14 bg-background border-b flex items-center px-4 gap-4 shadow-sm z-30 relative justify-between">
      {/* Group 1: File Actions (Simplified) */}
      <div className="flex items-center gap-1">
        <Button variant="ghost" size="icon" onClick={() => setShowCreateProjectModal(true)} title="New Project">
          <FilePlus className="h-4 w-4" />
        </Button>
        <Button variant="ghost" size="icon" onClick={() => setShowOpenProjectModal(true)} title="Open Project">
          <FolderOpen className="h-4 w-4" />
        </Button>
        {project && (
          <Button variant="ghost" size="icon" onClick={() => setShowExportModal(true)} title="Export">
            <Download className="h-4 w-4" />
          </Button>
        )}

        {project && (
          <>
            <Separator orientation="vertical" className="h-6 mx-1" />
            <span className="text-xs text-muted-foreground font-mono truncate max-w-[220px]" title={project.root_dir}>
              {projectTitle(project)}
            </span>
          </>
        )}
      </div>

      {/* Group 2: Tools & Actions */}
      <div className="flex items-center gap-2">
        {/* Tools Group */}
        <div className="flex items-center bg-muted/50 rounded-lg p-1 gap-1 border">
          {tools.map((tool) => (
            <Button
              key={tool.mode}
              variant={mode === tool.mode ? "secondary" : "ghost"}
              size="sm"
              className={`h-8 px-2 gap-1 ${mode === tool.mode ? 'bg-background shadow-sm text-foreground' : 'text-muted-foreground'}`}
              onClick={() => setMode(tool.mode)}
              title={`${tool.label} (${tool.shortcut})`}
            >
              <tool.icon className="h-4 w-4" />
              <span className="text-[10px] font-mono opacity-60">{tool.shortcut}</span>
            </Button>
          ))}
        </div>

        <Separator orientation="vertical" className="h-6" />

        {/* Action: Segment */}
        <Button
          variant="default"
          size="sm"
          className="h-8 gap-2 bg-blue-600 hover:bg-blue-700 text-white"
          onClick={triggerSegmentation}
          title="Segment the selected box with SAM (S)"
        >
          <Sparkles className="h-4 w-4" />
          <span className="hidden sm:inline">Segment</span>
          <span className="text-[10px] font-mono opacity-60 hidden sm:inline">(S)</span>
        </Button>

        {/* Action: Review */}
        <Button
          variant={reviewModeEnabled ? "secondary" : "ghost"}
          size="sm"
          className={`h-8 gap-2 ${reviewModeEnabled ? 'bg-amber-500/20 text-amber-500 hover:bg-amber-500/30' : 'text-muted-foreground'}`}
          onClick={() => setReviewMode(!reviewModeEnabled)}
          title="Toggle Review Mode (Q)"
        >
          <CheckCircle2 className="h-4 w-4" />
          <span className="hidden sm:inline">Review</span>
          <span className="text-[10px] font-mono opacity-60 hidden sm:inline">(Q)</span>
        </Button>
      </div>

      {/* Group 3: Settings */}
      <div className="flex items-center gap-2">
        <Button variant="ghost" size="icon" onClick={() => setShowSettingsModal(true)} title="Settings">
          <Settings className="h-4 w-4" />
        </Button>
      </div>
    </header>
  );
}
