import { useUIStore } from '../../stores/uiStore';
import { useProjectStore } from '../../stores/projectStore';
import { useAnnotationStore } from '../../stores/annotationStore';
import { useEffect, useState } from 'react';
import * as api from '../../api/client';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { ListOrdered, RefreshCw, Trash2 } from 'lucide-react';

export function SettingsModal() {
  const {
    showSettingsModal,
    setShowSettingsModal,
    maskOpacity,
    setMaskOpacity,
    followAgent,
    setFollowAgent,
    addToast,
  } = useUIStore();

  const { resyncImages, project, images } = useProjectStore();
  const { labels, loadLabels } = useAnnotationStore();

  const [isRefreshing, setIsRefreshing] = useState(false);
  const [bulkDeleteAfter, setBulkDeleteAfter] = useState('');
  const [isDeleting, setIsDeleting] = useState(false);
  const [showLabelMap, setShowLabelMap] = useState(false);

  useEffect(() => {
    if (showSettingsModal && project) {
      void loadLabels();
    }
  }, [showSettingsModal, project, loadLabels]);

  const handleRefreshImages = async () => {
    setIsRefreshing(true);
    try {
      const result = await resyncImages();
      if (result.added > 0 || result.removed > 0) {
        addToast(`Updated image list: +${result.added} new, -${result.removed} removed`, 'success');
      } else {
        addToast('Image list is up to date', 'info');
      }
    } catch (err) {
      addToast('Failed to refresh images', 'error');
      console.error('Failed to refresh images:', err);
    } finally {
      setIsRefreshing(false);
    }
  };

  const handleBulkDelete = async () => {
    const afterIndex = parseInt(bulkDeleteAfter, 10);
    if (isNaN(afterIndex) || afterIndex < 0) {
      addToast('Please enter a valid image number', 'error');
      return;
    }
    if (!project) return;

    const imagesToAffect = images.length - afterIndex - 1;
    if (imagesToAffect <= 0) {
      addToast('No images after that index', 'info');
      return;
    }

    const confirmed = window.confirm(
      `This will delete ALL annotations from images after #${afterIndex + 1} (${imagesToAffect} images). This cannot be undone. Continue?`
    );
    if (!confirmed) return;

    setIsDeleting(true);
    try {
      const result = await api.deleteAnnotationsAfterIndex(project.id, afterIndex);
      addToast(`Deleted ${result.count} annotations`, 'success');
      setBulkDeleteAfter('');
    } catch (err) {
      addToast('Failed to delete annotations', 'error');
      console.error('Bulk delete failed:', err);
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <>
    <Dialog open={showSettingsModal} onOpenChange={setShowSettingsModal}>
      <DialogContent className="sm:max-w-[550px] max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Settings</DialogTitle>
        </DialogHeader>

        <div className="space-y-8 py-4">

          {/* Project Settings */}
          {project && (
            <div className="space-y-4">
              <h3 className="text-sm font-medium leading-none flex items-center gap-2 text-muted-foreground">
                Project
                <Separator className="flex-1" />
              </h3>

              <div className="space-y-3 pl-2">
                <div className="flex items-center justify-between">
                  <div className="space-y-1">
                    <Label>Refresh Image List</Label>
                    <p className="text-xs text-muted-foreground">
                      Scan the image directory for new or removed files.
                    </p>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleRefreshImages}
                    disabled={isRefreshing}
                  >
                    <RefreshCw className={`h-4 w-4 mr-2 ${isRefreshing ? 'animate-spin' : ''}`} />
                    {isRefreshing ? 'Scanning...' : 'Refresh'}
                  </Button>
                </div>

                <div className="flex items-center justify-between pt-2 border-t border-border/50">
                  <div className="space-y-1">
                    <Label>Export Label IDs</Label>
                    <p className="text-xs text-muted-foreground">
                      View the class IDs that YOLO/COCO exports will write for this project.
                    </p>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setShowLabelMap(true)}
                    disabled={labels.length === 0}
                  >
                    <ListOrdered className="h-4 w-4 mr-2" />
                    View IDs
                  </Button>
                </div>

                <div className="space-y-2 pt-2 border-t border-border/50">
                  <div className="space-y-1">
                    <Label>Delete Annotations After Image</Label>
                    <p className="text-xs text-muted-foreground">
                      Remove all annotations from images after a specific index. Enter the last image number to keep (1-based).
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <Input
                      type="number"
                      min={1}
                      max={images.length}
                      placeholder={`1-${images.length}`}
                      value={bulkDeleteAfter}
                      onChange={(e) => setBulkDeleteAfter(e.target.value)}
                      className="w-28"
                    />
                    <Button
                      variant="destructive"
                      size="sm"
                      onClick={handleBulkDelete}
                      disabled={isDeleting || !bulkDeleteAfter}
                    >
                      <Trash2 className="h-4 w-4 mr-2" />
                      {isDeleting ? 'Deleting...' : 'Delete'}
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Display Settings */}
          <div className="space-y-4">
            <h3 className="text-sm font-medium leading-none flex items-center gap-2 text-muted-foreground">
              Display
              <Separator className="flex-1" />
            </h3>

            <div className="space-y-3 pl-2">
              <div className="flex items-center justify-between gap-4 border-b border-border/50 pb-3">
                <div className="space-y-1">
                  <Label>Follow the agent</Label>
                  <p className="text-xs text-muted-foreground">
                    Keep the canvas on the frame the agent is currently inspecting.
                  </p>
                </div>
                <Switch checked={followAgent} onCheckedChange={setFollowAgent} aria-label="Follow the agent" />
              </div>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label>Mask Opacity</Label>
                  <span className="text-xs font-mono bg-muted px-2 py-0.5 rounded text-muted-foreground">
                    {(maskOpacity * 100).toFixed(0)}%
                  </span>
                </div>
                <Slider
                  min={0}
                  max={100}
                  step={1}
                  value={[maskOpacity * 100]}
                  onValueChange={(vals) => setMaskOpacity(vals[0] / 100)}
                />
              </div>
            </div>
          </div>

        </div>

        <div className="flex justify-end pt-2">
          <Button onClick={() => setShowSettingsModal(false)}>
            Done
          </Button>
        </div>
      </DialogContent>
    </Dialog>
    <Dialog open={showLabelMap} onOpenChange={setShowLabelMap}>
      <DialogContent className="sm:max-w-[560px]">
        <DialogHeader>
          <DialogTitle>Export Label IDs</DialogTitle>
          <DialogDescription>
            Export IDs are assigned from the project label list sorted by name, matching the current export code.
          </DialogDescription>
        </DialogHeader>

        {labels.length > 0 ? (
          <div className="max-h-[60vh] overflow-y-auto rounded-md border">
            <div className="grid grid-cols-[80px_80px_90px_1fr] gap-3 border-b bg-muted/40 px-3 py-2 text-xs font-medium text-muted-foreground">
              <span>YOLO ID</span>
              <span>COCO ID</span>
              <span>Label ID</span>
              <span>Name</span>
            </div>
            {labels.map((label, index) => (
              <div
                key={label.id}
                className="grid grid-cols-[80px_80px_90px_1fr] items-center gap-3 border-b px-3 py-2 text-sm last:border-b-0"
              >
                <span className="font-mono text-muted-foreground">{index}</span>
                <span className="font-mono text-muted-foreground">{index + 1}</span>
                <span className="font-mono text-muted-foreground">{label.id}</span>
                <span className="flex min-w-0 items-center gap-2">
                  <span
                    className="h-3 w-3 shrink-0 rounded-full border"
                    style={{ backgroundColor: label.color }}
                  />
                  <span className="truncate">{label.name}</span>
                </span>
              </div>
            ))}
          </div>
        ) : (
          <p className="rounded-md border p-4 text-sm text-muted-foreground">
            No labels have been created for this project yet.
          </p>
        )}

        <p className="text-xs text-muted-foreground">
          YOLO segmentation/detection label files use the zero-based YOLO ID. COCO export category IDs use the one-based COCO ID.
        </p>
      </DialogContent>
    </Dialog>
    </>
  );
}
