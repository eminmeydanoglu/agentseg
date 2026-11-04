// Types for AgentSeg

export interface Project {
  id: number;
  name: string;
  root_dir: string;
  image_count: number;
  yolo_summary?: object | null;
}

export interface ImageRecord {
  id: number;
  project_id: number;
  path: string;
  width: number;
  height: number;
  order_index: number;
}

export interface Label {
  id: number;
  project_id: number;
  name: string;
  color: string;
}

export interface Annotation {
  id: number;
  image_id: number;
  label_id: number;
  bbox: [number, number, number, number] | null;  // [x1, y1, x2, y2]
  polygon: number[] | null;
  mask_rle: object | null;
  source: 'manual' | 'sam' | 'propagated' | 'tracked' | 'yolo' | 'yolo_sam' | 'yolo_seg';
  confidence: number | null;
  status: 'approved' | 'pending' | 'needs_review' | 'rejected';
  track_id?: number | null;  // Object identity (tracks.id); null = untracked
  is_anchor?: boolean;  // approved annotation a track can re-initialise from
}

export type InteractionMode = 'view' | 'draw' | 'refine';

export interface SegmentResult {
  mask_rle: object;
  polygon: number[];
  score: number;
  bbox: [number, number, number, number];
}

// Point for refinement
export interface RefinePoint {
  x: number;
  y: number;
  type: 'positive' | 'negative';
}

// Bounding box being drawn
export interface DrawingBbox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}
