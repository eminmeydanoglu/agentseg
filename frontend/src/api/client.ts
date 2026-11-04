import axios from 'axios';
import type { 
  Project, 
  ImageRecord, 
  Label, 
  Annotation, 
  SegmentResult,
} from '../types';

const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

// ==================== Projects ====================

export async function createProject(projectDir: string, imageDir: string, name: string): Promise<Project> {
  const response = await api.post<Project>('/projects', {
    project_dir: projectDir,
    image_dir: imageDir,
    name,
  });
  return response.data;
}

export async function openProject(projectDir: string): Promise<Project> {
  const response = await api.post<Project>('/projects/open', {
    project_dir: projectDir,
  });
  return response.data;
}

export async function getCurrentProject(): Promise<Project | null> {
  const response = await api.get<Project | null>('/projects/current');
  return response.data;
}

export async function closeProject(): Promise<void> {
  await api.post('/projects/close');
}

export async function getSetting(key: string): Promise<string | null> {
  const response = await api.get<{ key: string; value: string | null }>(`/projects/settings/${key}`);
  return response.data.value;
}

export async function setSetting(key: string, value: string): Promise<void> {
  await api.put(`/projects/settings/${key}`, null, { params: { value } });
}

export interface ResyncImagesResult {
  added: number;
  removed: number;
  unchanged: number;
  total: number;
}

export async function resyncImages(): Promise<ResyncImagesResult> {
  const response = await api.post<ResyncImagesResult>('/projects/resync-images');
  return response.data;
}

// ==================== Images ====================

export async function listImages(): Promise<ImageRecord[]> {
  const response = await api.get<ImageRecord[]>('/images');
  return response.data;
}

export function getOptimizedImageUrl(
  imageId: number,
  maxWidth = 2048,
  quality = 85,
  projectRootDir?: string
): string {
  let url = `/api/images/${imageId}/optimized?max_width=${maxWidth}&quality=${quality}`;
  // Namespace both HTTP/bitmap caches AND the backend request. An old
  // prefetch must not cache the new project's bytes under the old DB's key.
  if (projectRootDir) url += `&project_root_dir=${encodeURIComponent(projectRootDir)}`;
  return url;
}

export async function getImagesWithStatus(status: string): Promise<{
  status: string;
  image_indices: number[];
  count: number;
}> {
  const response = await api.get(`/images/with-status/${status}`);
  return response.data;
}

// ==================== Labels ====================

export async function listLabels(): Promise<Label[]> {
  const response = await api.get<Label[]>('/labels');
  return response.data;
}

export async function createLabel(name: string, color?: string): Promise<Label> {
  const response = await api.post<Label>('/labels', { name, color });
  return response.data;
}

export async function updateLabel(
  labelId: number,
  data: { name?: string; color?: string }
): Promise<Label> {
  const response = await api.patch<Label>(`/labels/${labelId}`, data);
  return response.data;
}

// ==================== Annotations ====================

export async function listAnnotations(imageId: number, signal?: AbortSignal): Promise<Annotation[]> {
  const response = await api.get<Annotation[]>('/annotations', {
    params: { image_id: imageId },
    signal,
  });
  return response.data;
}

export async function createAnnotation(data: {
  image_id: number;
  label_id: number;
  bbox: [number, number, number, number];
  source?: string;
  status?: string;
  mask_rle?: object;
  polygon?: number[];
  track_id?: number | null;  // omit for new initial labels -> backend creates a new track
}): Promise<Annotation> {
  const response = await api.post<Annotation>('/annotations', data);
  return response.data;
}

export async function updateAnnotation(
  annotationId: number,
  data: Partial<{
    label_id: number;
    bbox: [number, number, number, number];
    polygon: number[];
    mask_rle: object;
    status: string;
    track_id: number;
    clear_track: boolean;
  }>
): Promise<Annotation> {
  const response = await api.put<Annotation>(`/annotations/${annotationId}`, data);
  return response.data;
}

export async function deleteAnnotation(annotationId: number): Promise<void> {
  await api.delete(`/annotations/${annotationId}`);
}

export async function deleteAnnotationsAfterIndex(
  projectId: number,
  afterIndex: number
): Promise<{ count: number }> {
  const response = await api.delete<{ status: string; count: number; after_index: number }>(
    `/annotations/after-index/${projectId}`,
    { params: { after_index: afterIndex } }
  );
  return { count: response.data.count };
}

export interface MissingAnnotationsResult {
  image_indices: number[];
  total_missing: number;
}

export async function findImagesMissingAnnotations(
  projectId: number,
  labelId?: number
): Promise<MissingAnnotationsResult> {
  const response = await api.get<MissingAnnotationsResult>(`/annotations/missing/${projectId}`, {
    params: labelId !== undefined ? { label_id: labelId } : {},
  });
  return response.data;
}

// ==================== ML ====================

export async function segment(
  imageId: number,
  bbox: [number, number, number, number],
  posPoints?: [number, number][],
  negPoints?: [number, number][]
): Promise<SegmentResult> {
  const response = await api.post<SegmentResult>('/ml/segment', {
    image_id: imageId,
    bbox,
    pos_points: posPoints,
    neg_points: negPoints,
  });
  return response.data;
}

// ==================== Export ====================

export interface ValidationWarning {
  annotation_id: number;
  severity: string;
  code: string;
  message: string;
}

export interface ValidateResponse {
  total_images: number;
  total_annotations: number;
  error_count: number;
  warning_count: number;
  is_valid: boolean;
  errors: ValidationWarning[];
  warnings: ValidationWarning[];
}

export async function validateProject(): Promise<ValidateResponse> {
  const response = await api.get<ValidateResponse>('/export/validate');
  return response.data;
}

export async function exportYolo(data: {
  output_dir: string;
  train_split?: number;
  seed?: number;
  approved_only?: boolean;
  include_negative?: boolean;
  labels_only?: boolean;
  labels_colocate?: boolean;
  image_intervals?: string;
}): Promise<{
  train_images: number;
  val_images: number;
  total_annotations: number;
  warnings: string[];
  is_valid: boolean;
  validation_errors: string[];
}> {
  const response = await api.post('/export/yolo', data);
  return response.data;
}

export async function exportBbox(data: {
  output_dir: string;
  format: 'yolo-detect' | 'coco';
  train_split?: number;
  seed?: number;
  approved_only?: boolean;
  include_segmentation?: boolean;
  include_negative?: boolean;
  labels_only?: boolean;
  labels_colocate?: boolean;
  image_intervals?: string;
}): Promise<{
  format: string;
  train_images: number;
  val_images: number;
  total_annotations: number;
  warnings: string[];
}> {
  const response = await api.post('/export/bbox', data);
  return response.data;
}

// ==================== Files ====================

export interface DirectoryEntry {
  name: string;
  path: string;
  is_dir: boolean;
}

export interface DirectoryListing {
  path: string;
  parent: string | null;
  entries: DirectoryEntry[];
}

export async function listDirectory(path: string, dirsOnly = true): Promise<DirectoryListing> {
  const response = await api.get<DirectoryListing>('/files/list', {
    params: { path, dirs_only: dirsOnly },
  });
  return response.data;
}

export async function getHomeDirectory(): Promise<{ path: string }> {
  const response = await api.get<{ path: string }>('/files/home');
  return response.data;
}

export default api;
