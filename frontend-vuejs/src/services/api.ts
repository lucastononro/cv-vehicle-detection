import axios from 'axios';

const API_URL = 'http://localhost:8000/api/v1';

export interface Video {
  filename: string;
  size: number;
  last_modified: string;
  thumbnail?: string;
}

export interface Image {
  filename: string;
  size: number;
  last_modified: string;
}

export interface Detection {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: number[];
  text?: string;
  text_confidence?: number;
}

export interface ImageInferenceResult {
  model_name: string;
  detections: Detection[];
  processed_image: string;
}

const api = axios.create({
  baseURL: API_URL,
});

export const videoService = {
  async uploadVideo(file: File) {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post('/videos/upload/', formData);
    return response.data;
  },

  async listVideos(): Promise<Video[]> {
    const response = await api.get('/videos/list/');
    return response.data;
  },

  async processVideo(videoName: string) {
    const response = await api.post(`/videos/inference/${videoName}/save`);
    return response.data;
  },

  getVideoStreamUrl(videoName: string) {
    return `${API_URL}/videos/stream/${encodeURIComponent(videoName)}`;
  },

  async getAvailableModels(): Promise<string[]> {
    const response = await api.get('/videos/models');
    return response.data;
  },

  getVideoInferenceStreamUrl(videoName: string, modelName?: string) {
    const baseUrl = `${API_URL}/videos/inference/${encodeURIComponent(videoName)}/stream`;
    const url = modelName ? `${baseUrl}?model_name=${encodeURIComponent(modelName)}` : baseUrl;
    console.log('Generated inference URL:', url);
    return url;
  }
};

export const imageService = {
  async uploadImage(file: File) {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post('/images/upload/', formData);
    return response.data;
  },

  async listImages(): Promise<Image[]> {
    const response = await api.get('/images/list/');
    return response.data;
  },

  async processImage(imageName: string, useOcr: boolean = true, modelName?: string): Promise<ImageInferenceResult> {
    const response = await api.get(`/images/inference/${imageName}`, {
      params: { 
        use_ocr: useOcr,
        model_name: modelName
      }
    });
    return response.data;
  },

  getImageUrl(imageName: string): string {
    return `${API_URL}/images/${encodeURIComponent(imageName)}`;
  },

  getInferenceImageUrl(imageName: string): string {
    return `${API_URL}/images/inference/${encodeURIComponent(imageName)}`;
  },

  async getAvailableModels(): Promise<string[]> {
    const response = await api.get('/videos/models');
    return response.data;
  }
}; 