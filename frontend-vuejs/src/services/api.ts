import axios from 'axios';

const API_URL = 'http://localhost:8000/api/v1';

export interface Video {
  filename: string;
  size: number;
  last_modified: string;
  thumbnail?: string;
}

export interface Detection {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: number[];
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