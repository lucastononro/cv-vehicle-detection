<script setup lang="ts">
import { ref } from 'vue';
import { videoService } from '@/services/api';

const emit = defineEmits<{
  'upload-complete': []
}>();

const isDragging = ref(false);
const isUploading = ref(false);
const uploadProgress = ref(0);

const handleDrop = async (e: DragEvent) => {
  e.preventDefault();
  isDragging.value = false;

  const files = e.dataTransfer?.files;
  if (files && files.length > 0) {
    await uploadFile(files[0]);
  }
};

const handleFileInput = async (e: Event) => {
  const input = e.target as HTMLInputElement;
  if (input.files && input.files.length > 0) {
    await uploadFile(input.files[0]);
  }
};

const uploadFile = async (file: File) => {
  if (!file.name.toLowerCase().match(/\.(mp4|avi|mov)$/)) {
    alert('Please upload a valid video file (MP4, AVI, or MOV)');
    return;
  }

  try {
    isUploading.value = true;
    await videoService.uploadVideo(file);
    emit('upload-complete');
  } catch (error) {
    console.error('Error uploading video:', error);
    alert('Failed to upload video. Please try again.');
  } finally {
    isUploading.value = false;
    uploadProgress.value = 0;
  }
};
</script>

<template>
  <div
    class="upload-zone"
    :class="{ dragging: isDragging, uploading: isUploading }"
    @dragenter.prevent="isDragging = true"
    @dragover.prevent="isDragging = true"
    @dragleave.prevent="isDragging = false"
    @drop.prevent="handleDrop"
  >
    <div class="upload-content">
      <div class="upload-icon">
        <font-awesome-icon icon="cloud-upload-alt" />
      </div>
      <div class="upload-text">
        <h3>Drag and drop your video here</h3>
        <p>or</p>
        <label class="upload-button">
          Browse Files
          <input
            type="file"
            accept=".mp4,.avi,.mov"
            @change="handleFileInput"
            style="display: none"
          >
        </label>
      </div>
    </div>

    <div v-if="isUploading" class="upload-progress">
      <div class="progress-bar">
        <div class="progress" :style="{ width: uploadProgress + '%' }"></div>
      </div>
      <p>Uploading... {{ uploadProgress }}%</p>
    </div>
  </div>
</template>

<style scoped>
.upload-zone {
  border: 2px dashed var(--border-color);
  border-radius: var(--radius-md);
  padding: 3rem 2rem;
  text-align: center;
  transition: all 0.3s ease;
  background: var(--surface-color);
  cursor: pointer;
  position: relative;
  overflow: hidden;
}

.upload-zone.uploading {
  border-color: var(--primary-color);
  pointer-events: none;
}

.upload-zone::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: var(--primary-color);
  opacity: 0;
  transition: opacity 0.3s ease;
  z-index: 0;
}

.upload-zone.dragging {
  border-color: var(--primary-color);
  transform: scale(1.02);
}

.upload-zone.dragging::before {
  opacity: 0.05;
}

.upload-content {
  position: relative;
  z-index: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1.5rem;
  transition: opacity 0.3s ease;
}

.uploading .upload-content {
  opacity: 0.5;
}

.upload-icon {
  font-size: 3rem;
  color: var(--primary-color);
  transition: transform 0.3s ease;
}

.dragging .upload-icon {
  transform: translateY(-5px);
}

.upload-text {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.75rem;
}

.upload-text h3 {
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
}

.upload-text p {
  color: var(--text-secondary);
  margin: 0;
}

.upload-button {
  background-color: var(--primary-color);
  color: white;
  padding: 0.75rem 1.5rem;
  border-radius: var(--radius-md);
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.upload-button:hover {
  background-color: var(--primary-dark);
  transform: translateY(-1px);
}

.upload-progress {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  padding: 1rem;
  background: rgba(255, 255, 255, 0.9);
  backdrop-filter: blur(4px);
  border-top: 1px solid var(--border-color);
  z-index: 2;
}

.progress-bar {
  width: 100%;
  height: 6px;
  background: var(--border-color);
  border-radius: 3px;
  overflow: hidden;
  margin-bottom: 0.5rem;
  position: relative;
}

.progress {
  height: 100%;
  background: var(--primary-color);
  transition: width 0.3s ease;
  position: relative;
}

.progress::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(
    90deg,
    transparent,
    rgba(255, 255, 255, 0.3),
    transparent
  );
  animation: shimmer 1.5s infinite;
}

@keyframes shimmer {
  0% {
    transform: translateX(-100%);
  }
  100% {
    transform: translateX(100%);
  }
}

.upload-progress p {
  color: var(--text-secondary);
  font-size: 0.875rem;
  margin: 0;
}

@media (max-width: 480px) {
  .upload-zone {
    padding: 2rem 1rem;
  }

  .upload-icon {
    font-size: 2.5rem;
  }

  .upload-text h3 {
    font-size: 1.1rem;
  }
}
</style> 