<script setup lang="ts">
import { ref, onMounted, onUnmounted, nextTick } from 'vue';
import { useRoute } from 'vue-router';
import Breadcrumb from '@/components/Breadcrumb.vue';
import { videoService } from '@/services/api';

const route = useRoute();
const videoName = route.params.filename as string;
const isProcessing = ref(false);
const showInference = ref(false);
const originalVideoRef = ref<HTMLVideoElement | null>(null);
const inferenceVideoRef = ref<HTMLImageElement | null>(null);
const inferenceUrl = ref<string>('');
const availableModels = ref<string[]>([]);
const selectedModel = ref<string>('yolo11n.pt');

const loadAvailableModels = async () => {
  try {
    availableModels.value = await videoService.getAvailableModels();
    if (availableModels.value.length > 0) {
      selectedModel.value = availableModels.value[0];
    }
  } catch (error) {
    console.error('Error loading models:', error);
  }
};

const handleInferenceError = () => {
  console.error('Error loading inference stream');
  isProcessing.value = false;
  showInference.value = false;
};

const handleInferenceLoaded = () => {
  console.log('Inference stream loaded successfully');
  isProcessing.value = false;
};

const handleVisibilityChange = () => {
  if (document.hidden && showInference.value) {
    console.log('Page hidden, stopping inference');
    resetInference();
  }
};

const resetInference = () => {
  if (inferenceVideoRef.value) {
    inferenceVideoRef.value.src = '';
  }
  // Abort the fetch request if it's still ongoing
  if (inferenceUrl.value) {
    const controller = new AbortController();
    controller.abort();
  }
  showInference.value = false;
  isProcessing.value = false;
  inferenceUrl.value = '';
};

const startInference = async () => {
  try {
    // Reset previous inference if it exists
    if (showInference.value) {
      resetInference();
    }
    
    isProcessing.value = true;
    showInference.value = true;
    
    // Wait for the next tick to ensure the image element is mounted
    await nextTick();
    
    if (!inferenceVideoRef.value) {
      console.error('Inference image element not found');
      resetInference();
      return;
    }
    
    console.log('Starting inference stream...');
    inferenceUrl.value = videoService.getVideoInferenceStreamUrl(videoName, selectedModel.value);
    console.log('Setting stream source to:', inferenceUrl.value);
    
  } catch (error) {
    console.error('Error starting inference:', error);
    resetInference();
  }
};

onMounted(async () => {
  document.addEventListener('visibilitychange', handleVisibilityChange);
  await loadAvailableModels();
});

onUnmounted(() => {
  document.removeEventListener('visibilitychange', handleVisibilityChange);
  resetInference();
});
</script>

<template>
  <div class="video-view">
    <div class="header-section">
      <h1>Detection System</h1>
      <p class="subtitle">Upload your videos and detect vehicles using advanced AI technology powered by YOLOv8</p>
    </div>

    <div class="content-section">
      <Breadcrumb />
      
      <div class="video-container" :class="{ 'with-inference': showInference }">
        <div class="video-grid">
          <div class="video-player original">
            <h3>Original Video</h3>
            <video 
              ref="originalVideoRef"
              controls
            >
              <source :src="videoService.getVideoStreamUrl(videoName)" type="video/mp4">
              Your browser does not support the video tag.
            </video>
          </div>
          
          <div v-if="showInference" class="video-player inference">
            <h3>Live Detection</h3>
            <div class="inference-container">
              <img
                ref="inferenceVideoRef"
                :src="inferenceUrl"
                :style="{
                  display: showInference && !isProcessing ? 'block' : 'none',
                  height: '100%',
                  objectFit: 'contain'
                }"
                alt="Processed video stream"
                @error="handleInferenceError"
                @load="handleInferenceLoaded"
              />
              <div v-if="isProcessing" class="loading-overlay">
                <font-awesome-icon icon="spinner" spin />
                <span>Processing video...</span>
              </div>
            </div>
          </div>
        </div>
        <div class="video-name">
            <h2>{{ decodeURIComponent(videoName) }}</h2>
        </div>
        
        
        <div class="video-actions">
          <div class="video-info">
            <div class="model-selection">
              <label for="model-select">Select Detection Model:</label>
              <select 
                id="model-select" 
                v-model="selectedModel"
                :disabled="isProcessing"
                class="model-dropdown"
              >
                <option 
                  v-for="model in availableModels" 
                  :key="model" 
                  :value="model"
                >
                  {{ model }}
                </option>
              </select>
            </div>
          </div>
          <button 
            class="process-button"
            :disabled="isProcessing"
            @click="startInference"
          >
            <font-awesome-icon :icon="isProcessing ? 'spinner' : 'play'" :class="{ 'fa-spin': isProcessing }" />
            <span>{{ isProcessing ? 'Processing...' : 'Process Video' }}</span>
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.video-view {
  display: flex;
  flex-direction: column;
  gap: 2rem;
  width: 100%;
}

.header-section {
  text-align: center;
  max-width: 800px;
  margin: 0 auto;
  padding: 2rem 0;
  width: 100%;
}

h1 {
  font-size: 3.5rem;
  font-weight: 800;
  color: var(--text-primary);
  margin-bottom: 1.5rem;
  line-height: 1.2;
  background: linear-gradient(to right, var(--primary-color), var(--primary-dark));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.subtitle {
  font-size: 1.25rem;
  color: var(--text-secondary);
  max-width: 600px;
  margin: 0 auto;
}

.content-section {
  display: flex;
  flex-direction: column;
  gap: 2rem;
  width: 100%;
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 2rem;
}

.video-container {
  background: var(--surface-color);
  border-radius: var(--radius-md);
  border: 1px solid var(--border-color);
  overflow: hidden;
  width: 33.333%;
  min-width: 480px;
  align-self: flex-start;
  transition: width 0.3s ease;
}

.video-container.with-inference {
  width: 100%;
}

.video-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 2rem;
  padding: 2rem;
  transition: all 0.3s ease;
}

.with-inference .video-grid {
  grid-template-columns: 1fr 2fr;
}

.video-player {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.video-player h3 {
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
}

.video-player video {
  width: 100%;
  aspect-ratio: 16/9;
  background: black;
  border-radius: var(--radius-md);
  object-fit: contain;
}

.inference-container {
  position: relative;
  width: 100%;
  aspect-ratio: 16/9;
  background: black;
  border-radius: var(--radius-md);
  overflow: hidden;
}

.inference-container img {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.video-name {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  padding: 0 2rem;
  margin-top: 1rem;
}

.video-name h2 {
  font-size: 1.2rem;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
}

.video-actions {
  padding: 2rem;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  border-top: 1px solid var(--border-color);
}

.video-info {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.video-info h2 {
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
}

.model-selection {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.model-selection label {
  font-size: 1rem;
  color: var(--text-secondary);
  white-space: nowrap;
}

.model-dropdown {
  padding: 0.5rem 1rem;
  font-size: 1rem;
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  background-color: var(--surface-color);
  color: var(--text-primary);
  cursor: pointer;
  min-width: 200px;
  transition: all 0.2s;
}

.model-dropdown:hover:not(:disabled) {
  border-color: var(--primary-color);
}

.model-dropdown:disabled {
  opacity: 0.7;
  cursor: not-allowed;
}

.process-button {
  align-self: flex-end;
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem 1.5rem;
  font-size: 1.1rem;
  font-weight: 500;
  background-color: var(--primary-color);
  color: white;
  border: none;
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: all 0.2s;
}

.process-button:hover:not(:disabled) {
  background-color: var(--primary-dark);
  transform: translateY(-1px);
}

.process-button:disabled {
  opacity: 0.7;
  cursor: not-allowed;
}

.loading-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 1rem;
  color: white;
  font-size: 1.2rem;
}

.loading-overlay svg {
  font-size: 2rem;
}

@media (max-width: 1200px) {
  .video-container:not(.with-inference) {
    width: 50%;
  }

  .with-inference .video-grid {
    grid-template-columns: 1fr;
  }

  .inference-container {
    width: 100%;
  }
}

@media (max-width: 768px) {
  .content-section {
    padding: 0 1rem;
  }

  .video-container {
    width: 100%;
    min-width: unset;
  }

  h1 {
    font-size: 2.5rem;
  }

  .subtitle {
    font-size: 1.1rem;
  }

  .video-actions {
    padding: 1.5rem;
  }

  .process-button {
    align-self: stretch;
  }

  .video-info {
    margin-right: 0;
    gap: 0.75rem;
  }

  .video-name {
    padding: 0 1.5rem;
  }

  .model-selection {
    min-width: unset;
  }
}
</style> 