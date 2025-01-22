<!-- ImageView.vue -->
<template>
  <div class="image-view">
    <div class="content-section">
      <Breadcrumb />
      
      <div class="image-container" :class="{ 'with-inference': showInference }">
        <div class="image-grid">
          <div class="image-display original">
            <h3>Original Image</h3>
            <div class="image-wrapper">
              <img 
                :src="imageService.getImageUrl(imageName)"
                alt="Original image"
              />
            </div>
          </div>
          
          <div v-if="showInference" class="image-display inference">
            <h3>Detection Results</h3>
            <div class="inference-container">
              <img
                v-if="processedImage && !isProcessing"
                :src="processedImage"
                alt="Processed image"
              />
              <div v-if="isProcessing" class="loading-overlay">
                <font-awesome-icon icon="spinner" spin class="fa-spin" />
                <span>Processing image...</span>
              </div>
            </div>
          </div>
        </div>
        
        <div class="image-name">
          <h2>{{ decodeURIComponent(imageName) }}</h2>
        </div>
        
        <div class="image-actions">
          <div class="image-info">
            <!-- Pipeline Mode Toggle -->
            <div class="pipeline-toggle">
              <label class="toggle-label">
                <input
                  type="checkbox"
                  v-model="isPipelineMode"
                  :disabled="isProcessing"
                />
                <span class="toggle-text">Pipeline Mode</span>
                <span class="toggle-description">
                  {{ isPipelineMode ? 'Configure Detection Pipeline' : 'Select Single Model' }}
                </span>
              </label>
            </div>

            <!-- Pipeline Configuration (shown in pipeline mode) -->
            <div v-if="isPipelineMode" class="pipeline-config">
              <h4>Detection Pipeline</h4>
              <div class="pipeline-steps">
                <div v-for="(step, index) in pipelineSteps" :key="index" class="pipeline-step">
                  <label class="toggle-label">
                    <input
                      type="checkbox"
                      v-model="step.enabled"
                      :disabled="isProcessing || (index === 0 && step.enabled)"
                    />
                    <span class="toggle-text">{{ step.model }}</span>
                  </label>
                  <div class="step-arrow" v-if="index < pipelineSteps.length - 1">→</div>
                </div>
              </div>
            </div>

            <!-- Model Selection (only shown in single model mode) -->
            <div v-if="showModelSelection" class="model-selection">
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
            
            <!-- OCR Toggle -->
            <div class="ocr-toggle">
              <label class="toggle-label">
                <input
                  type="checkbox"
                  v-model="useOcr"
                  :disabled="isProcessing"
                />
                <span class="toggle-text">Enable OCR</span>
                <span class="toggle-description">Extract text from license plates</span>
              </label>
            </div>
          </div>
          
          <button 
            class="process-button"
            :disabled="isProcessing"
            @click="startInference"
          >
            <font-awesome-icon :icon="isProcessing ? 'spinner' : 'play'" :class="{ 'fa-spin': isProcessing }" />
            <span>{{ isProcessing ? 'Processing...' : 'Process Image' }}</span>
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, computed } from 'vue';
import { useRoute } from 'vue-router';
import { imageService, type PipelineStep, type Detection } from '@/services/api';
import Breadcrumb from '@/components/Breadcrumb.vue';

const route = useRoute();
const imageName = route.params.filename as string;
const processedImage = ref<string | null>(null);
const detections = ref<Detection[]>([]);
const isProcessing = ref<boolean>(false);
const showInference = ref<boolean>(false);
const availableModels = ref<string[]>([]);
const selectedModel = ref<string>('yolo11n');
const useOcr = ref<boolean>(true);
const isPipelineMode = ref<boolean>(false);
const isLoading = ref(false);

const pipelineSteps = ref<PipelineStep[]>([
  { model: 'yolo11n', enabled: true },
  { model: 'license_plate_detector', enabled: true },
  { model: 'easyocr', enabled: true }
]);

const showModelSelection = computed(() => !isPipelineMode.value);

const loadAvailableModels = async () => {
  try {
    availableModels.value = await imageService.getAvailableModels();
    if (availableModels.value.length > 0) {
      selectedModel.value = availableModels.value[0];
    }
  } catch (error) {
    console.error('Error loading models:', error);
  }
};

const startInference = async () => {
  if (isProcessing.value) return;
  
  try {
    isProcessing.value = true;
    showInference.value = true;
    
    const result = isPipelineMode.value
      ? await imageService.processPipeline(
          imageName,
          pipelineSteps.value,
          useOcr.value
        )
      : await imageService.processImage(
          imageName,
          useOcr.value,
          selectedModel.value
        );
    
    processedImage.value = result.processed_image;
    detections.value = result.detections;
    isProcessing.value = false;
  } catch (error) {
    console.error('Error processing image:', error);
    isProcessing.value = false;
    showInference.value = false;
  }
};

onMounted(async () => {
  await loadAvailableModels();
});
</script>

<style scoped>
.image-view {
  display: flex;
  flex-direction: column;
  gap: 2rem;
  width: 100%;
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

.image-container {
  background: var(--surface-color);
  border-radius: var(--radius-md);
  border: 1px solid var(--border-color);
  overflow: hidden;
  width: 33.333%;
  min-width: 480px;
  align-self: flex-start;
  transition: width 0.3s ease;
}

.image-container.with-inference {
  width: 100%;
}

.image-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 2rem;
  padding: 2rem;
  transition: all 0.3s ease;
}

.with-inference .image-grid {
  grid-template-columns: 1fr 2fr;
}

.image-display {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.image-display h3 {
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
}

.image-wrapper,
.inference-container {
  position: relative;
  width: 100%;
  aspect-ratio: 16/9;
  background: black;
  border-radius: var(--radius-md);
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
}

.image-wrapper img,
.inference-container img {
  max-width: 100%;
  max-height: 100%;
  width: auto;
  height: auto;
  object-fit: contain;
}

.image-name {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  padding: 0 2rem;
  margin-top: 1rem;
}

.image-name h2 {
  font-size: 1.2rem;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
}

.image-actions {
  padding: 2rem;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  border-top: 1px solid var(--border-color);
}

.image-info {
  display: flex;
  flex-direction: column;
  gap: 1rem;
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
  flex: 1;
  padding: 0.75rem;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border-color);
  background: var(--background-color);
  color: var(--text-primary);
  font-size: 0.875rem;
  cursor: pointer;
}

.ocr-toggle {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  margin-top: 1rem;
}

.toggle-label {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  cursor: pointer;
}

.toggle-label input[type="checkbox"] {
  appearance: none;
  width: 3rem;
  height: 1.5rem;
  background-color: var(--border-color);
  border-radius: 1rem;
  position: relative;
  cursor: pointer;
  transition: all 0.3s;
}

.toggle-label input[type="checkbox"]:checked {
  background-color: var(--primary-color);
}

.toggle-label input[type="checkbox"]::before {
  content: '';
  position: absolute;
  width: 1.25rem;
  height: 1.25rem;
  border-radius: 50%;
  background-color: white;
  top: 0.125rem;
  left: 0.125rem;
  transition: transform 0.3s;
}

.toggle-label input[type="checkbox"]:checked::before {
  transform: translateX(1.5rem);
}

.toggle-label input[type="checkbox"]:disabled {
  opacity: 0.7;
  cursor: not-allowed;
}

.toggle-text {
  font-size: 1rem;
  color: var(--text-primary);
  user-select: none;
}

.toggle-description {
  font-size: 0.875rem;
  color: var(--text-secondary);
}

.process-button {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.75rem;
  padding: 0.875rem 1.5rem;
  border-radius: var(--radius-sm);
  background: var(--primary-color);
  color: white;
  font-weight: 500;
  font-size: 1rem;
  cursor: pointer;
  transition: background-color 0.2s;
  border: none;
  width: 100%;
}

.process-button:hover {
  background: var(--primary-dark);
}

.process-button:disabled {
  background: var(--disabled-color);
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
}

.loading-overlay .fa-spin {
  font-size: 2rem;
}

.pipeline-toggle {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  margin-bottom: 1rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid var(--border-color);
}

.toggle-label {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  cursor: pointer;
}

.toggle-label input[type="checkbox"] {
  appearance: none;
  width: 3rem;
  height: 1.5rem;
  background-color: var(--border-color);
  border-radius: 1rem;
  position: relative;
  cursor: pointer;
  transition: all 0.3s;
}

.toggle-label input[type="checkbox"]:checked {
  background-color: var(--primary-color);
}

.toggle-label input[type="checkbox"]::before {
  content: '';
  position: absolute;
  width: 1.25rem;
  height: 1.25rem;
  border-radius: 50%;
  background-color: white;
  top: 0.125rem;
  left: 0.125rem;
  transition: transform 0.3s;
}

.toggle-label input[type="checkbox"]:checked::before {
  transform: translateX(1.5rem);
}

.toggle-label input[type="checkbox"]:disabled {
  opacity: 0.7;
  cursor: not-allowed;
}

.toggle-text {
  font-size: 1rem;
  color: var(--text-primary);
  user-select: none;
}

.toggle-description {
  font-size: 0.875rem;
  color: var(--text-secondary);
  margin-left: auto;
}

.pipeline-config {
  margin-top: 1rem;
  padding: 1rem;
  background: var(--background-color);
  border-radius: var(--radius-md);
  border: 1px solid var(--border-color);
}

.pipeline-config h4 {
  margin: 0 0 1rem 0;
  font-size: 1rem;
  color: var(--text-primary);
}

.pipeline-steps {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.pipeline-step {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.step-arrow {
  color: var(--text-secondary);
  font-size: 1.2rem;
  margin: 0 0.5rem;
}

/* Update toggle-label for pipeline steps */
.pipeline-step .toggle-label {
  flex: 1;
  padding: 0.5rem;
  background: var(--surface-color);
  border-radius: var(--radius-sm);
  border: 1px solid var(--border-color);
}

.pipeline-step .toggle-label:hover {
  border-color: var(--primary-color);
}
</style> 