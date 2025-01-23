<!-- LabelOCRView.vue -->
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
            <h3>Label Generation Results</h3>
            <div class="inference-container">
              <img
                v-if="processedImage && !isProcessing"
                :src="processedImage"
                alt="Processed image"
              />
              <div v-if="isProcessing" class="loading-overlay">
                <font-awesome-icon icon="spinner" spin class="fa-spin" />
                <span>Generating labels...</span>
              </div>
            </div>
          </div>
        </div>
        
        <div class="image-name">
          <h2>{{ decodeURIComponent(imageName) }}</h2>
        </div>
        
        <div class="image-actions">
          <div class="image-info">
            <div class="info-text">
              <p>Using GPT-4 Vision for high-quality label generation</p>
            </div>
          </div>
          
          <button 
            class="process-button"
            :disabled="isProcessing"
            @click="startInference"
          >
            <font-awesome-icon :icon="isProcessing ? 'spinner' : 'play'" :class="{ 'fa-spin': isProcessing }" />
            <span>{{ isProcessing ? 'Generating...' : 'Generate Labels' }}</span>
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { useRoute } from 'vue-router';
import { imageService, type Detection } from '@/services/api';
import Breadcrumb from '@/components/Breadcrumb.vue';

const API_URL = 'http://localhost:8000/api/v1';
const route = useRoute();
const imageName = route.params.filename as string;
const processedImage = ref<string | null>(null);
const detections = ref<Detection[]>([]);
const isProcessing = ref<boolean>(false);
const showInference = ref<boolean>(false);

const startInference = async () => {
  if (isProcessing.value) return;
  
  try {
    isProcessing.value = true;
    showInference.value = true;
    
    // Use the labelling endpoint
    const response = await fetch(`${API_URL}/labelling/inference/${imageName}`);
    if (!response.ok) {
      throw new Error('Failed to process image');
    }
    const result = await response.json();
    
    if (!result.success) {
      throw new Error(result.message || 'Failed to generate labels');
    }
    
    processedImage.value = result.processed_image;
    detections.value = result.labels;
    isProcessing.value = false;
  } catch (error) {
    console.error('Error processing image:', error);
    alert(error.message);
    isProcessing.value = false;
    showInference.value = false;
  }
};

onMounted(() => {
  // Any initialization if needed
});
</script>

<style scoped>
/* Reusing all the styles from ImageView.vue */
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

.ocr-model-selection {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.ocr-model-selection label {
  font-size: 1rem;
  color: var(--text-secondary);
  white-space: nowrap;
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

.info-text {
  padding: 1rem;
  background: var(--background-color);
  border-radius: var(--radius-sm);
  border: 1px solid var(--border-color);
}

.info-text p {
  color: var(--text-secondary);
  font-size: 0.875rem;
  margin: 0;
}
</style> 