<script setup lang="ts">
import { ref, onMounted } from 'vue';
import Breadcrumb from '@/components/Breadcrumb.vue';
import ImageUpload from '@/components/ImageUpload.vue';
import LabelImageList from '@/components/LabelImageList.vue';
import { imageService } from '@/services/api';

const API_URL = 'http://localhost:8000/api/v1';
const imageListRef = ref<InstanceType<typeof LabelImageList> | null>(null);
const labels = ref<Array<any>>([]);
const isLoadingLabels = ref(false);

const handleUploadComplete = () => {
  imageListRef.value?.refresh();
  fetchLabels();
};

const fetchLabels = async () => {
  try {
    isLoadingLabels.value = true;
    const response = await fetch(`${API_URL}/labelling/labels/`);
    if (!response.ok) throw new Error('Failed to fetch labels');
    labels.value = await response.json();
  } catch (error) {
    console.error('Error fetching labels:', error);
  } finally {
    isLoadingLabels.value = false;
  }
};

onMounted(() => {
  fetchLabels();
});
</script>

<template>
  <div class="image-view">
    <div class="content-section">
      <Breadcrumb />
      
      <div class="image-container">
        <!-- Upload Section -->
        <div class="upload-section">
          <div class="section-header">
            <h2>Upload Image</h2>
            <p>Supported formats: JPG, PNG</p>
          </div>
          <div class="upload-form">
            <ImageUpload @upload-complete="handleUploadComplete" />
          </div>
        </div>

        <!-- Images Section -->
        <div class="items-section">
          <div class="section-header">
            <h2>Your Images</h2>
            <p>Click on an image to generate OCR labels</p>
          </div>
          <LabelImageList 
            ref="imageListRef" 
          />
        </div>

        <!-- Labels Section -->
        <div class="labels-section">
          <div class="section-header">
            <h2>Generated Labels</h2>
            <p>Previously generated OCR labels</p>
          </div>
          
          <div v-if="isLoadingLabels" class="loading-state">
            <div class="spinner"></div>
            <p>Processing...</p>
          </div>
          
          <div v-else-if="labels.length === 0" class="empty-state">
            <div class="empty-icon">🏷️</div>
            <h3>No Labels Generated Yet</h3>
            <p>Process some images to generate OCR labels</p>
          </div>
          
          <div v-else class="labels-grid">
            <div v-for="label in labels" :key="label.text" class="label-card">
              <div class="label-image">
                <img :src="label.cropped_image" :alt="label.text" />
              </div>
              <div class="label-info">
                <h3>{{ label.text }}</h3>
                <p class="confidence">Confidence: {{ (label.confidence * 100).toFixed(1) }}%</p>
                <p class="source">From: {{ label.image_name }}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

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
}

.upload-section,
.items-section,
.labels-section {
  padding: 2rem;
}

.upload-section,
.items-section {
  border-bottom: 1px solid var(--border-color);
}

.section-header {
  margin-bottom: 1.5rem;
}

.section-header h2 {
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
}

.section-header p {
  color: var(--text-secondary);
  margin-top: 0.5rem;
}

.empty-state {
  text-align: center;
  padding: 4rem 2rem;
  background: var(--background-color);
  border-radius: var(--radius-md);
  border: 1px solid var(--border-color);
}

.empty-icon {
  font-size: 4rem;
  color: var(--text-secondary);
  margin-bottom: 1.5rem;
}

.empty-state h3 {
  font-size: 1.5rem;
  color: var(--text-primary);
  margin-bottom: 0.75rem;
}

.empty-state p {
  color: var(--text-secondary);
  font-size: 1.2rem;
}

.labels-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 1.5rem;
}

.label-card {
  background: var(--background-color);
  border-radius: var(--radius-md);
  border: 1px solid var(--border-color);
  overflow: hidden;
  transition: transform 0.2s, box-shadow 0.2s;
}

.label-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.label-image {
  width: 100%;
  height: 150px;
  overflow: hidden;
  background: #000;
}

.label-image img {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.label-info {
  padding: 1rem;
}

.label-info h3 {
  font-size: 1.25rem;
  color: var(--text-primary);
  margin-bottom: 0.5rem;
}

.confidence {
  color: var(--primary-color);
  font-weight: 500;
  margin-bottom: 0.5rem;
}

.source {
  color: var(--text-secondary);
  font-size: 0.875rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.loading-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 4rem;
  background: var(--background-color);
  border-radius: var(--radius-md);
  border: 1px solid var(--border-color);
}

.spinner {
  width: 40px;
  height: 40px;
  border: 3px solid var(--border-color);
  border-top-color: var(--primary-color);
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 1rem;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.loading-state p {
  color: var(--text-secondary);
  font-size: 1.1rem;
}
</style> 