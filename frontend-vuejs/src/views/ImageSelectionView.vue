<script setup lang="ts">
import { ref } from 'vue';
import Breadcrumb from '@/components/Breadcrumb.vue';
import ImageUpload from '@/components/ImageUpload.vue';
import ImageList from '@/components/ImageList.vue';

const imageListRef = ref<InstanceType<typeof ImageList> | null>(null);

const handleUploadComplete = () => {
  imageListRef.value?.refresh();
};
</script>

<template>
  <div class="selection-view">
    <Breadcrumb />
    
    <div class="content-section">
      <div class="upload-section">
        <div class="section-header">
          <h2>Upload Image</h2>
          <p>Supported formats: JPG, PNG</p>
        </div>
        <div class="upload-form">
          <ImageUpload @upload-complete="handleUploadComplete" />
        </div>
      </div>

      <div class="items-section">
        <div class="section-header">
          <h2>Your Images</h2>
          <p>Click on an image to view and process it</p>
        </div>
        <ImageList ref="imageListRef" />
      </div>
    </div>
  </div>
</template>

<style scoped>
@import '@/styles/selection-view.css';

.empty-state {
  text-align: center;
  padding: 6rem 2rem;
  background: var(--surface-color);
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

.image-grid {
  display: grid !important;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)) !important;
  gap: 1.5rem !important;
}

.image-item {
  position: relative;
  aspect-ratio: 16/9;
  padding: 0 !important;
  overflow: hidden;
}

.image-item img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.image-item .item-name {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  padding: 0.5rem;
  background: rgba(0, 0, 0, 0.7);
  color: white;
  font-size: 0.875rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
</style> 