<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { imageService, type Image } from '@/services/api';
import { useRouter } from 'vue-router';

const API_URL = 'http://localhost:8000/api/v1';
const images = ref<Image[]>([]);
const isLoading = ref(true);
const isRefreshing = ref(false);
const router = useRouter();

const formatFileSize = (bytes: number): string => {
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  if (bytes === 0) return '0 Byte';
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${Math.round(bytes / Math.pow(1024, i))} ${sizes[i]}`;
};

const formatDate = (dateString: string): string => {
  return new Date(dateString).toLocaleString();
};

const viewImage = (filename: string) => {
  router.push(`/image/${encodeURIComponent(filename)}`);
};

const getImageUrl = (image: Image) => {
  return `${API_URL}/images/inference/${encodeURIComponent(image.filename)}`;
};

const fetchImages = async () => {
  if (images.value.length > 0) {
    isRefreshing.value = true;
  } else {
    isLoading.value = true;
  }
  
  try {
    images.value = await imageService.listImages();
  } catch (error) {
    console.error('Error fetching images:', error);
  } finally {
    isLoading.value = false;
    isRefreshing.value = false;
  }
};

// Expose the refresh method
defineExpose({
  refresh: fetchImages
});

onMounted(fetchImages);
</script>

<template>
  <div class="image-list">
    <div v-if="isLoading" class="grid">
      <div v-for="n in 3" :key="n" class="image-card skeleton">
        <div class="image-thumbnail">
          <div class="skeleton-thumbnail"></div>
        </div>
        <div class="image-info">
          <div class="skeleton-title"></div>
          <div class="image-meta">
            <div class="skeleton-meta"></div>
            <div class="skeleton-meta"></div>
          </div>
        </div>
      </div>
    </div>

    <div v-else-if="images.length === 0" class="empty-state">
      <font-awesome-icon icon="image" class="empty-icon" />
      <h3>No images uploaded yet</h3>
      <p>Upload your first image to get started</p>
    </div>
    
    <div v-else class="grid">
      <template v-if="isRefreshing">
        <div class="image-card skeleton">
          <div class="image-thumbnail">
            <div class="skeleton-thumbnail"></div>
          </div>
          <div class="image-info">
            <div class="skeleton-title"></div>
            <div class="image-meta">
              <div class="skeleton-meta"></div>
              <div class="skeleton-meta"></div>
            </div>
          </div>
        </div>
      </template>
      
      <div 
        v-for="image in images" 
        :key="image.filename" 
        class="image-card" 
        :class="{ 'fade': isRefreshing }"
        @click="viewImage(image.filename)"
      >
        <div class="image-thumbnail">
          <img 
            :src="image.thumbnail || getImageUrl(image)" 
            :alt="image.filename"
            loading="lazy"
          />
        </div>
        <div class="image-info">
          <h3>{{ image.filename }}</h3>
          <div class="image-meta">
            <span>{{ formatFileSize(image.size) }}</span>
            <span>{{ formatDate(image.last_modified) }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.image-list {
  width: 100%;
}

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

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 2rem;
}

.image-card {
  background: var(--surface-color);
  border-radius: var(--radius-md);
  padding: 1.5rem;
  display: flex;
  align-items: center;
  gap: 1.5rem;
  cursor: pointer;
  transition: all 0.2s ease;
  border: 1px solid var(--border-color);
  position: relative;
  overflow: hidden;
  min-height: 120px;
}

.image-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: var(--primary-color);
  opacity: 0;
  transition: opacity 0.2s ease;
  z-index: 0;
}

.image-card:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
  border-color: var(--primary-color);
}

.image-card:hover::before {
  opacity: 0.05;
}

.image-thumbnail {
  flex-shrink: 0;
  width: 160px;
  height: 90px;
  border-radius: var(--radius-md);
  background: var(--background-color);
  position: relative;
  z-index: 1;
  overflow: hidden;
}

.image-thumbnail img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  transition: transform 0.3s ease;
}

.image-card:hover .image-thumbnail img {
  transform: scale(1.05);
}

.image-info {
  flex: 1;
  min-width: 0;
  position: relative;
  z-index: 1;
  padding-right: 1rem;
}

.image-info h3 {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 500;
  color: var(--text-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin-bottom: 0.5rem;
}

.image-meta {
  display: flex;
  gap: 1rem;
  color: var(--text-secondary);
  font-size: 0.875rem;
}

/* Skeleton Loading */
.skeleton {
  pointer-events: none;
}

.skeleton-thumbnail {
  width: 100%;
  height: 100%;
  background: linear-gradient(110deg, var(--border-color) 8%, var(--surface-color) 18%, var(--border-color) 33%);
  background-size: 200% 100%;
  animation: shine 1.5s linear infinite;
}

.skeleton-title {
  width: 80%;
  height: 1rem;
  background: var(--border-color);
  border-radius: var(--radius-sm);
}

.skeleton-meta {
  width: 40%;
  height: 0.875rem;
  background: var(--border-color);
  border-radius: var(--radius-sm);
}

@keyframes shine {
  to {
    background-position-x: -200%;
  }
}

.fade {
  opacity: 0.5;
  pointer-events: none;
}

@media (max-width: 768px) {
  .grid {
    grid-template-columns: 1fr;
  }
}
</style> 