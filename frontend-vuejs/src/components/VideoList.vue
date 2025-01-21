<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { videoService, type Video } from '@/services/api';
import { useRouter } from 'vue-router';

const videos = ref<Video[]>([]);
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

const viewVideo = (filename: string) => {
  router.push(`/video/${encodeURIComponent(filename)}`);
};

const fetchVideos = async () => {
  if (videos.value.length > 0) {
    isRefreshing.value = true;
  } else {
    isLoading.value = true;
  }
  
  try {
    videos.value = await videoService.listVideos();
  } catch (error) {
    console.error('Error fetching videos:', error);
  } finally {
    isLoading.value = false;
    isRefreshing.value = false;
  }
};

// Expose the refresh method
defineExpose({
  refresh: fetchVideos
});

onMounted(fetchVideos);
</script>

<template>
  <div class="video-list">
    <div v-if="isLoading" class="grid">
      <div v-for="n in 3" :key="n" class="video-card skeleton">
        <div class="video-thumbnail">
          <div class="skeleton-thumbnail"></div>
        </div>
        <div class="video-info">
          <div class="skeleton-title"></div>
          <div class="video-meta">
            <div class="skeleton-meta"></div>
            <div class="skeleton-meta"></div>
          </div>
        </div>
        <div class="card-action skeleton-action"></div>
      </div>
    </div>

    <div v-else-if="videos.length === 0" class="empty-state">
      <font-awesome-icon icon="film" class="empty-icon" />
      <h3>No videos uploaded yet</h3>
      <p>Upload your first video to get started</p>
    </div>
    
    <div v-else class="grid">
      <template v-if="isRefreshing">
        <div class="video-card skeleton">
          <div class="video-thumbnail">
            <div class="skeleton-thumbnail"></div>
          </div>
          <div class="video-info">
            <div class="skeleton-title"></div>
            <div class="video-meta">
              <div class="skeleton-meta"></div>
              <div class="skeleton-meta"></div>
            </div>
          </div>
          <div class="card-action skeleton-action"></div>
        </div>
      </template>
      
      <div 
        v-for="video in videos" 
        :key="video.filename" 
        class="video-card" 
        :class="{ 'fade': isRefreshing }"
        @click="viewVideo(video.filename)"
      >
        <div class="video-thumbnail">
          <img 
            v-if="video.thumbnail"
            :src="video.thumbnail"
            :alt="video.filename"
          />
          <div class="fallback-icon" v-else>
            <font-awesome-icon icon="film" />
          </div>
        </div>
        <div class="video-info">
          <h3>{{ video.filename }}</h3>
          <div class="video-meta">
            <span class="meta-item">
              <font-awesome-icon icon="file" />
              {{ formatFileSize(video.size) }}
            </span>
            <span class="meta-item">
              <font-awesome-icon icon="clock" />
              {{ formatDate(video.last_modified) }}
            </span>
          </div>
        </div>
        <div class="card-action">
          <font-awesome-icon icon="chevron-right" />
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.video-list {
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
  grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
  gap: 2rem;
}

.video-card {
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

.video-card::before {
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

.video-card:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
  border-color: var(--primary-color);
}

.video-card:hover::before {
  opacity: 0.05;
}

.video-thumbnail {
  flex-shrink: 0;
  width: 160px;
  height: 90px;
  border-radius: var(--radius-md);
  background: var(--background-color);
  position: relative;
  z-index: 1;
  overflow: hidden;
}

.video-thumbnail img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 0.3s ease;
}

.video-thumbnail .fallback-icon {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 2rem;
  color: var(--primary-color);
  background: var(--background-color);
}

.video-card:hover .video-thumbnail img {
  transform: scale(1.05);
}

.video-info {
  flex: 1;
  min-width: 0;
  position: relative;
  z-index: 1;
}

.video-info h3 {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 500;
  color: var(--text-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin-bottom: 0.75rem;
}

.video-meta {
  display: flex;
  gap: 2rem;
}

.meta-item {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  color: var(--text-secondary);
  font-size: 1rem;
}

.meta-item svg {
  font-size: 0.9rem;
}

.card-action {
  color: var(--text-secondary);
  position: relative;
  z-index: 1;
  transition: transform 0.2s ease;
}

.video-card:hover .card-action {
  transform: translateX(4px);
  color: var(--primary-color);
}

@media (max-width: 1200px) {
  .grid {
    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
    gap: 1.5rem;
  }
}

@media (max-width: 768px) {
  .grid {
    grid-template-columns: 1fr;
  }
}

.fade {
  opacity: 0.5;
  pointer-events: none;
  transition: opacity 0.3s ease;
}

/* Update skeleton animation to be smoother */
@keyframes shimmer {
  0% {
    background-position: -200% 0;
  }
  100% {
    background-position: 200% 0;
  }
}

.skeleton {
  pointer-events: none;
  animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.skeleton-icon,
.skeleton-title,
.skeleton-meta,
.skeleton-action {
  background: linear-gradient(90deg, 
    var(--border-color) 25%, 
    var(--background-color) 50%, 
    var(--border-color) 75%
  );
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
}

/* Update skeleton styles for thumbnail */
.skeleton-thumbnail {
  width: 160px;
  height: 90px;
  border-radius: var(--radius-md);
}
</style> 