<script setup lang="ts">
import { ref } from 'vue';
import VideoUpload from '@/components/VideoUpload.vue';
import VideoList from '@/components/VideoList.vue';

const videoListRef = ref<InstanceType<typeof VideoList> | null>(null);

const handleUploadComplete = () => {
  videoListRef.value?.refresh();
};
</script>

<template>
  <div class="home">
    <h1>Vehicle Detection System</h1>
    <div class="options-container">
      <div class="option-card" @click="$router.push('/video')">
        <div class="icon">🎥</div>
        <h2>Video Processing</h2>
        <p>Upload and process videos for vehicle and license plate detection</p>
      </div>
      
      <div class="option-card" @click="$router.push('/image')">
        <div class="icon">🖼️</div>
        <h2>Image Processing</h2>
        <p>Upload and process images for vehicle and license plate detection</p>
      </div>
    </div>
    
    <div class="card-grid">
      <RouterLink to="/label-ocr" class="card">
        <div class="card-content">
          <div class="icon">🏷️</div>
          <h2>Label OCR Generation</h2>
          <p>Generate automatic labels for OCR using big guns</p>
        </div>
      </RouterLink>
    </div>
  </div>
</template>

<style scoped>
.home {
  padding: 40px;
  text-align: center;
}

.options-container {
  display: flex;
  justify-content: center;
  gap: 40px;
  margin-top: 40px;
  flex-wrap: wrap;
}

.option-card {
  background: white;
  border-radius: 10px;
  padding: 30px;
  width: 300px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  cursor: pointer;
  transition: transform 0.2s, box-shadow 0.2s;
}

.option-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
}

.icon {
  font-size: 48px;
  margin-bottom: 20px;
}

h1 {
  margin-bottom: 30px;
  color: #2c3e50;
}

h2 {
  margin: 10px 0;
  color: #2c3e50;
}

p {
  color: #666;
  line-height: 1.5;
}

.card {
  background: var(--surface-color);
  border-radius: var(--radius-lg);
  padding: 2rem;
  text-decoration: none;
  color: inherit;
  transition: transform 0.2s, box-shadow 0.2s;
  border: 1px solid var(--border-color);
}

.card:hover {
  transform: translateY(-4px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.card-icon {
  font-size: 2.5rem;
  color: var(--primary-color);
  margin-bottom: 1rem;
}

.card h2 {
  font-size: 1.5rem;
  margin-bottom: 0.5rem;
  color: var(--text-primary);
}

.card p {
  color: var(--text-secondary);
  line-height: 1.5;
}

.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 2rem;
  padding: 2rem;
}
</style>
