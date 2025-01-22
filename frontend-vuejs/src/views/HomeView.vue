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
</style>
