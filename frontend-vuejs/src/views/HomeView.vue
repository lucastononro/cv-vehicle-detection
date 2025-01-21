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
    <div class="header-section">
      <h1>Detection System</h1>
      <p class="subtitle">Upload your videos and detect custom stuff using AI computer vision powered by YOLOvX</p>
    </div>

    <div class="content-section">
      <div class="upload-section">
        <div class="section-header">
          <h2>Upload Video</h2>
          <p>Supported formats: MP4, AVI, MOV</p>
        </div>
        <VideoUpload @upload-complete="handleUploadComplete" />
      </div>

      <div class="videos-section">
        <div class="section-header">
          <h2>Your Videos</h2>
          <p>Click on a video to view and process it</p>
        </div>
        <VideoList ref="videoListRef" />
      </div>
    </div>
  </div>
</template>

<style scoped>
.home {
  display: flex;
  flex-direction: column;
  gap: 4rem;
  width: 100%;
  max-width: 1440px;
  padding: 0 2rem;
  margin: 0 auto;
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
  display: grid;
  grid-template-columns: minmax(300px, 400px) minmax(300px, 1fr);
  gap: 3rem;
  width: 100%;
  margin: 0;
  align-items: start;
}

.section-header {
  margin-bottom: 1.5rem;
}

.section-header h2 {
  font-size: 1.75rem;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 0.5rem;
}

.section-header p {
  color: var(--text-secondary);
  font-size: 1rem;
}

.upload-section {
  position: sticky;
  top: 90px;
  width: 100%;
}

.videos-section {
  width: 100%;
}

@media (max-width: 1200px) {
  .content-section {
    grid-template-columns: 1fr;
    gap: 2rem;
  }

  .upload-section {
    position: static;
  }
}

@media (max-width: 768px) {
  .home {
    padding: 0 1rem;
  }

  h1 {
    font-size: 2.5rem;
  }

  .subtitle {
    font-size: 1.1rem;
  }

  .section-header h2 {
    font-size: 1.5rem;
  }
}
</style>
