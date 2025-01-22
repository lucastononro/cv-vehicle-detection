<script setup lang="ts">
import { ref } from 'vue';
import VideoUpload from '@/components/VideoUpload.vue';
import VideoList from '@/components/VideoList.vue';
import Breadcrumb from '@/components/Breadcrumb.vue';

const videoListRef = ref<InstanceType<typeof VideoList> | null>(null);

const handleUploadComplete = () => {
  videoListRef.value?.refresh();
};
</script>

<template>
  <div class="selection-view">
    <Breadcrumb />
    
    <div class="content-section">
      <div class="upload-section">
        <div class="section-header">
          <h2>Upload Video</h2>
          <p>Supported formats: MP4</p>
        </div>
        <div class="upload-form">
          <VideoUpload @upload-complete="handleUploadComplete" />
        </div>
      </div>

      <div class="items-section">
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
@import '@/styles/selection-view.css';
</style> 