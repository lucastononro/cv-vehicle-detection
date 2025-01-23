import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '@/views/HomeView.vue'
import VideoSelectionView from '@/views/VideoSelectionView.vue'
import ImageSelectionView from '@/views/ImageSelectionView.vue'
import VideoView from '@/views/VideoView.vue'
import ImageView from '@/views/ImageView.vue'
import LabelOCRGenerationView from '@/views/LabelOCRGenerationView.vue'
import LabelOCRView from '@/views/LabelOCRView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView,
    },
    {
      path: '/video',
      name: 'video-list',
      component: VideoSelectionView,
    },
    {
      path: '/video/:filename',
      name: 'video-detail',
      component: VideoView,
    },
    {
      path: '/image',
      name: 'image-list',
      component: ImageSelectionView,
    },
    {
      path: '/image/:filename',
      name: 'image-detail',
      component: ImageView,
    },
    {
      path: '/label-ocr',
      name: 'label-ocr',
      component: LabelOCRGenerationView,
    },
    {
      path: '/label-ocr/:filename',
      name: 'label-ocr-detail',
      component: LabelOCRView,
    },
  ],
})

export default router
