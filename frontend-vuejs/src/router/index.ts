import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '@/views/HomeView.vue'
import VideoSelectionView from '@/views/VideoSelectionView.vue'
import ImageSelectionView from '@/views/ImageSelectionView.vue'
import VideoView from '@/views/VideoView.vue'
import ImageView from '@/views/ImageView.vue'

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
    }
  ],
})

export default router
