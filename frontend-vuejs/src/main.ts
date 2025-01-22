import './assets/main.css'

import { createApp } from 'vue'
import { createPinia } from 'pinia'
import { library } from '@fortawesome/fontawesome-svg-core'
import { FontAwesomeIcon } from '@fortawesome/vue-fontawesome'
import { 
  faFilm, 
  faCloudUploadAlt,
  faCog,
  faHome,
  faVideo,
  faFile,
  faClock,
  faChevronRight,
  faSpinner,
  faPlay,
  faUpload
} from '@fortawesome/free-solid-svg-icons'
import { faGithub } from '@fortawesome/free-brands-svg-icons'

import App from './App.vue'
import router from './router'

// Add icons to the library
library.add(
  faFilm, 
  faCloudUploadAlt, 
  faCog, 
  faHome, 
  faVideo, 
  faGithub,
  faFile,
  faClock,
  faChevronRight,
  faSpinner,
  faPlay,
  faUpload
)

const app = createApp(App)

app.use(createPinia())
app.use(router)
app.component('font-awesome-icon', FontAwesomeIcon)

app.mount('#app')
