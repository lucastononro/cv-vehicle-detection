import { library } from '@fortawesome/fontawesome-svg-core'
import { FontAwesomeIcon } from '@fortawesome/vue-fontawesome'
import { 
  faHome,
  faVideo,
  faSpinner,
  faUpload,
  faPlay
} from '@fortawesome/free-solid-svg-icons'
import { faGithub } from '@fortawesome/free-brands-svg-icons'

// Add icons to the library
library.add(
  faHome,
  faVideo,
  faSpinner,
  faUpload,
  faPlay,
  faGithub
)

export { FontAwesomeIcon } 