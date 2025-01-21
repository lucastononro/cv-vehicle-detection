<script setup lang="ts">
import { computed } from 'vue';
import { useRoute } from 'vue-router';

const route = useRoute();

const breadcrumbs = computed(() => {
  const paths = route.path.split('/').filter(Boolean);
  // If we're in a video route, only show the filename
  if (paths[0] === 'video') {
    return [{
      path: `/${paths.join('/')}`,
      label: decodeURIComponent(paths[1]),
      isLast: true
    }];
  }
  // For other routes, show all path segments
  return paths.map((path, index) => {
    const isLast = index === paths.length - 1;
    const to = `/${paths.slice(0, index + 1).join('/')}`;
    const label = path.replace(/-/g, ' ');
    return { path: to, label, isLast };
  });
});
</script>

<template>
  <nav class="breadcrumb" aria-label="breadcrumb">
    <router-link to="/" class="breadcrumb-item">
      <font-awesome-icon icon="home" />
      <span>Home</span>
    </router-link>
    <span class="separator" v-if="breadcrumbs.length > 0">
      <font-awesome-icon icon="chevron-right" />
    </span>
    <template v-for="(crumb, index) in breadcrumbs" :key="crumb.path">
      <router-link 
        v-if="!crumb.isLast"
        :to="crumb.path" 
        class="breadcrumb-item"
      >
        {{ crumb.label }}
      </router-link>
      <span 
        v-else 
        class="breadcrumb-item current"
      >
        {{ crumb.label }}
      </span>
      <span class="separator" v-if="index < breadcrumbs.length - 1">
        <font-awesome-icon icon="chevron-right" />
      </span>
    </template>
  </nav>
</template>

<style scoped>
.breadcrumb {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 1rem 0;
  color: var(--text-secondary);
  font-size: 1rem;
}

.breadcrumb-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: var(--text-secondary);
  text-decoration: none;
  transition: color 0.2s;
}

.breadcrumb-item:not(.current):hover {
  color: var(--primary-color);
}

.breadcrumb-item.current {
  color: var(--text-primary);
  font-weight: 500;
}

.separator {
  color: var(--text-secondary);
  opacity: 0.5;
  font-size: 0.875rem;
}
</style> 