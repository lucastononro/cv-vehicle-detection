<script setup lang="ts">
import { RouterView } from 'vue-router'
</script>

<template>
  <div class="app-container">
    <nav class="navbar">
      <div class="nav-content">
        <router-link to="/" class="logo">
          <font-awesome-icon icon="video" class="logo-icon" />
          <span>Video Detection</span>
        </router-link>
        <div class="nav-links">
          <router-link to="/" class="nav-link">
            <font-awesome-icon icon="home" />
            <span>Home</span>
          </router-link>
          <a href="https://github.com/lucastononro/cv-vehicle-detection" target="_blank" class="nav-link">
            <font-awesome-icon :icon="['fab', 'github']" />
            <span>GitHub</span>
          </a>
        </div>
      </div>
    </nav>

    <main class="main-content">
      <div class="content-wrapper">
        <RouterView />
      </div>
    </main>
  </div>
</template>

<style>
:root {
  --primary-color: #2563eb;
  --primary-dark: #1d4ed8;
  --secondary-color: #64748b;
  --background-color: #f8fafc;
  --surface-color: #ffffff;
  --text-primary: #1e293b;
  --text-secondary: #64748b;
  --border-color: #e2e8f0;
  --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
  --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
  --radius-md: 0.5rem;
}

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

html, body {
  margin: 0;
  padding: 0;
  width: 100%;
  min-height: 100vh;
  overflow-x: hidden;
}

body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background-color: var(--background-color);
  color: var(--text-primary);
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}

.app-container {
  min-height: 100vh;
  width: 100%;
  display: flex;
  flex-direction: column;
  margin: 0;
  padding: 0;
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
}

.navbar {
  background-color: var(--surface-color);
  box-shadow: var(--shadow-sm);
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 100;
  border-bottom: 1px solid var(--border-color);
  height: 70px;
  width: 100%;
}

.nav-content {
  max-width: 1440px;
  margin: 0 auto;
  padding: 0 2rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  height: 100%;
}

.logo {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--primary-color);
  text-decoration: none;
  transition: color 0.2s;
}

.logo:hover {
  color: var(--primary-dark);
}

.logo-icon {
  font-size: 1.75rem;
}

.nav-links {
  display: flex;
  gap: 2rem;
  align-items: center;
}

.nav-link {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  color: var(--text-secondary);
  text-decoration: none;
  font-weight: 500;
  font-size: 1.1rem;
  padding: 0.75rem 1rem;
  border-radius: var(--radius-md);
  transition: all 0.2s;
}

.nav-link:hover {
  color: var(--primary-color);
  background-color: var(--background-color);
}

.main-content {
  flex: 1;
  width: 100%;
  padding-top: 70px;
  background-color: var(--background-color);
  min-height: calc(100vh - 70px);
  display: flex;
  justify-content: center;
}

.content-wrapper {
  width: 100%;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
}

@media (max-width: 768px) {
  .nav-content {
    flex-direction: column;
    gap: 1rem;
    padding: 1rem;
  }

  .nav-links {
    width: 100%;
    justify-content: center;
  }
}

/* Add smooth transitions */
.router-link-active {
  color: var(--primary-color) !important;
  background-color: var(--background-color);
}

/* Global button styles */
button {
  background-color: var(--primary-color);
  color: white;
  border: none;
  padding: 0.75rem 1.5rem;
  border-radius: var(--radius-md);
  font-weight: 500;
  cursor: pointer;
  transition: background-color 0.2s;
}

button:hover {
  background-color: var(--primary-dark);
}

button:disabled {
  background-color: var(--secondary-color);
  cursor: not-allowed;
}
</style>
