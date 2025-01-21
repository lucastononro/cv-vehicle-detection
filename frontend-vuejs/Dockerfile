FROM node:18.18-alpine3.17

# Set working directory
WORKDIR /app

# Set environment variables
ENV NODE_ENV=development
ENV VITE_BACKEND_URL=http://localhost:8000
ENV NODE_OPTIONS="--max-old-space-size=4096"
ENV ROLLUP_SKIP_NODEJS_CHECKS=true
ENV VITE_USE_NODE_VERSION=18.18.0
ENV npm_config_legacy_peer_deps=true

# Copy package.json and package-lock.json
COPY package.json package-lock.json ./

# Install dependencies
RUN npm install --legacy-peer-deps

# Copy the rest of the application
COPY . ./

# Expose port 5173 (Vite's default port)
EXPOSE 5173

# Start the app
CMD ["npm", "run", "dev"] 