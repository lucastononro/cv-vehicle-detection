FROM node:20-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Clean install for reproducible builds
RUN npm ci

# Copy the rest of the app
COPY . .

# Expose port
EXPOSE 5173

# Start the app with proper host binding
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]