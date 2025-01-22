# frontend-vuejs

## Project Setup

```sh
npm install
```

### Compile and Hot-Reload for Development

```sh
npm run dev
```

### Type-Check, Compile and Minify for Production

```sh
npm run build
```

### Run Unit Tests with [Vitest](https://vitest.dev/)

```sh
npm run test:unit
```

### Lint with [ESLint](https://eslint.org/)

```sh
npm run lint
```

## Running with Docker Compose (Recommended)

The easiest way to run the frontend is using Docker Compose from the root directory:

```bash
docker compose up --build
```

This will start the frontend application at `http://localhost:3000` (or the port specified in your configuration).

### Useful Commands

```bash
# Build and start all services
docker compose up --build

# Start specific service
docker compose up frontend

# View logs
docker compose logs -f frontend

# Stop all services
docker compose down

# Reset everything (including volumes)
docker compose down -v
```
