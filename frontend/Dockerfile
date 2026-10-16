# FinRAG 2.0 frontend image.
# Build context is the frontend/ directory (see infra/docker-compose.yml).

FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:20-alpine AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ENV NEXT_TELEMETRY_DISABLED=1
# next.config.js resolves the /api/* rewrite from BACKEND_URL, but Next.js
# compiles rewrites into .next/routes-manifest.json at BUILD time — a runtime
# env var cannot change them. So the backend address must be supplied here.
# Override with: docker compose build --build-arg BACKEND_URL=http://host:8000
ARG BACKEND_URL=http://backend:8000
ENV BACKEND_URL=${BACKEND_URL}
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0
COPY --from=build /app ./
RUN chown -R node:node /app
USER node
EXPOSE 3000
CMD ["npm", "run", "start"]
