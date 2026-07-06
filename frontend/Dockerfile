# ---- Build stage: install deps and build Next.js ----
FROM node:20-alpine AS builder
WORKDIR /app

# Install dependencies (ci uses lock file for reproducible builds)
COPY package.json package-lock.json ./
RUN npm ci

# Copy source and build
COPY . .
ENV NEXT_PUBLIC_API_URL=http://localhost:8000
RUN npm run build

# ---- Production stage: minimal runtime ----
FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

# Copy only what's needed for production
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 3000
CMD ["node", "server.js"]
