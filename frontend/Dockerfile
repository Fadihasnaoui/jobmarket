# Multi-stage build: compile the Vite app, then serve the static output with nginx.
# VITE_API_BASE_URL is baked in at BUILD time (Vite inlines import.meta.env.VITE_* into
# the bundle) — it must point to wherever the API is reachable from the *browser*, not
# from inside this container. With docker-compose publishing the api service on the
# host's 8123, the default below is correct for the demo out of the box.
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
ARG VITE_API_BASE_URL=http://localhost:8123
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
