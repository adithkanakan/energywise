FROM node:20-slim

WORKDIR /app

COPY package.json ./
# The cache mount and retry flags make npm survive a slow connection.
RUN --mount=type=cache,target=/root/.npm \
    npm install --fetch-retries=5 --fetch-retry-maxtimeout=120000

COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host"]
