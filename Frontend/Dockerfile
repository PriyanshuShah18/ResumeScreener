FROM node:20-slim

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

EXPOSE 5173

# Vite default dev server
CMD ["npm", "run", "dev", "--", "--host"]
