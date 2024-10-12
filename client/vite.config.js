import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react-swc';
//import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/NASASpaceAppsChallenge2024/',
  server: {
    port: process.env.PORT || 3000,
    host: '0.0.0.0',
  },
  preview: {
    port: process.env.PORT || 3000,
    host: '0.0.0.0',
  },
})