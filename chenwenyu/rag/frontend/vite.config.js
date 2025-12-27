import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

console.log('Vite config loaded with proxy settings')

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
        configure: (proxy, options) => {
          console.log('Proxy configured for /api ->', options.target)
          proxy.on('proxyReq', (proxyReq, req, res) => {
            console.log('Proxying request:', req.method, req.url, '→', options.target + req.url)
          })
        }
      }
    }
  }
})