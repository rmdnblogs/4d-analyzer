importScripts('https://storage.googleapis.com/workbox-cdn/releases/6.5.4/workbox-sw.js');

workbox.setConfig({ debug: false });

const { precacheAndRoute } = workbox.precaching;
const { registerRoute } = workbox.routing;
const { CacheFirst, StaleWhileRevalidate } = workbox.strategies;
const { ExpirationPlugin } = workbox.expiration;

// Precache file utama
precacheAndRoute([
  { url: '/', revision: '1' },
  { url: '/index.html', revision: '1' },
  { url: '/trainWorker.js', revision: '1' },
  { url: '/data/historical_data.csv', revision: '1' },
  { url: '/icon-192x192.png', revision: '1' },
  { url: '/icon-512x512.png', revision: '1' },
  { url: '/manifest.json', revision: '1' },
  { url: '/sw.js', revision: '1' }
]);

// Cache CDN eksternal
registerRoute(
  ({ url }) => url.origin === 'https://cdn.tailwindcss.com' || url.origin === 'https://cdn.jsdelivr.net' || url.origin === 'https://unpkg.com',
  new StaleWhileRevalidate({
    cacheName: 'cdn-cache',
    plugins: [new ExpirationPlugin({ maxAgeSeconds: 7 * 24 * 60 * 60 })]
  })
);

// Cache dinamis untuk data pengguna
registerRoute(
  ({ request }) => request.destination === 'image' || request.destination === 'script',
  new CacheFirst({
    cacheName: 'dynamic-cache',
    plugins: [new ExpirationPlugin({ maxEntries: 50, maxAgeSeconds: 30 * 24 * 60 * 60 })]
  })
);

// Cache untuk API backend
registerRoute(
  ({ url }) => url.origin === 'http://localhost:8000' && url.pathname.startsWith('/'),
  new StaleWhileRevalidate({
    cacheName: 'api-cache',
    plugins: [new ExpirationPlugin({ maxAgeSeconds: 24 * 60 * 60 })]
  })
);