const CACHE_NAME = 'syngenta-ai-v2';
const OFFLINE_URL = '/offline';

// Files to cache on install (app shell)
const PRECACHE_URLS = [
  '/',
  '/static/style.css',
  '/static/manifest.json',
  '/offline'
];

// API routes to cache dynamically (cache last response)
const API_CACHE_PATTERNS = [
  '/api/dashboard-stats',
  '/api/locations',
  '/api/map-data'
];

// Install — precache app shell
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      console.log('[SW] Precaching app shell');
      return cache.addAll(PRECACHE_URLS);
    })
  );
  self.skipWaiting();
});

// Activate — clean old caches
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(
        keys.filter(key => key !== CACHE_NAME).map(key => caches.delete(key))
      )
    )
  );
  self.clients.claim();
});

// Fetch — network first, fallback to cache
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);

  // Skip non-GET requests
  if (event.request.method !== 'GET') return;

  // For API calls — network first, cache the response for offline
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(
      fetch(event.request)
        .then(response => {
          // Cache successful API responses for offline use
          if (response.ok && API_CACHE_PATTERNS.some(p => url.pathname.startsWith(p))) {
            const clone = response.clone();
            caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
          }
          return response;
        })
        .catch(() => {
          // Offline — try serving from cache
          return caches.match(event.request).then(cached => {
            if (cached) return cached;
            // Return a JSON offline message for uncached API calls
            return new Response(
              JSON.stringify({ offline: true, message: 'You are offline. Showing cached data.' }),
              { headers: { 'Content-Type': 'application/json' } }
            );
          });
        })
    );
    return;
  }

  // For page/static requests — network first, fallback to cache, then offline page
  event.respondWith(
    fetch(event.request)
      .then(response => {
        // Cache static assets
        if (response.ok) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
        }
        return response;
      })
      .catch(() => {
        return caches.match(event.request).then(cached => {
          if (cached) return cached;
          // For navigation requests, show offline page
          if (event.request.mode === 'navigate') {
            return caches.match('/offline');
          }
          return new Response('', { status: 503 });
        });
      })
  );
});
