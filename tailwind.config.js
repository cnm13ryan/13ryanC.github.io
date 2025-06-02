module.exports = {
  content: [
    './layouts/**/*.html',
    './content/**/*.{md,html}',
    './assets/**/*.js',
    // If you use Hugo's hugo_stats.json approach, include that too
    './hugo_stats.json',
  ],
  safelist: [
    // Keeps the class from ever being purged, even if Tailwind
    // doesn't see it in the content scan for some reason
    'toggle-heading',
  ],
  theme: { extend: {} },
  plugins: [],
};
