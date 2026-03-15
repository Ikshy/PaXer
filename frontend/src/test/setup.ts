/**
 * src/test/setup.ts
 * Vitest global test setup.
 * Imports jest-dom matchers so we can use .toBeInTheDocument() etc.
 */

import "@testing-library/jest-dom";

// Stub window.matchMedia (not available in jsdom)
Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  }),
});

// Stub ResizeObserver (used by Leaflet internally)
global.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
};

// Stub URL.createObjectURL
global.URL.createObjectURL = () => "blob:mock";
