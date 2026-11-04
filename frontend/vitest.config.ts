import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'jsdom',
    env: { NODE_ENV: 'test' },
    restoreMocks: true,
    include: ['tests/**/*.test.{ts,tsx}'],
  },
});
