import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    // CI Windows runners are slow on `runHttpServer` integration tests —
    // the delegated-mode test ran 5008ms locally, 30s+ on hosted Windows.
    // 60s gives generous margin without masking real hangs.
    testTimeout: 60000,
    hookTimeout: 60000,
    coverage: {
      // `--coverage` fails below these. Statements and lines hold the repo's
      // 95% floor; branches and functions are ratchets set to the whole-number
      // part of the lowest level the CI matrix measured, so they may only rise.
      thresholds: {
        statements: 95,
        lines: 95,
        branches: 97,
        functions: 100
      }
    }
  }
})
