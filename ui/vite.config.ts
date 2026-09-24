/**
 * Dev server, build, and the two backends:
 *
 *   npm run dev        proxy /api/v1 → PLEROMA_API_TARGET (default http://127.0.0.1:8767)
 *   npm run dev:mock   serve /api/v1 from ui/fixtures (no GPU, no server)
 *
 * The proxy strips the browser's Origin header: to the loom server the dev
 * proxy is a local script (like curl), so no --cors-origin is needed for it.
 * If PLEROMA_API_TOKEN is set in the environment the proxy adds the bearer
 * header itself, so the token never reaches the browser.
 */
import preact from "@preact/preset-vite";
import { defineConfig, loadEnv, type Plugin } from "vite";
import { MockBackend } from "./mock/backend.ts";

function mockBackend(opts: ConstructorParameters<typeof MockBackend>[0]): Plugin {
  const backend = new MockBackend(opts);
  return {
    name: "loom-mock-backend",
    configureServer(server) {
      server.middlewares.use(backend.middleware());
    },
    configurePreviewServer(server) {
      server.middlewares.use(backend.middleware());
    },
  };
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const isMock = mode === "mock";
  const target = env.PLEROMA_API_TARGET || "http://127.0.0.1:8767";
  const token = env.PLEROMA_API_TOKEN || "";
  const loomSeconds = Number(env.MOCK_LOOM_SECONDS || "4");
  return {
    // relative asset URLs: the loom server serves index.html at both / and /ui
    base: "./",
    plugins: [
      preact(),
      isMock &&
        mockBackend({
          loomSeconds: Number.isFinite(loomSeconds) ? loomSeconds : 4,
          token: env.MOCK_API_TOKEN || null,
        }),
    ],
    server: {
      port: 5173,
      proxy: isMock
        ? {}
        : {
            "/api/v1": {
              target,
              changeOrigin: true,
              configure(proxy) {
                proxy.on("proxyReq", (req) => {
                  req.removeHeader("origin");
                  if (token && !req.getHeader("authorization"))
                    req.setHeader("authorization", `Bearer ${token}`);
                });
              },
            },
          },
    },
    build: {
      outDir: "dist",
      sourcemap: true,
      chunkSizeWarningLimit: 900,
    },
  };
});
