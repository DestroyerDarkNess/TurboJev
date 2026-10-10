import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { resolve } from "node:path";
export function loadNative(options = {}) {
    const require = createRequire(import.meta.url);
    const candidates = [
        options.nativePath,
        process.env.TURBOJEV_NATIVE_PATH,
        resolve(process.cwd(), "turbojev.node"),
        `@turbojev/node-${process.platform}-${process.arch}`,
    ].filter(Boolean);
    const failures = [];
    for (const candidate of candidates) {
        try {
            if (candidate.endsWith(".node") && !existsSync(candidate))
                continue;
            return require(candidate);
        }
        catch (error) {
            failures.push(`${candidate}: ${String(error)}`);
        }
    }
    throw new Error(`RUNTIME_UNAVAILABLE: TurboJev native addon not found. Tried: ${failures.join(" | ")}`);
}
