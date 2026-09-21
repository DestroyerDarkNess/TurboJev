# Node / TypeScript binding

`@turbojev/node` resolves model paths/cache in Node and executes all native calls in one worker thread. The native addon is a thin wrapper over `turbojev-host-native`; no scoring/calibration/schema logic lives in TypeScript.
