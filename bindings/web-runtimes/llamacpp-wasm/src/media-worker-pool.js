// mtmd's mel preprocessors spawn three pthreads independently of n_threads.
// llama and the media encoder also retain CPU thread pools. Emscripten must
// prestart these workers: spawning one while a synchronous join is waiting can
// deadlock before the new worker receives its initialization message.
export function reserveMediaWorkers(source, inferenceThreads) {
  const header = 'var Module=typeof Module!="undefined"?Module:{};';
  if (!source.startsWith(header)) {
    throw new Error('Unsupported Wllama worker script: cannot reserve multimodal workers safely.');
  }
  const poolSize = Math.max(4, 2 * inferenceThreads + 4);
  const setup = `\nModule.pthreadPoolSize = Math.max(Module.pthreadPoolSize || 0, ${poolSize});
if (Module.mainScriptUrlOrBlob === 'throw new Error("Multithreading is not enabled")') {
  Module.mainScriptUrlOrBlob = new Blob([${JSON.stringify(source)}], { type: 'text/javascript' });
}\n`;
  // Keep the original declaration first: Wllama replaces that declaration when
  // embedding the glue in its main worker. Pthreads load the unmodified glue.
  return header + setup + source.slice(header.length);
}
