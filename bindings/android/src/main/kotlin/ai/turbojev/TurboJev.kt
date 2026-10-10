package ai.turbojev

import android.os.Handler
import android.os.HandlerThread
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.CompletableFuture

class TurboJev private constructor(
    private val thread: HandlerThread,
    private val handler: Handler,
    private var handle: Long,
) : AutoCloseable {
    companion object {
        fun compileSchema(schema: JSONObject): JSONArray =
            JSONArray(NativeBridge.compileSchema(schema.toString()))

        fun load(options: JSONObject): CompletableFuture<TurboJev> {
            val future = CompletableFuture<TurboJev>()
            val thread = HandlerThread("TurboJevNative")
            thread.start()
            val handler = Handler(thread.looper)
            handler.post {
                try {
                    val handle = NativeBridge.open(options.toString())
                    future.complete(TurboJev(thread, handler, handle))
                } catch (error: Throwable) {
                    thread.quitSafely()
                    future.completeExceptionally(error)
                }
            }
            return future
        }
    }

    fun evaluate(request: JSONObject): CompletableFuture<JSONObject> = submit {
        JSONObject(NativeBridge.evaluate(requireHandle(), request.toString()))
    }

    fun generate(request: JSONObject): CompletableFuture<JSONObject> = submit {
        JSONObject(NativeBridge.generate(requireHandle(), request.toString()))
    }

    fun evaluateCompatible(profile: String, request: JSONObject): CompletableFuture<JSONObject> = submit {
        JSONObject(NativeBridge.compatible(requireHandle(), profile, request.toString()))
    }

    private fun requireHandle(): Long {
        check(handle != 0L) { "TurboJev is closed" }
        return handle
    }

    private fun <T> submit(block: () -> T): CompletableFuture<T> {
        val future = CompletableFuture<T>()
        if (!handler.post {
            try {
                future.complete(block())
            } catch (error: Throwable) {
                future.completeExceptionally(error)
            }
        }) {
            future.completeExceptionally(IllegalStateException("TurboJev is closed"))
        }
        return future
    }

    override fun close() {
        if (Thread.currentThread() === thread) {
            releaseHandle()
        } else {
            // A stopped HandlerThread cannot process another close request.
            if (!thread.isAlive) return
            submit { releaseHandle() }.join()
        }
        thread.quitSafely()
    }

    private fun releaseHandle() {
        if (handle != 0L) {
            NativeBridge.close(handle)
            handle = 0L
        }
    }
}
