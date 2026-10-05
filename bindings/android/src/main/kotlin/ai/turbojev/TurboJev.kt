package ai.turbojev
import android.os.Handler
import android.os.HandlerThread
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.CompletableFuture

class TurboJev private constructor(private val thread:HandlerThread, private val handler:Handler, private var handle:Long):AutoCloseable {
    companion object {
        fun load(options:JSONObject):CompletableFuture<TurboJev> {
            val future=CompletableFuture<TurboJev>(); val thread=HandlerThread("TurboJevNative"); thread.start(); val handler=Handler(thread.looper)
            handler.post { try { val h=NativeBridge.open(options.toString()); future.complete(TurboJev(thread,handler,h)) } catch(t:Throwable){thread.quitSafely();future.completeExceptionally(t)} }
            return future
        }
    }
    fun evaluate(request:JSONObject)=submit { JSONObject(NativeBridge.evaluate(requireHandle(),request.toString())) }
    fun generate(request:JSONObject)=submit { JSONObject(NativeBridge.generate(requireHandle(),request.toString())) }
    fun evaluateCompatible(profile:String,request:JSONObject)=submit { JSONObject(NativeBridge.compatible(requireHandle(),profile,request.toString())) }
    private fun requireHandle():Long { if(handle==0L) throw IllegalStateException("TurboJev is closed"); return handle }
    private fun <T> submit(block:()->T):CompletableFuture<T>{val f=CompletableFuture<T>();handler.post{try{f.complete(block())}catch(t:Throwable){f.completeExceptionally(t)}};return f}
    override fun close(){val f=CompletableFuture<Unit>();handler.post{try{if(handle!=0L){NativeBridge.close(handle);handle=0};f.complete(Unit)}catch(t:Throwable){f.completeExceptionally(t)}};f.join();thread.quitSafely()}
}
