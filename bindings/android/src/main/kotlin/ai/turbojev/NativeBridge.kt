package ai.turbojev
internal object NativeBridge {
    init { System.loadLibrary("turbojev_jni") }
    @JvmStatic external fun open(optionsJson:String):Long
    @JvmStatic external fun evaluate(handle:Long, requestJson:String):String
    @JvmStatic external fun compatible(handle:Long, profile:String, requestJson:String):String
    @JvmStatic external fun classify(handle:Long, state:String, optionsJson:String):String
    @JvmStatic external fun close(handle:Long)
}
