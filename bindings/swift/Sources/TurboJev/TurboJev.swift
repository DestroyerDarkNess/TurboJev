import Foundation
import CTurboJev

public enum TurboJevError: Error {
    case native(code: String, message: String)
    case closed
}

private final class NativeThread {
    private final class State {
        let condition = NSCondition()
        var jobs: [() -> Void] = []
        var stopped = false
    }

    private final class ResultBox<T> {
        var result: Result<T, Error>?
    }

    private let state: State
    private let thread: Thread

    init() {
        let state = State()
        self.state = state
        self.thread = Thread {
            while true {
                state.condition.lock()
                while state.jobs.isEmpty && !state.stopped {
                    state.condition.wait()
                }
                if state.stopped && state.jobs.isEmpty {
                    state.condition.unlock()
                    return
                }
                let job = state.jobs.removeFirst()
                state.condition.unlock()
                job()
            }
        }
        self.thread.name = "TurboJevNative"
        self.thread.start()
    }

    func sync<T>(_ body: @escaping () throws -> T) throws -> T {
        if Thread.current === thread {
            return try body()
        }

        let semaphore = DispatchSemaphore(value: 0)
        let box = ResultBox<T>()
        state.condition.lock()
        if state.stopped {
            state.condition.unlock()
            throw TurboJevError.closed
        }
        state.jobs.append {
            box.result = Result { try body() }
            semaphore.signal()
        }
        state.condition.signal()
        state.condition.unlock()
        semaphore.wait()
        return try box.result!.get()
    }

    func stop() {
        state.condition.lock()
        state.stopped = true
        state.condition.signal()
        state.condition.unlock()
    }
}

public final class TurboJevEngine {
    private let dispatcher = NativeThread()
    private var handle: OpaquePointer?

    public init(optionsJSON: String) throws {
        var opened: OpaquePointer?
        let rc = try dispatcher.sync {
            optionsJSON.withCString { tj_engine_open_json($0, &opened) }
        }
        try Self.check(rc)
        guard let opened else {
            throw TurboJevError.native(
                code: "turbojev.native.null_handle",
                message: "TurboJev native engine returned a null handle"
            )
        }
        handle = opened
    }

    deinit {
        try? close()
    }

    public func evaluate(_ requestJSON: String) throws -> String {
        try call { engine, out in
            requestJSON.withCString { request in
                tj_engine_evaluate_json(engine, request, out)
            }
        }
    }
    public static func compileSchema(_ schemaJSON: String) throws -> String {
        var output: UnsafeMutablePointer<CChar>?
        let status = schemaJSON.withCString { tj_compile_schema_json($0, &output) }
        guard status == 0 else {
            throw TurboJevError.native(code: String(cString: tj_last_error_code()), message: String(cString: tj_last_error_message()))
        }
        defer { tj_string_free(output) }
        return output.map { String(cString: $0) } ?? ""
    }

    public func generate(_ requestJSON: String) throws -> String {
        try call { engine, out in
            requestJSON.withCString { request in tj_engine_generate_json(engine, request, out) }
        }
    }

    public func evaluateCompatible(profile: String, requestJSON: String) throws -> String {
        try call { engine, out in
            profile.withCString { profilePointer in
                requestJSON.withCString { requestPointer in
                    tj_engine_evaluate_compatible_json(engine, profilePointer, requestPointer, out)
                }
            }
        }
    }

    public func info() throws -> String {
        try call { engine, out in
            tj_engine_info_json(engine, out)
        }
    }

    public func close() throws {
        guard let engine = handle else {
            return
        }
        let rc = try dispatcher.sync {
            tj_engine_close(engine)
        }
        try Self.check(rc)
        handle = nil
        dispatcher.stop()
    }

    private func call(
        _ function: @escaping (OpaquePointer, UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>) -> Int32
    ) throws -> String {
        guard let engine = handle else {
            throw TurboJevError.closed
        }
        return try dispatcher.sync {
            var output: UnsafeMutablePointer<CChar>?
            let rc = function(engine, &output)
            try Self.check(rc)
            guard let pointer = output else {
                return ""
            }
            defer { tj_string_free(pointer) }
            return String(cString: pointer)
        }
    }

    private static func check(_ rc: Int32) throws {
        if rc == TJ_OK {
            return
        }
        let code = tj_last_error_code().map { String(cString: $0) } ?? "turbojev.native.unknown"
        let message = tj_last_error_message().map { String(cString: $0) } ?? "unknown native error"
        throw TurboJevError.native(code: code, message: message)
    }
}
