import XCTest
@testable import TurboJev

final class CompatibilityTests: XCTestCase {
    func testOpenFailureRetainsNativeThreadDiagnostics() {
        XCTAssertThrowsError(try TurboJevEngine(optionsJSON: "{}")) { error in
            guard case TurboJevError.native(let code, let message) = error else {
                return XCTFail("Expected a native SDK error, received \(error)")
            }
            XCTAssertNotEqual(code, "turbojev.native.unknown")
            XCTAssertNotEqual(message, "unknown native error")
            XCTAssertFalse(message.isEmpty)
        }
    }

    func testNativeUTF8GenerationAndIdempotentClose() throws {
        let engine = try TurboJevEngine(optionsJSON: "{\"model\":\"fixture://swift-compatibility\"}")
        defer { try? engine.close() }
        let request = "{\"context\":\"facts\",\"questions\":{\"label\":{\"type\":\"string\",\"enum\":[\"niño 🐈 中文\"],\"permutations\":1}}}"
        let response = try engine.generate(request)
        XCTAssertTrue(response.contains("niño 🐈 中文"))
        try engine.close()
        try engine.close()
        XCTAssertThrowsError(try engine.generate(request))
    }
}
