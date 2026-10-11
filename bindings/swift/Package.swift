// swift-tools-version: 5.7
import PackageDescription
let package = Package(
    name: "TurboJev",
    platforms: [.iOS(.v12), .macOS(.v10_13)],
    products: [.library(name: "TurboJev", targets: ["TurboJev"])],
    targets: [
        .systemLibrary(name: "CTurboJev", path: "Sources/CTurboJev"),
        .target(name: "TurboJev", dependencies: ["CTurboJev"], path: "Sources/TurboJev"),
        .testTarget(name: "TurboJevTests", dependencies: ["TurboJev"], path: "Tests/TurboJevTests")
    ]
)
