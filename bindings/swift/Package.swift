// swift-tools-version: 5.9
import PackageDescription
let package = Package(name:"TurboJev", platforms:[.iOS(.v13),.macOS(.v12)], products:[.library(name:"TurboJev",targets:["TurboJev"])], targets:[.systemLibrary(name:"CTurboJev",path:"Sources/CTurboJev"),.target(name:"TurboJev",dependencies:["CTurboJev"],path:"Sources/TurboJev")])
