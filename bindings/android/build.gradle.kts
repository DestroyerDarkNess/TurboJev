plugins {
    id("com.android.library") version "8.7.3"
    kotlin("android") version "2.0.21"
}

group = "ai.turbojev"
version = "0.28.2"

android {
    namespace = "ai.turbojev"
    compileSdk = 35
    defaultConfig { minSdk = 26 }
}
repositories { google(); mavenCentral() }
dependencies { implementation("org.json:json:20240303") }
