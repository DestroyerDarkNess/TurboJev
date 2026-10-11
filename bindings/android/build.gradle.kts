plugins {
    id("com.android.library") version "8.7.3"
    kotlin("android") version "2.0.21"
}

group = "ai.turbojev"
version = "0.33.1"

android {
    namespace = "ai.turbojev"
    compileSdk = 35
    // CompletableFuture is available from Android 7.0 / API 24.
    defaultConfig { minSdk = 24 }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    kotlinOptions { jvmTarget = "1.8" }
}
// Repositories live in settings.gradle.kts (FAIL_ON_PROJECT_REPOS).
// org.json is provided by Android; do not bundle a desktop copy.
