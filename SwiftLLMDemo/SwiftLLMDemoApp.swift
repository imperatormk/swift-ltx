import SwiftUI

@main
struct SwiftLLMDemoApp: App {
    @State private var selectedTab = 2
    var body: some Scene {
        WindowGroup {
            TabView(selection: $selectedTab) {
                ContentView()
                    .tabItem {
                        Label("LLM", systemImage: "text.bubble")
                    }
                    .tag(0)
                VAEView()
                    .tabItem {
                        Label("VAE", systemImage: "photo.stack")
                    }
                    .tag(1)
                VideoView()
                    .tabItem {
                        Label("Video", systemImage: "film.stack")
                    }
                    .tag(2)
            }
            .preferredColorScheme(.dark)
        }
    }
}
