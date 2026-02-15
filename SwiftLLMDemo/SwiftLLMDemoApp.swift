import SwiftUI

@main
struct SwiftLLMDemoApp: App {
    @State private var selectedTab = 1
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
            }
            .preferredColorScheme(.dark)
        }
    }
}
