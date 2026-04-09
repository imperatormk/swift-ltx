import SwiftUI

@main
struct SwiftLLMDemoApp: App {
    @State private var selectedTab = 0

    var body: some Scene {
        WindowGroup {
            TabView(selection: $selectedTab) {
                VideoView()
                    .tabItem {
                        Label("Video", systemImage: "film.stack")
                    }
                    .tag(0)

                ContentView()
                    .tabItem {
                        Label("LLM", systemImage: "text.bubble")
                    }
                    .tag(1)
            }
            .preferredColorScheme(.dark)
        }
    }
}
