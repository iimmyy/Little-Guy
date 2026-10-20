// Little Guy menu bar app: a dot (and temperature) in the menu bar, a click-open mini console of the board,
// and a small bubble that drops out of the menu bar when something needs a look.
// It reads the board's status server (http://<board>:8080/api/status), so it works on home Wi-Fi.
import ServiceManagement
import SwiftUI

// boardURL, boardSSH and claudeLog come from Settings.swift, which build.sh writes from littleguy.env.

struct Proc: Decodable, Identifiable { let pid: Int; let name: String; let cpu: Int; let mb: Int; var id: Int { pid } }
struct Status: Decodable {
    let time: String; let cpu_temp: Int; let cores: [Int]; let procs: [Proc]
    let journal: [String]; let feed: [String]; let feed_age_s: Int?; let model: Bool; let ram_used: Int; let ram_total: Int
    let uptime_min: Int?; let throttled: Bool?; let brain_service: Bool?; let quiet: Bool?
}

// Palette from bymartins.ca
extension Color {
    static let paper = Color(red: 0.965, green: 0.953, blue: 0.922)
    static let ink = Color(red: 0.161, green: 0.153, blue: 0.137)
    static let clay = Color(red: 0.678, green: 0.341, blue: 0.231)
    static let sage = Color(red: 0.373, green: 0.478, blue: 0.322)
    static let inkBlue = Color(red: 0.184, green: 0.290, blue: 0.420)  // Little Guy's own accent
    static let muted = Color(red: 0.412, green: 0.400, blue: 0.369)
    static let paperDeep = Color(red: 0.937, green: 0.922, blue: 0.882)
    static let line = Color(red: 0.812, green: 0.792, blue: 0.749)
}

// MARK: - the bubble

/// A small paper-coloured bubble that drops out from under the menu bar icon for a few seconds. No sound.
@MainActor final class Bubble {
    private var panel: NSPanel?
    private var hideTask: Task<Void, Never>?

    func show(_ title: String, _ detail: String) {
        hideTask?.cancel()
        panel?.orderOut(nil)
        let view = NSHostingView(rootView: BubbleView(title: title, detail: detail) { [weak self] in self?.hide() })
        view.frame.size = view.fittingSize
        let p = NSPanel(contentRect: NSRect(origin: .zero, size: view.fittingSize),
                        styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: false)
        p.contentView = view
        p.isOpaque = false
        p.backgroundColor = .clear
        p.hasShadow = true
        p.level = .statusBar
        p.collectionBehavior = [.canJoinAllSpaces, .transient]
        // sit just under our menu bar icon (its window is the app's NSStatusBarWindow)
        let icon = NSApp.windows.first { String(describing: type(of: $0)).contains("StatusBar") }?.frame
            ?? NSRect(x: (NSScreen.main?.frame.maxX ?? 800) - 200, y: (NSScreen.main?.frame.maxY ?? 600) - 24, width: 24, height: 24)
        p.setFrameOrigin(NSPoint(x: icon.midX - view.fittingSize.width / 2, y: icon.minY - view.fittingSize.height + 2))
        p.alphaValue = 0
        p.orderFrontRegardless()
        NSAnimationContext.runAnimationGroup { $0.duration = 0.25; p.animator().alphaValue = 1 }
        panel = p
        hideTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(8))
            if !Task.isCancelled { self?.hide() }
        }
    }

    func hide() {
        guard let p = panel else { return }
        NSAnimationContext.runAnimationGroup({ $0.duration = 0.3; p.animator().alphaValue = 0 }) { p.orderOut(nil) }
        panel = nil
    }
}

struct BubbleView: View {
    let title: String
    let detail: String
    let dismiss: () -> Void

    var body: some View {
        VStack(spacing: -1) {
            // the little ^ that points up at the menu bar dot
            Caret().fill(Color.paper).overlay(Caret().stroke(Color.line, lineWidth: 1)).frame(width: 16, height: 8)
                .zIndex(1)
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.custom("Palatino-Italic", size: 15)).foregroundColor(.inkBlue)
                if !detail.isEmpty {
                    Text(detail).font(.serif(13)).foregroundColor(.muted).lineLimit(2)
                }
            }
            .padding(.horizontal, 14).padding(.vertical, 9)
            .frame(maxWidth: 300, alignment: .leading)
            .background(Color.paper)
            .clipShape(RoundedRectangle(cornerRadius: 10))
            .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.line, lineWidth: 1))
        }
        .padding(2)
        .onTapGesture { dismiss() }
    }
}

/// An open triangle (no bottom edge), so the stroke blends into the bubble below it.
struct Caret: Shape {
    func path(in r: CGRect) -> Path {
        Path { p in
            p.move(to: CGPoint(x: r.minX, y: r.maxY))
            p.addLine(to: CGPoint(x: r.midX, y: r.minY))
            p.addLine(to: CGPoint(x: r.maxX, y: r.maxY))
        }
    }
}

// MARK: - board state

@MainActor final class Board: ObservableObject {
    @Published var status: Status?
    @Published var online = false
    @Published var history: [Int] = []  // average CPU %, one point per second
    @Published var temps: [Int] = []    // hottest core, one point per second
    /// CPU % per program, smoothed over ~5 s so the ranking only moves when something really changes
    @Published var smoothed: [String: Double] = [:]
    /// which console the panel shows; nil = pick automatically (his brain if he spoke in the last 10 minutes)
    @Published var view: String? = nil
    @Published var claudeLines: [String] = []
    @Published var logAtBottom = true  // the console follows new lines only while scrolled to the bottom
    @Published var popAtBottom = true  // the same for the popped-out window
    @Published var panelJump = 0       // bumped by "Jump to latest"
    @Published var popJump = 0
    @Published var actionNote = ""     // result of the last button press
    @Published var chatNote = ""       // what happened to the last message you sent him
    private var replyBefore: String?   // his last reply when you sent it; a different one means he answered
    @Published var atLogin = SMAppService.mainApp.status == .enabled

    let bubble = Bubble()
    let popOut = PopOut()
    // what the bubble has already told you about
    private var started = false
    private var lastDone = ""
    private var lastCmd = ""
    private var failures = 0
    private var toldOffline = false
    private var lastUptime: Int?
    private var hotLatched = false
    private var wasThrottled = false

    var shownView: String { view ?? ((status?.feed_age_s ?? .max) < 600 ? "brain" : "board") }

    /// The latest "stats:" line his brain wrote to the feed (first-word time and talking speed).
    var lastStats: String? {
        guard let line = status?.feed.last(where: { $0.contains("stats:") }) else { return nil }
        return line.components(separatedBy: "stats:").last?.trimmingCharacters(in: .whitespaces)
    }

    func setAtLogin(_ on: Bool) {
        try? on ? SMAppService.mainApp.register() : SMAppService.mainApp.unregister()
        atLogin = SMAppService.mainApp.status == .enabled
    }

    init() {
        Task { while true { await poll(); try? await Task.sleep(for: .seconds(1)) } }
    }

    /// Last ~80 lines of Claude's live test log (only the tail of the file is read).
    func readClaudeLog() {
        guard let h = try? FileHandle(forReadingFrom: claudeLog) else { claudeLines = ["(no log yet)"]; return }
        defer { try? h.close() }
        let size = (try? h.seekToEnd()) ?? 0
        try? h.seek(toOffset: size > 16_384 ? size - 16_384 : 0)
        let text = String(decoding: h.readDataToEndOfFile(), as: UTF8.self)
        claudeLines = Array(text.split(separator: "\n", omittingEmptySubsequences: false).suffix(80).map(String.init))
    }

    func poll() async {
        readClaudeLog()
        do {
            var req = URLRequest(url: boardURL); req.timeoutInterval = 2
            let (data, _) = try await URLSession.shared.data(for: req)
            status = try JSONDecoder().decode(Status.self, from: data); online = true
            failures = 0
            history = Array((history + [average]).suffix(90))
            temps = Array((temps + [status?.cpu_temp ?? 0]).suffix(90))
            var now: [String: Double] = [:]
            // the board reports % of one core; divide by the core count so it matches the graph (% of the whole board)
            let cores = Double(max(status?.cores.count ?? 1, 1))
            for p in status?.procs ?? [] { now[p.name, default: 0] += Double(p.cpu) / cores }
            var next: [String: Double] = [:]
            for name in Set(smoothed.keys).union(now.keys) {
                let v = 0.8 * (smoothed[name] ?? 0) + 0.2 * (now[name] ?? 0)
                if v >= 0.05 { next[name] = v }
            }
            smoothed = next
            if let before = replyBefore, lastReply != before { chatNote = ""; replyBefore = nil }
        } catch {
            failures += 1
            if failures >= 5 { online = false }
        }
        checkForNews()
        started = true
    }

    /// Decide whether anything is worth a bubble. Each kind of news is told once, not every second.
    func checkForNews() {
        if let cmdIdx = claudeLines.lastIndex(where: { $0.hasPrefix("$ ") }) {
            // the header line two above the command carries the start time, so each run is unique
            let key = (cmdIdx > 0 ? claudeLines[cmdIdx - 1] : "") + claudeLines[cmdIdx]
            if started && key != lastCmd && claudeRunning {
                bubble.show("Claude started a run.", String(claudeLines[cmdIdx].dropFirst(2).prefix(70)))
            }
            lastCmd = key
        }
        if let done = claudeLines.last(where: { $0.hasPrefix("✓ done") }) {
            if started && done != lastDone {
                let cmd = claudeLines.last(where: { $0.hasPrefix("$ ") }).map { String($0.dropFirst(2).prefix(60)) } ?? ""
                bubble.show("Claude's run finished.", "\(done.dropFirst(2)) · \(cmd)")
            }
            lastDone = done
        }
        if failures >= 5 && !toldOffline && started {
            bubble.show("Board went offline.", "No answer for 5 s. Wi-Fi, power, or a crash?")
            toldOffline = true
        }
        guard let s = status, failures == 0 else { return }
        if toldOffline { toldOffline = false; bubble.show("Board is back.", "") }
        if let up = s.uptime_min {
            if let last = lastUptime, up < last { bubble.show("Board rebooted.", "Up \(up) min. Check the log if you didn't restart it.") }
            lastUptime = up
        }
        if s.cpu_temp >= 80 && !hotLatched { bubble.show("Getting hot: \(s.cpu_temp)°C.", "The heat guard steps in around 80°C."); hotLatched = true }
        if s.cpu_temp < 72 { hotLatched = false }
        let th = s.throttled ?? false
        if th && !wasThrottled { bubble.show("Throttling.", "The heat guard is slowing him down to cool off.") }
        wasThrottled = th
    }

    var busiest: [(name: String, pct: Double)] {
        smoothed.sorted { $0.value > $1.value }.prefix(3).filter { $0.value >= 0.1 }.map { ($0.key, $0.value) }
    }

    var average: Int { guard let c = status?.cores, !c.isEmpty else { return 0 }; return c.reduce(0, +) / c.count }

    /// Is one of Claude's runs in progress? (a "$ command" line with no "✓ done" after it)
    var claudeRunning: Bool {
        let lastCmd = claudeLines.lastIndex(where: { $0.hasPrefix("$ ") }) ?? -1
        let lastDone = claudeLines.lastIndex(where: { $0.hasPrefix("✓ done") }) ?? -1
        return lastCmd > lastDone
    }

    /// Brighter versions of the palette so the state reads at a glance in the menu bar:
    /// gray = can't reach the board, orange = hot / throttling / working hard, blue = a Claude run in progress, green = fine
    var dotColor: NSColor {
        guard online, let s = status else { return NSColor(red: 0.55, green: 0.55, blue: 0.55, alpha: 1) }
        if s.cpu_temp >= 80 || (s.throttled ?? false) { return NSColor(red: 0.93, green: 0.42, blue: 0.20, alpha: 1) }
        if claudeRunning { return NSColor(red: 0.26, green: 0.56, blue: 0.98, alpha: 1) }
        if average >= 40 { return NSColor(red: 0.93, green: 0.62, blue: 0.18, alpha: 1) }
        return NSColor(red: 0.30, green: 0.76, blue: 0.40, alpha: 1)
    }

    var mood: String {
        guard online else { return "Offline." }
        if status?.throttled ?? false { return "Throttling." }
        let a = average
        return a < 5 ? "Idling." : a < 40 ? "Doing a little." : "Working hard."
    }

    var lastReply: String { status?.feed.last { $0.contains("] replied") } ?? "" }

    /// Talk to him: the message goes into his inbox through the guard, exactly like tools/room/talk.sh, from this Mac
    /// only (never through the porch, which only looks in). His reply shows up in his brain feed and is spoken.
    func tell(_ text: String) {
        chatNote = "Sending…"
        let before = lastReply
        Task.detached {
            let p = Process()
            p.executableURL = URL(fileURLWithPath: orbPath)
            // arguments go through as they are (no shell), so quotes and several lines arrive intact
            p.arguments = ["-m", roomMachine, "-u", "root", "sudo", "-u", "littleguy-guard", "python3",
                           "/opt/littleguy/guard.py", "tell", text]
            let out = Pipe()
            p.standardOutput = out
            p.standardError = out
            let ran = (try? p.run()) != nil
            if ran { p.waitUntilExit() }
            let code = ran ? p.terminationStatus : -1
            let err = String(decoding: out.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            await MainActor.run {
                if code == 0 {
                    let first = text.split(separator: "\n").first.map(String.init) ?? text
                    self.chatNote = "Sent: \u{201C}\(first.prefix(60))\(text.count > first.count || first.count > 60 ? "…" : "")\u{201D} He's reading it."
                    self.replyBefore = before
                } else {
                    self.chatNote = "Couldn't reach his room (\(code)). \(err.prefix(120))"
                }
            }
        }
    }

    // MARK: buttons: commands go over SSH from this Mac only (key auth), never through the status server

    func confirmAndRun(_ title: String, _ explain: String, _ command: String) {
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = explain
        alert.addButton(withTitle: title)
        alert.addButton(withTitle: "Cancel")
        NSApp.activate(ignoringOtherApps: true)
        guard alert.runModal() == .alertFirstButtonReturn else { return }
        run(title, command)
    }

    /// Runs one command on the board without asking first; for harmless, undoable switches.
    func run(_ title: String, _ command: String) {
        actionNote = "\(title)…"
        Task.detached {
            let p = Process()
            p.executableURL = URL(fileURLWithPath: "/usr/bin/ssh")
            p.arguments = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=5", boardSSH, command]
            let out = Pipe()
            p.standardOutput = out
            p.standardError = out
            try? p.run()
            p.waitUntilExit()
            let text = String(decoding: out.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let code = p.terminationStatus
            await MainActor.run {
                self.actionNote = code == 0 ? "\(title): done. \(text)" : "\(title) failed (\(code)). \(text)"
            }
        }
    }
}

/// A plain coloured dot; menu bar images are template (monochrome) unless told otherwise.
func dot(_ color: NSColor) -> NSImage {
    let img = NSImage(size: NSSize(width: 16, height: 16), flipped: false) { rect in
        color.setFill(); NSBezierPath(ovalIn: rect.insetBy(dx: 2.5, dy: 2.5)).fill(); return true
    }
    img.isTemplate = false
    return img
}

// MARK: - the panel

extension Font {
    /// The panel's one typeface, so data lines match the headings.
    static func serif(_ size: CGFloat, italic: Bool = false) -> Font { .custom(italic ? "Palatino-Italic" : "Palatino", size: size) }
}

struct Console: View {
    @ObservedObject var board: Board

    var body: some View {
        // Same layout family as Headphone EQ: "by martins.", rule, big serif title, "State." line, muted details.
        VStack(alignment: .leading, spacing: 10) {
            Text("by martins.").font(.custom("Palatino", size: 20))
            rule
            HStack(alignment: .firstTextBaseline) {
                Text("Little ").font(.custom("Palatino", size: 44)) + Text("Guy.").font(.custom("Palatino-Italic", size: 44)).foregroundColor(.inkBlue)
                Spacer()
                Text(board.mood).font(.custom(board.status?.throttled ?? false ? "Palatino-Italic" : "Palatino", size: 22))
                    .foregroundColor(board.status?.throttled ?? false ? .clay : .ink)
            }
            if let s = board.status, board.online {
                Text("hottest core \(s.cpu_temp)°C  ·  memory \(String(format: "%.1f", Double(s.ram_used) / 1024)) GB  ·  brain \(s.model ? "awake" : "resting")")
                    .font(.serif(15)).foregroundColor(s.cpu_temp >= 80 ? .clay : .muted).padding(.top, -6)
                if let stats = board.lastStats {
                    Text("last reply: \(stats)").font(.custom("Palatino-Italic", size: 15)).foregroundColor(.inkBlue).padding(.top, -6)
                }
                LoadLine(history: board.history, temps: board.temps).padding(.top, 4)
                busyRow
            } else {
                Text("Can't reach the board. It only answers on home Wi-Fi for now.").font(.serif(15, italic: true)).foregroundColor(.muted)
            }
            rule
            tabs
            switch board.shownView {
            case "claude": log(board.claudeLines)
            case "brain":  log(board.status?.feed ?? [])
            default:       log(board.status?.journal ?? [])
            }
            if board.shownView == "brain" { TalkRow(board: board, size: 14, height: 44) }
            rule
            actions
            Toggle("Start at login", isOn: Binding(get: { board.atLogin }, set: { board.setAtLogin($0) }))
                .toggleStyle(.checkbox).tint(.ink)
            HStack {
                Spacer()
                Button("Quit") { NSApp.terminate(nil) }
            }
        }
        .padding(.horizontal, 26).padding(.vertical, 22).frame(width: 620).background(Color.paper).foregroundColor(.ink)
    }

    var rule: some View { Rectangle().fill(Color.line).frame(height: 1) }

    /// Two safe buttons. Both ask first, and both run over SSH from this Mac.
    var actions: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 10) {
                Button("Cool down") {
                    board.confirmAndRun("Cool down",
                                        "Stops anything heavy on the board (model servers, benchmarks, NPU runs) so it can cool off. Tests in progress will stop.",
                                        "pkill -x llama-server; pkill -x llama-bench; pkill -x llama-cli; pkill -x qnn-net-run; pkill -x llama-perplexity; echo stopped")
                }
                Button("Restart brain") {
                    board.confirmAndRun("Restart brain", "Restarts Little Guy's brain service.", "sudo -n systemctl restart littleguy-brain && echo restarted")
                }
                .disabled(!(board.status?.brain_service ?? false))
                .help(board.status?.brain_service ?? false ? "Restart his brain" : "His brain doesn't run as a service yet")
                Spacer()
                // power mode: quiet = status lights off and clocks capped so he runs cool without a fan
                if let quiet = board.status?.quiet, board.online {
                    HStack(spacing: 14) {
                        mode("Full power", on: !quiet) { board.run("Full power", "quiet-mode off") }
                        mode("Quiet", on: quiet) { board.run("Quiet mode", "quiet-mode on") }
                    }
                    .help("Quiet: lights off and slower clocks, for nights without a fan. Full power: everything at full speed.")
                }
            }
            if !board.actionNote.isEmpty {
                Text(board.actionNote).font(.serif(13, italic: true)).foregroundColor(.muted).lineLimit(1)
            }
        }
    }

    /// One side of the power-mode switch, drawn like the tabs; tapping the side that's already on does nothing.
    func mode(_ title: String, on: Bool, _ action: @escaping () -> Void) -> some View {
        Button { if !on { action() } } label: {
            VStack(alignment: .leading, spacing: 3) {
                Text(title).font(.serif(15, italic: on)).foregroundColor(on ? .inkBlue : .muted)
                Rectangle().fill(on ? Color.inkBlue : Color.clear).frame(height: 1.5)
            }
            .fixedSize()
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    /// Serif text tabs like the rest of the panel: the selected one is italic ink blue with a thin underline.
    var tabs: some View {
        HStack(spacing: 20) {
            tab("His brain", "brain")
            tab("Board log", "board")
            tab("What Claude sees", "claude")
            Spacer()
            // his brain and the chat in a window of its own, which you can resize (this panel can't be)
            Button {
                board.popOut.open(board)
            } label: {
                Text("Pop out \u{2197}").font(.serif(15, italic: true)).foregroundColor(.inkBlue)
            }
            .buttonStyle(.plain)
            .help("Open his brain and the chat in a window you can resize")
        }
    }

    func tab(_ title: String, _ key: String) -> some View {
        let on = board.shownView == key
        return Button { board.view = key } label: {
            VStack(alignment: .leading, spacing: 3) {
                Text(title).font(.custom(on ? "Palatino-Italic" : "Palatino", size: 16)).foregroundColor(on ? .inkBlue : .muted)
                Rectangle().fill(on ? Color.inkBlue : Color.clear).frame(height: 1.5)
            }
            .fixedSize()
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    /// The busiest programs in fixed columns, so names and numbers don't jump around.
    var busyRow: some View {
        HStack(spacing: 0) {
            if board.busiest.isEmpty {
                Text("nothing busy").font(.serif(14, italic: true)).foregroundColor(.muted)
            }
            ForEach(board.busiest, id: \.name) { p in
                HStack(spacing: 6) {
                    // the name gets first claim on the column; the spacer only takes what's left
                    Text(p.name).font(.serif(14)).lineLimit(1).truncationMode(.tail).layoutPriority(1)
                    Spacer(minLength: 4)
                    // Palatino's digits vary in width, so the number gets a fixed slot to keep it from jumping
                    Text(String(format: p.pct < 10 ? "%.1f%%" : "%.0f%%", p.pct)).font(.serif(14, italic: true)).foregroundColor(.inkBlue)
                        .frame(width: 42, alignment: .trailing)
                }
                // three columns across the panel's 568 pt, so Palatino names fit
                .foregroundColor(.muted).frame(width: 178, alignment: .leading).padding(.trailing, 10)
            }
            Spacer(minLength: 0)
        }
    }

    /// The console: lines under a rule like EQ's sections, one block of text so you can select across lines.
    /// His brain wraps, so his sentences fit; the logs scroll sideways like a terminal, so tables stay aligned.
    func log(_ lines: [String]) -> some View {
        LogText(lines: lines, font: .monospacedSystemFont(ofSize: 10, weight: .regular), wrap: board.shownView == "brain",
                spacing: 2, follow: $board.logAtBottom, jump: board.panelJump)
            .id(board.shownView)   // a fresh view per tab, since wrapping is set up once
            .frame(height: 220)
            .onChange(of: board.shownView) { _, _ in board.logAtBottom = true; board.panelJump += 1 }
            .overlay(alignment: .bottomTrailing) {
                JumpButton(show: !board.logAtBottom) { board.logAtBottom = true; board.panelJump += 1 }
            }
    }
}

/// "Jump to latest", shown while you're scrolled up in a log.
struct JumpButton: View {
    let show: Bool
    let action: () -> Void
    var body: some View {
        if show {
            Button(action: action) {
                Text("Jump to latest \u{2193}")
                    .font(.custom("Palatino-Italic", size: 13)).foregroundColor(.inkBlue)
                    .padding(.horizontal, 10).padding(.vertical, 4)
                    .background(Color.paper)
                    .overlay(Capsule().stroke(Color.line, lineWidth: 1))
                    .clipShape(Capsule())
            }
            .buttonStyle(.plain)
            .padding(6)
        }
    }
}

// MARK: - text you can select, and the chat box

/// A read-only log as one block of text, so you can drag across many lines, or ⌘A, and copy them all at once
/// (separate Text views only let you pick one line at a time). It holds still while you have something selected,
/// and follows new lines only while you're at the bottom.
struct LogText: NSViewRepresentable {
    let lines: [String]
    let font: NSFont
    let wrap: Bool
    let spacing: CGFloat
    @Binding var follow: Bool
    let jump: Int

    final class Coordinator: NSObject {
        var follow: Binding<Bool>
        var jump = 0
        var updating = false
        weak var scrollView: NSScrollView?
        init(_ follow: Binding<Bool>) { self.follow = follow }
        /// the view got its size, or the window was resized: stay on the newest line if you were following
        @objc func resized(_ n: Notification) {
            guard follow.wrappedValue, let sv = scrollView else { return }
            updating = true
            LogText.toBottom(sv)
            updating = false
        }
        /// you scrolled: follow again once you're back at the bottom, stop following when you scroll up
        @objc func scrolled(_ n: Notification) {
            guard !updating, let clip = n.object as? NSClipView, let doc = clip.documentView else { return }
            let bottom = clip.bounds.maxY >= doc.frame.height - 12
            if bottom != follow.wrappedValue { DispatchQueue.main.async { self.follow.wrappedValue = bottom } }
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator($follow) }

    func makeNSView(context: Context) -> NSScrollView {
        let sv = NSTextView.scrollableTextView()
        let tv = sv.documentView as! NSTextView
        tv.isEditable = false
        tv.isSelectable = true
        tv.drawsBackground = false
        tv.textContainerInset = NSSize(width: 0, height: 2)
        tv.textContainer?.lineFragmentPadding = 0
        sv.drawsBackground = false
        // no scroll bars, like before (the system's always-on ones are big white tracks on paper); the trackpad scrolls
        sv.hasVerticalScroller = false
        sv.hasHorizontalScroller = false
        if !wrap {
            tv.isHorizontallyResizable = true
            tv.maxSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: .greatestFiniteMagnitude)
            tv.textContainer?.widthTracksTextView = false
            tv.textContainer?.containerSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: .greatestFiniteMagnitude)
        }
        sv.contentView.postsBoundsChangedNotifications = true
        sv.contentView.postsFrameChangedNotifications = true
        NotificationCenter.default.addObserver(context.coordinator, selector: #selector(Coordinator.scrolled(_:)),
                                               name: NSView.boundsDidChangeNotification, object: sv.contentView)
        NotificationCenter.default.addObserver(context.coordinator, selector: #selector(Coordinator.resized(_:)),
                                               name: NSView.frameDidChangeNotification, object: sv.contentView)
        context.coordinator.scrollView = sv
        return sv
    }

    func updateNSView(_ sv: NSScrollView, context: Context) {
        let c = context.coordinator
        c.follow = $follow
        let tv = sv.documentView as! NSTextView
        let text = lines.map { $0.replacingOccurrences(of: " radxa-dragon-q6a ", with: " ") }.joined(separator: "\n")
        var changed = false
        // while you have something selected, new lines wait (replacing the text would drop your selection)
        if tv.string != text && tv.selectedRange().length == 0 {
            c.updating = true
            let keep = sv.contentView.bounds.origin
            tv.textStorage?.setAttributedString(styled(text))
            if !follow { sv.contentView.scroll(to: keep); sv.reflectScrolledClipView(sv.contentView) }
            c.updating = false
            changed = true
        }
        if follow && (changed || jump != c.jump) { c.updating = true; LogText.toBottom(sv); c.updating = false }
        c.jump = jump
    }

    /// the newest line, from the left edge (scrollToEndOfDocument also slides sideways to the last character)
    static func toBottom(_ sv: NSScrollView) {
        guard let doc = sv.documentView else { return }
        doc.layoutSubtreeIfNeeded()
        if let tv = doc as? NSTextView, let c = tv.textContainer { tv.layoutManager?.ensureLayout(for: c) }
        sv.contentView.scroll(to: NSPoint(x: 0, y: max(0, doc.frame.height - sv.contentView.bounds.height)))
        sv.reflectScrolledClipView(sv.contentView)
    }

    /// his replies and Claude's finished runs in ink blue, what you said in ink, everything else muted
    func styled(_ text: String) -> NSAttributedString {
        let para = NSMutableParagraphStyle()
        para.lineSpacing = spacing
        para.paragraphSpacing = wrap ? spacing : 0
        let out = NSMutableAttributedString()
        for (i, line) in text.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
            let color: Color = line.hasPrefix("✓ done") || line.contains("] replied") ? .inkBlue
                : line.contains("] heard:") ? .ink : .muted
            out.append(NSAttributedString(string: (i > 0 ? "\n" : "") + line,
                                          attributes: [.font: font, .foregroundColor: NSColor(color), .paragraphStyle: para]))
        }
        return out
    }
}

/// The chat box: Return sends, Shift-Return (or Option-Return) starts a new line, and pasting several lines keeps them.
final class ChatTextView: NSTextView {
    var onSend: (String) -> Void = { _ in }
    var placeholder = ""

    override func keyDown(with e: NSEvent) {
        let isReturn = e.keyCode == 36 || e.keyCode == 76
        if isReturn && e.modifierFlags.intersection([.shift, .option]).isEmpty && !hasMarkedText() {
            let t = string.trimmingCharacters(in: .whitespacesAndNewlines)
            if !t.isEmpty { onSend(t); string = ""; needsDisplay = true }
            return
        }
        super.keyDown(with: e)
    }

    override func didChangeText() { super.didChangeText(); needsDisplay = true }

    override func draw(_ rect: NSRect) {
        super.draw(rect)
        guard string.isEmpty else { return }
        placeholder.draw(at: NSPoint(x: textContainerInset.width + (textContainer?.lineFragmentPadding ?? 0),
                                     y: textContainerInset.height),
                         withAttributes: [.font: font ?? NSFont.systemFont(ofSize: 13),
                                          .foregroundColor: NSColor(Color.muted).withAlphaComponent(0.7)])
    }
}

struct ChatBox: NSViewRepresentable {
    let placeholder: String
    let size: CGFloat
    var focus = false
    let send: (String) -> Void

    func makeNSView(context: Context) -> NSScrollView {
        let sv = NSScrollView()
        sv.drawsBackground = false
        sv.hasVerticalScroller = true
        sv.autohidesScrollers = true
        let tv = ChatTextView(frame: .zero)
        tv.isRichText = false            // pasted text arrives plain
        tv.allowsUndo = true
        tv.drawsBackground = false
        tv.font = NSFont(name: "Palatino", size: size) ?? .systemFont(ofSize: size)
        tv.textColor = NSColor(Color.ink)
        tv.insertionPointColor = NSColor(Color.inkBlue)
        tv.textContainerInset = NSSize(width: 8, height: 7)
        tv.isVerticallyResizable = true
        tv.isHorizontallyResizable = false
        tv.autoresizingMask = [.width]
        tv.maxSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: .greatestFiniteMagnitude)
        tv.textContainer?.widthTracksTextView = true
        tv.placeholder = placeholder
        sv.documentView = tv
        if focus { DispatchQueue.main.async { tv.window?.makeFirstResponder(tv) } }
        return sv
    }

    func updateNSView(_ sv: NSScrollView, context: Context) {
        (sv.documentView as? ChatTextView)?.onSend = send
    }
}

/// The chat box with its frame and the note under it about the last message you sent.
struct TalkRow: View {
    @ObservedObject var board: Board
    let size: CGFloat
    let height: CGFloat
    var focus = false
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            ChatBox(placeholder: "Talk to him. Return sends, Shift-Return for a new line.", size: size, focus: focus) {
                board.tell($0)
            }
            .frame(height: height)
            .background(Color.paperDeep)
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.line, lineWidth: 1))
            if !board.chatNote.isEmpty {
                Text(board.chatNote).font(.serif(13, italic: true)).foregroundColor(.muted).lineLimit(1)
            }
        }
    }
}

// MARK: - the popped-out window

/// "Pop out": closes the menu bar panel and opens his brain and the chat in a normal window. While it's open the app
/// becomes a regular app (Dock icon, ⌘-Tab), so the window can't get lost behind others; closing it makes the app
/// menu-bar-only again. macOS remembers its size and place.
@MainActor final class PopOut: NSObject, NSWindowDelegate {
    private var window: NSWindow?

    func open(_ board: Board) {
        // the menu bar panel stays up on its own otherwise, on top of the new window
        NSApp.windows.first { String(describing: type(of: $0)).contains("MenuBarExtra") }?.close()
        if window == nil {
            let w = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 760, height: 640),
                             styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
            w.title = "Little Guy"
            w.isReleasedWhenClosed = false
            w.contentView = NSHostingView(rootView: BrainWindow(board: board))
            w.contentMinSize = NSSize(width: 420, height: 360)
            w.delegate = self
            if !w.setFrameUsingName("LittleGuyBrain") { w.center() }
            w.setFrameAutosaveName("LittleGuyBrain")
            window = w
        }
        board.popAtBottom = true
        NSApp.setActivationPolicy(.regular)
        window?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        DispatchQueue.main.async { board.popJump += 1 }   // once it has its size, show the newest line
    }

    func windowWillClose(_ n: Notification) {
        NSApp.setActivationPolicy(.accessory)
    }
}

/// His brain feed and the chat box in a window of its own, sized however you like (macOS remembers it).
struct BrainWindow: View {
    @ObservedObject var board: Board
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                Text("Little ").font(.custom("Palatino", size: 30)) + Text("Guy.").font(.custom("Palatino-Italic", size: 30)).foregroundColor(.inkBlue)
                Spacer()
                if let mood = board.status?.feed.last(where: { $0.hasPrefix("mood now") }) {
                    Text(mood).font(.serif(13, italic: true)).foregroundColor(.muted)
                }
            }
            Rectangle().fill(Color.line).frame(height: 1)
            LogText(lines: board.status?.feed ?? ["(can't reach his room yet)"], font: NSFont(name: "Palatino", size: 15) ?? .systemFont(ofSize: 15),
                    wrap: true, spacing: 5, follow: $board.popAtBottom, jump: board.popJump)
                .overlay(alignment: .bottomTrailing) {
                    JumpButton(show: !board.popAtBottom) { board.popAtBottom = true; board.popJump += 1 }
                }
            Rectangle().fill(Color.line).frame(height: 1)
            TalkRow(board: board, size: 15, height: 72, focus: true)
        }
        .padding(.horizontal, 24).padding(.vertical, 18)
        .frame(minWidth: 420, minHeight: 360)
        .background(Color.paper).foregroundColor(.ink)
    }
}

/// CPU load (ink blue) and the hottest core's temperature (faint clay, 30-90°C) over the last 90 seconds,
/// drawn like Headphone EQ's curve.
struct LoadLine: View {
    let history: [Int]
    let temps: [Int]

    func curve(_ values: [Int], in size: CGSize, scale: (Int) -> CGFloat) -> Path {
        Path { p in
            guard values.count > 1 else { return }
            let step = size.width / 89
            let x0 = size.width - step * CGFloat(values.count - 1)
            for (i, v) in values.enumerated() {
                let pt = CGPoint(x: x0 + step * CGFloat(i), y: size.height * (1 - min(max(scale(v), 0), 1)))
                i == 0 ? p.move(to: pt) : p.addLine(to: pt)
            }
        }
    }

    var body: some View {
        VStack(spacing: 6) {
            GeometryReader { g in
                ZStack {
                    ForEach([0.0, 0.5, 1.0], id: \.self) { f in
                        Rectangle().fill(Color.line).frame(height: 1).position(x: g.size.width / 2, y: g.size.height * f)
                    }
                    curve(temps, in: g.size) { CGFloat($0 - 30) / 60 }
                        .stroke(Color.clay.opacity(0.55), style: StrokeStyle(lineWidth: 1.5, lineCap: .round, lineJoin: .round, dash: [4, 3]))
                    curve(history, in: g.size) { CGFloat($0) / 100 }
                        .stroke(Color.inkBlue, style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))
                }
            }.frame(height: 90)
            HStack {
                Text("90 s ago").font(.serif(13, italic: true))
                Spacer()
                Text("CPU load").font(.serif(14, italic: true)).foregroundColor(.inkBlue)
                Text("·").font(.serif(14))
                Text("temperature").font(.serif(14, italic: true)).foregroundColor(.clay)
                Spacer()
                Text("now \(history.last ?? 0)% · \(temps.last ?? 0)°C").font(.serif(14))
            }.foregroundColor(.muted)
        }
    }
}

@main struct LittleGuyBar: App {
    @StateObject private var board = Board()
    var body: some Scene {
        MenuBarExtra {
            Console(board: board)
        } label: {
            // the dot, plus the hottest core's temperature when the board is reachable
            HStack(spacing: 3) {
                Image(nsImage: dot(board.dotColor))
                if board.online, let t = board.status?.cpu_temp { Text("\(t)°") }
            }
        }
        .menuBarExtraStyle(.window)
    }
}
