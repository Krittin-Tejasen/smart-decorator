# smart_decorator

A Flutter application that helps users visualize and design their room by scanning it with the phone camera, LiDAR (on supported iOS devices), or AR vision. Users can choose a room type, select a style and color, and generate a design.

## Features

- **Room Scanning** — Opens the phone camera to capture the room. Automatically uses LiDAR on supported iOS devices or AR Vision on all other devices.
- **Manual Builder** — Manually set room dimensions (width × length) as a fallback if camera scanning is not preferred.
- **Room Type Selection** — Choose between Living Room, Bedroom, and Dining Room.
- **Theme Selection** — Pick a color theme for your room design.
- **Upload Room Photo** — Upload an existing photo of your room.
- **Generate Design** — Generate a decoration design based on the scanned room and selected options.

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | Flutter (Dart) |
| State Management | Riverpod |
| Navigation | go_router |
| Camera / Image Capture | image_picker |
| HTTP Client | Dio |
| Fonts | Google Fonts |
| Animations | flutter_animate |
| Backend | FastAPI (Python) |
| Image Generation | Gemini / Replicate (nano-banana) |
| Furniture Segmentation | Grounded SAM 2 (Replicate) with Gemini Vision fallback |

## Project Structure

```
lib/
├── core/
│   ├── constants/            # App colors
│   ├── services/             # HardwareService, AI/API/Supabase services
│   └── theme/                # App theme
├── features/
│   ├── home/                 # Home screen
│   ├── splash/                # Splash screen
│   ├── processing/           # Processing screen + provider
│   ├── results/               # Results screen + widgets
│   ├── history/               # History screen
│   └── segmentation/          # (planned) Furniture segmentation UI
│       ├── presentation/      # segmentation_screen.dart
│       ├── widgets/           # segmentation_overlay.dart
│       └── providers/         # segmentation_provider.dart
├── shared/
│   ├── models/                # RoomType, DesignTheme, Furniture, Product, etc.
│   └── providers/             # app_state_provider.dart
├── routes/                    # go_router configuration
└── screens/
    ├── scan/                  # RoomScannerScreen
    ├── views/                 # LidarView, ARVisionView, ManualBuilderView
    └── widgets/                # Shared scan widgets (crosshair, floor plan mini)

backend/
├── main.py                    # FastAPI app — /generate-room, mounts segmentation router
├── segmentation.py            # /segment-furniture — Grounded SAM 2 / Gemini Vision detection
├── test_segmentation.py       # CLI script to test /segment-furniture against local images
└── requirements.txt           # Python dependencies

ios/
├── Runner/
│   ├── AppDelegate.swift             # App entry point, plugin registration
│   ├── SceneDelegate.swift           # Scene lifecycle
│   ├── ARCameraView.swift            # Native ARKit camera view (UiKitView bridge)
│   ├── Runner-Bridging-Header.h      # Objective-C / Swift bridge
│   ├── GeneratedPluginRegistrant.h/m # Auto-generated Flutter plugin registrant
│   ├── Info.plist                    # App permissions (camera, etc.)
│   ├── Assets.xcassets/              # App icons and launch images
│   └── Base.lproj/                   # Storyboards (LaunchScreen, Main)
├── Runner.xcodeproj/                 # Xcode project file
├── Runner.xcworkspace/               # Xcode workspace (used when building)
├── Flutter/                          # Flutter-generated Xcode configs
└── RunnerTests/                      # iOS unit tests

android/
├── app/
│   ├── build.gradle.kts              # App-level Gradle build config
│   └── src/
│       ├── main/
│       │   ├── AndroidManifest.xml   # App permissions (camera, etc.) + activities
│       │   ├── kotlin/com/example/smart_decorator/
│       │   │   └── MainActivity.kt   # Flutter activity entry point
│       │   ├── java/io/flutter/plugins/
│       │   │   └── GeneratedPluginRegistrant.java  # Auto-generated plugin registrant
│       │   └── res/                  # App icons (mipmap), styles, drawables
│       ├── debug/
│       │   └── AndroidManifest.xml   # Debug-only manifest overrides
│       └── profile/
│           └── AndroidManifest.xml   # Profile-mode manifest overrides
├── build.gradle.kts                  # Project-level Gradle build config
├── settings.gradle.kts               # Gradle settings (module declarations)
└── gradle/wrapper/                   # Gradle wrapper version config
```

## Furniture Segmentation

The backend exposes `POST /segment-furniture` (multipart image upload), which detects
furniture/decor items and returns each one with a labelled bounding box, a cropped
image, a per-item segmentation mask, dominant colours, and shape features — ready for
downstream product matching.

Pipeline (when `REPLICATE_API_TOKEN` is set):

```
Generated image → Grounding DINO → 2D bounding boxes → SAM 2 → one mask per furniture item
```

- **Grounding DINO** — open-vocabulary text-prompted detection, produces a labelled box per item.
- **SAM 2** — takes those boxes as prompts and returns one precise pixel mask per box.
- **Gemini Vision** — fallback used when Replicate isn't configured, or if either
  Replicate stage fails. It only produces boxes, so its masks are a rectangular
  approximation of the box (`mask_precise: false` on the returned item).

Each result is returned in the HTTP response **and** written to disk as JSON under
`backend/segmentation_results/` for reuse without re-running detection.

`/generate-room` also accepts a `segment=true` form field to run segmentation on the
generated image and include `furniture_segments` in the response.

The Flutter-side UI for displaying segmentation results (`segmentation_screen.dart`,
`segmentation_overlay.dart`, `segmentation_provider.dart`) is still planned.

### Backend Setup

```bash
cd backend
python -m venv .venv
./.venv/Scripts/activate        # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
```

Create `backend/.env` with at least one provider key:

```
GEMINI_API_KEY=your-gemini-key
REPLICATE_API_TOKEN=your-replicate-token   # optional, enables Grounding DINO + SAM 2
```

Run the server:

```bash
uvicorn main:app --reload
```

Test segmentation against a local image:

```bash
python test_segmentation.py path/to/room.jpg
```

## Getting Started

### Prerequisites

- Flutter SDK `>=3.11.5` — install from [flutter.dev](https://flutter.dev/docs/get-started/install)
- Dart SDK `>=3.11.5` (bundled with Flutter)
- Android Studio or Xcode for device/emulator setup

### Installation

```bash
git clone https://github.com/Krittin-Tejasen/smart-decorator.git
cd smart-decorator
flutter pub get
flutter run
```

### Platform Notes

**iOS**
- Camera usage requires `NSCameraUsageDescription` — already configured in `ios/Runner/Info.plist`.
- LiDAR scanning is available on iPhone 12 Pro and later, iPad Pro (2020) and later.

**Android**
- Camera permission is declared in `android/app/src/main/AndroidManifest.xml`.
- All Android devices use AR Vision mode (LiDAR is iOS-only).

**Windows (development)**
- Enable Developer Mode (`Settings → System → For developers`) before running `flutter pub get`, as Flutter requires symlink support for plugins.
- Flutter SDK must be at `C:\flutter` or added to your system PATH.

## Contributing

This project is developed as a thesis project. Branches:
- `main` — stable releases
- `scanRoomSize` — room scanning feature development
- `appFlow` — app navigation and flow
- `backend` — backend integration
