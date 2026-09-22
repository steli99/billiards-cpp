# Billiards Vision

C++17 and OpenCV project for detecting billiard balls, segmenting the table, tracking trajectories, and displaying a top-view minimap. Uses classical computer vision only.

## Build

Requires CMake 3.20+, a C++17 compiler, and OpenCV 4 with video support.

```bash
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

## Run

```bash
./build/billiards image frame.png results/image
./build/billiards video clip.mp4 results/video
./build/billiards benchmark data results/benchmark
```

On Windows with Visual Studio, use `build/Release/billiards.exe`.

Download the [benchmark dataset](https://drive.google.com/drive/folders/1dzNrhDpc2DXRqmQgbO5l2WMjzfhMdxVn) into `data/`, preserving its clip folders. Outputs include annotated videos, segmentation masks, ball positions, minimaps, and evaluation metrics.
