# Billiards Vision — Technical report

## Dataset and protocol

The [provided benchmark](https://drive.google.com/drive/folders/1dzNrhDpc2DXRqmQgbO5l2WMjzfhMdxVn) contains ten clips from four matches. All 20 supplied first/last images and their original bounding boxes and label masks were evaluated. All **1464 video frames** were processed. The original dataset is not redistributed; derived output images and videos are included. Dataset copyrights remain with the original owners.

Class IDs: 0 background; 1 cue ball; 2 black 8-ball; 3 solid; 4 stripe; 5 table. Prediction text files add confidence as a sixth column after x, y, width, height, class. Coordinates are native-image pixels. Position CSVs contain per-frame track ID, class, image position, minimap position and confidence. Minimap coordinates have an arbitrary 800×420-pixel reference, not physical units.

For each clip, table geometry is estimated from the first image and reused on the last image. For video, geometry is estimated from the first decoded frame. No annotations are passed to the analyzer: ground truth is opened only after predictions, inside the benchmark evaluator. Thresholds were inspected and adjusted during development on benchmark imagery; these are exploratory benchmark results, not an independently held-out evaluation. No claim of learned confidence calibration is made.

## Method

### Table and perspective

A weighted-by-count hue histogram over the central image estimates the dominant saturated cloth color. Circular hue distance, saturation and brightness produce a cloth mask. Closing/opening removes small holes and noise; the largest connected contour is converted to a convex hull and approximated with four vertices. Failure to find a sufficiently large quadrilateral produces an error rather than fabricated corners.

The four sides of this quadrilateral are the detected playing-field boundaries. Corners are cyclically ordered, with the longer opposing side pair mapped horizontally. A perspective transform maps them onto an assumed 2:1 table rectangle. Lens distortion, precise cushion offsets and the real camera calibration are not estimated. The camera is assumed stationary within each clip, as permitted by the example.

### Balls and categories

Pixels inside the table are compared against the estimated cloth hue, saturation and brightness. Morphology cleans the non-cloth mask. Connected components are filtered by area, approximate radius, circularity and circle coverage. Hough circles propose additional candidates when a ball is similar to the cloth; candidates still need non-cloth support, and spatial duplicates are suppressed.

A disk around each candidate measures light low-saturation pixels (including warm-tinted white), dark pixels, and colored pixels. Mostly light disks become cue balls; mostly dark disks become black balls; mixed white/colored disks become striped balls; remaining disks become solids. The resulting confidence is a heuristic ranking score. It is not a calibrated probability. A cue/8-ball uniqueness constraint is not imposed, so false duplicates can remain.

Segmentation starts with background 0 and table 5. Non-cloth pixels inside each accepted ball circle receive the ball's class. This avoids labeling an entire rectangular bounding box as a ball, but shadows, highlights and cloth-colored ball regions still cause under/over-segmentation.

### Tracking and minimap

Tracks predict the next center with smoothed velocity. Candidate associations are sorted by spatial distance plus a category-disagreement penalty, then accepted one-to-one within a motion gate. Tracks tolerate 12 missed frames; longer-missing tracks are retired and cannot be reassigned. Their stored paths remain visible on the final minimap. The trail is bounded to 2,000 points per track. IDs are local to a clip, not physical ball numbers.

This is a lightweight greedy association scheme. Fast shots, collisions, occlusions, false detections and inconsistent solid/stripe labels can cause fragmentation or identity switches. Missing detections are not declared pocketing events. Small stationary-ball jitter can produce short trails. No tracking ground truth is provided, so trajectory accuracy, IDF1 and pocketing accuracy are not claimed.

The video overlay shows detected boundaries and boxes, with a top-right minimap refreshed every frame. Cue balls are white, black balls dark, solids red, stripes yellow, and the table green. Trail colors use the track's original category; a reclassified detection can therefore differ from its trail color.

## Metrics

**mAP@0.5:** within each class, predictions are ranked by confidence and matched one-to-one to same-image, same-class ground-truth boxes at IoU ≥ 0.5. Duplicate predictions are false positives. AP integrates the all-points precision envelope over recall; mAP is the mean across the four ball classes. This is VOC-style all-points AP at one IoU threshold, not COCO mAP averaged over multiple thresholds.

**mIoU:** accumulate a 6×6 pixel confusion matrix; class IoU = TP/(TP+FP+FN). Dataset mIoU averages the six class IoUs computed from the accumulated counts. A class absent from both masks has undefined IoU and is omitted for that individual image; it is stored as `nan`. AP is also undefined for a class with no ground-truth instance in an individual image. All classes occur in the full dataset. Per-image scores are diagnostics and their mean is not the aggregate dataset score.

| Aggregate measurement | Value |
| --- | ---: |
| mAP@0.5 | 0.5334 |
| Six-class mIoU | 0.6170 |
| Cue-ball AP | 0.5313 |
| Black-ball AP | 0.7228 |
| Solid-ball AP | 0.3685 |
| Striped-ball AP | 0.5109 |

| Class | IoU |
| --- | ---: |
| Background | 0.9746 |
| Cue | 0.3659 |
| Black | 0.5873 |
| Solid | 0.3720 |
| Stripe | 0.4594 |
| Table | 0.9431 |

The strongest segmentation component is the table. Ball classification and small-object segmentation remain the main weaknesses. These results demonstrate a functioning classical baseline, not complete robustness across the benchmark.

## Per-image results

The table below includes every annotated image. Detailed class scores are in `results/benchmark/metrics.csv`. Detector counts include false positives; matching counts cannot be inferred solely from equal object counts.

| Image | GT balls | Detections | mAP@0.5 | mIoU |
| --- | ---: | ---: | ---: | ---: |
| game1_clip1/frame_first | 15 | 15 | 0.8452 | 0.6393 |
| game1_clip1/frame_last | 15 | 15 | 0.9539 | 0.7770 |
| game1_clip2/frame_first | 15 | 18 | 0.8149 | 0.7818 |
| game1_clip2/frame_last | 15 | 19 | 0.8090 | 0.7581 |
| game1_clip3/frame_first | 7 | 9 | 0.4167 | 0.5523 |
| game1_clip3/frame_last | 6 | 8 | 0.3750 | 0.5272 |
| game1_clip4/frame_first | 14 | 17 | 0.4669 | 0.6092 |
| game1_clip4/frame_last | 14 | 15 | 0.5135 | 0.5774 |
| game2_clip1/frame_first | 12 | 11 | 0.7679 | 0.6720 |
| game2_clip1/frame_last | 12 | 9 | 0.7857 | 0.6331 |
| game2_clip2/frame_first | 13 | 16 | 0.5690 | 0.5883 |
| game2_clip2/frame_last | 12 | 15 | 0.4635 | 0.6061 |
| game3_clip1/frame_first | 9 | 7 | 0.7083 | 0.6030 |
| game3_clip1/frame_last | 8 | 7 | 0.3906 | 0.4650 |
| game3_clip2/frame_first | 11 | 9 | 0.3750 | 0.5095 |
| game3_clip2/frame_last | 10 | 8 | 0.6806 | 0.6201 |
| game4_clip1/frame_first | 9 | 7 | 0.5556 | 0.5335 |
| game4_clip1/frame_last | 8 | 6 | 0.7407 | 0.6486 |
| game4_clip2/frame_first | 13 | 15 | 0.6488 | 0.6405 |
| game4_clip2/frame_last | 12 | 14 | 0.7698 | 0.6440 |

## Outputs and qualitative review for every clip

Each pair below shows bounding boxes and the predicted semantic mask. Raw 0–5 label masks and bounding-box predictions are also saved alongside these visualizations. The final minimap includes retained trajectories, including retired tracks; it is not an independently validated reconstruction.

### game1_clip1

The overhead view gives comparatively clear localization. Some stripes become solids and some solid balls become stripes; table boundaries include small cushion offsets.

Processed 187 frames at 29.970 fps. [Annotated video](../results/benchmark/game1_clip1/annotated.mp4).

**First frame**

![game1_clip1 frame_first boxes](../results/benchmark/game1_clip1/frame_first_boxes.jpg)

![game1_clip1 frame_first segmentation](../results/benchmark/game1_clip1/frame_first_segmentation.png)

**Last frame**

![game1_clip1 frame_last boxes](../results/benchmark/game1_clip1/frame_last_boxes.jpg)

![game1_clip1 frame_last segmentation](../results/benchmark/game1_clip1/frame_last_segmentation.png)

**Final top view and trajectories**

![game1_clip1 minimap](../results/benchmark/game1_clip1/final_minimap.png)

### game1_clip2

The perspective table is recovered well. Solid/striped confusion and dark pocket candidates remain; the final map shows moving-ball paths but also short spurious tracks.

Processed 157 frames at 29.970 fps. [Annotated video](../results/benchmark/game1_clip2/annotated.mp4).

**First frame**

![game1_clip2 frame_first boxes](../results/benchmark/game1_clip2/frame_first_boxes.jpg)

![game1_clip2 frame_first segmentation](../results/benchmark/game1_clip2/frame_first_segmentation.png)

**Last frame**

![game1_clip2 frame_last boxes](../results/benchmark/game1_clip2/frame_last_boxes.jpg)

![game1_clip2 frame_last segmentation](../results/benchmark/game1_clip2/frame_last_segmentation.png)

**Final top view and trajectories**

![game1_clip2 minimap](../results/benchmark/game1_clip2/final_minimap.png)

### game1_clip3

Cue-ball classification/localization fails on both annotated frames. Black-ball AP remains high; stripes and other balls are confused under this viewpoint.

Processed 158 frames at 29.970 fps. [Annotated video](../results/benchmark/game1_clip3/annotated.mp4).

**First frame**

![game1_clip3 frame_first boxes](../results/benchmark/game1_clip3/frame_first_boxes.jpg)

![game1_clip3 frame_first segmentation](../results/benchmark/game1_clip3/frame_first_segmentation.png)

**Last frame**

![game1_clip3 frame_last boxes](../results/benchmark/game1_clip3/frame_last_boxes.jpg)

![game1_clip3 frame_last segmentation](../results/benchmark/game1_clip3/frame_last_segmentation.png)

**Final top view and trajectories**

![game1_clip3 minimap](../results/benchmark/game1_clip3/final_minimap.png)

### game1_clip4

The player and cue overlap the cloth in the first frame. Cue-ball predictions fail on both evaluation frames; the quadrilateral still isolates most of the playing surface.

Processed 153 frames at 29.970 fps. [Annotated video](../results/benchmark/game1_clip4/annotated.mp4).

**First frame**

![game1_clip4 frame_first boxes](../results/benchmark/game1_clip4/frame_first_boxes.jpg)

![game1_clip4 frame_first segmentation](../results/benchmark/game1_clip4/frame_first_segmentation.png)

**Last frame**

![game1_clip4 frame_last boxes](../results/benchmark/game1_clip4/frame_last_boxes.jpg)

![game1_clip4 frame_last segmentation](../results/benchmark/game1_clip4/frame_last_segmentation.png)

**Final top view and trajectories**

![game1_clip4 minimap](../results/benchmark/game1_clip4/final_minimap.png)

### game2_clip1

Green cloth is detected without a hard-coded blue-only mask. Cue, black and stripe AP are strong here, but solid-ball classification is weak.

Processed 122 frames at 25.000 fps. [Annotated video](../results/benchmark/game2_clip1/annotated.mp4).

**First frame**

![game2_clip1 frame_first boxes](../results/benchmark/game2_clip1/frame_first_boxes.jpg)

![game2_clip1 frame_first segmentation](../results/benchmark/game2_clip1/frame_first_segmentation.png)

**Last frame**

![game2_clip1 frame_last boxes](../results/benchmark/game2_clip1/frame_last_boxes.jpg)

![game2_clip1 frame_last segmentation](../results/benchmark/game2_clip1/frame_last_segmentation.png)

**Final top view and trajectories**

![game2_clip1 minimap](../results/benchmark/game2_clip1/final_minimap.png)

### game2_clip2

The black ball is missed or misclassified at both endpoints. Multiple cue-like detections reduce cue AP; bright stripes are easier than dark balls.

Processed 133 frames at 25.000 fps. [Annotated video](../results/benchmark/game2_clip2/annotated.mp4).

**First frame**

![game2_clip2 frame_first boxes](../results/benchmark/game2_clip2/frame_first_boxes.jpg)

![game2_clip2 frame_first segmentation](../results/benchmark/game2_clip2/frame_first_segmentation.png)

**Last frame**

![game2_clip2 frame_last boxes](../results/benchmark/game2_clip2/frame_last_boxes.jpg)

![game2_clip2 frame_last segmentation](../results/benchmark/game2_clip2/frame_last_segmentation.png)

**Final top view and trajectories**

![game2_clip2 minimap](../results/benchmark/game2_clip2/final_minimap.png)

### game3_clip1

The oblique viewpoint compresses ball shapes. The final cue ball and solid classes are missed or misclassified, reducing both detection and segmentation scores.

Processed 116 frames at 24.859 fps. [Annotated video](../results/benchmark/game3_clip1/annotated.mp4).

**First frame**

![game3_clip1 frame_first boxes](../results/benchmark/game3_clip1/frame_first_boxes.jpg)

![game3_clip1 frame_first segmentation](../results/benchmark/game3_clip1/frame_first_segmentation.png)

**Last frame**

![game3_clip1 frame_last boxes](../results/benchmark/game3_clip1/frame_last_boxes.jpg)

![game3_clip1 frame_last segmentation](../results/benchmark/game3_clip1/frame_last_segmentation.png)

**Final top view and trajectories**

![game3_clip1 minimap](../results/benchmark/game3_clip1/final_minimap.png)

### game3_clip2

Table/background separation is stable, but cue detection differs between endpoints. Solid/stripe segmentation remains weak in the dark oblique view.

Processed 119 frames at 24.793 fps. [Annotated video](../results/benchmark/game3_clip2/annotated.mp4).

**First frame**

![game3_clip2 frame_first boxes](../results/benchmark/game3_clip2/frame_first_boxes.jpg)

![game3_clip2 frame_first segmentation](../results/benchmark/game3_clip2/frame_first_segmentation.png)

**Last frame**

![game3_clip2 frame_last boxes](../results/benchmark/game3_clip2/frame_last_boxes.jpg)

![game3_clip2 frame_last segmentation](../results/benchmark/game3_clip2/frame_last_segmentation.png)

**Final top view and trajectories**

![game3_clip2 minimap](../results/benchmark/game3_clip2/final_minimap.png)

### game4_clip1

Cue and black localization rank well, while stripes are frequently confused. No solid ground-truth ball exists in the final frame, hence undefined final solid AP.

Processed 156 frames at 29.943 fps. [Annotated video](../results/benchmark/game4_clip1/annotated.mp4).

**First frame**

![game4_clip1 frame_first boxes](../results/benchmark/game4_clip1/frame_first_boxes.jpg)

![game4_clip1 frame_first segmentation](../results/benchmark/game4_clip1/frame_first_segmentation.png)

**Last frame**

![game4_clip1 frame_last boxes](../results/benchmark/game4_clip1/frame_last_boxes.jpg)

![game4_clip1 frame_last segmentation](../results/benchmark/game4_clip1/frame_last_segmentation.png)

**Final top view and trajectories**

![game4_clip1 minimap](../results/benchmark/game4_clip1/final_minimap.png)

### game4_clip2

Most table pixels are correct, but small ball masks and category ambiguity reduce mIoU. Some dark candidates reduce first-frame black AP.

Processed 163 frames at 29.910 fps. [Annotated video](../results/benchmark/game4_clip2/annotated.mp4).

**First frame**

![game4_clip2 frame_first boxes](../results/benchmark/game4_clip2/frame_first_boxes.jpg)

![game4_clip2 frame_first segmentation](../results/benchmark/game4_clip2/frame_first_segmentation.png)

**Last frame**

![game4_clip2 frame_last boxes](../results/benchmark/game4_clip2/frame_last_boxes.jpg)

![game4_clip2 frame_last segmentation](../results/benchmark/game4_clip2/frame_last_segmentation.png)

**Final top view and trajectories**

![game4_clip2 minimap](../results/benchmark/game4_clip2/final_minimap.png)

## Build and validation

Compiled locally with GCC 13.3.0, CMake 4.4.3 and OpenCV 4.10.0, using C++17 and a Release build. `ctest` passes the algorithm test executable. The tests cover overlapping/disjoint box IoU, perfect AP, missing predictions, duplicate-detection penalties, per-image matching, absent classes, perfect and imperfect segmentation, invalid label rejection, tracking after reorder/short occlusion, a synthetic table with cue/black balls, and perspective corner mapping. These synthetic tests verify implementation behavior, not benchmark robustness.

The local OpenCV build lacks an FFmpeg video backend. For this experiment, the original MP4 streams were decoded into lossless PNG frame sequences using FFmpeg, then processed frame-by-frame by the unchanged C++ analyzer with the original frame rate supplied. No frame was skipped. C++ wrote MJPEG AVI outputs, which were transcoded to H.264 MP4 for compact distribution. The original annotated PNG images were evaluated directly. With a standard OpenCV build that supports MP4, the documented `video` and full `benchmark` commands accept the original clips directly; that direct MP4 backend path was not tested locally. The image-sequence alternative is `billiards video frames/%06d.png output 29.97` (use the source frame rate).

Output MP4 frame counts were checked against all ten source clips. All 20 frame predictions, ten final minimaps, ten annotated videos, position CSVs and metric rows are present. `results/verification.txt`, `results/tests.txt` and `results/run_metadata.json` record the checks, versions, source hashes and frame rates. The official Virtual Lab remains untested.

