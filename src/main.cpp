#include "billiards.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <stdexcept>
namespace fs = std::filesystem;
using namespace billiards;
namespace {
std::vector<Box> asBoxes(const Result &r, const std::string &id) {
    std::vector<Box> out;
    for (auto b : r.balls)
        out.push_back({cv::Rect2f(b.box), b.category, b.confidence, id});
    return out;
}
cv::Mat readImage(const fs::path &p, int flags = cv::IMREAD_COLOR) {
    auto im = cv::imread(p.string(), flags);
    if (im.empty())
        throw std::runtime_error("Cannot read " + p.string());
    return im;
}
void writeMetrics(std::ostream &out, const std::string &id, const DetectionMetrics &d,
                  const Confusion &c) {
    out << id << ',' << d.map << ',' << meanIoU(c);
    for (auto a : d.ap)
        out << ',' << a;
    for (auto i : ious(c))
        out << ',' << i;
    out << '\n';
}
void video(const fs::path &input, const fs::path &out, double fpsOverride = 0) {
    fs::create_directories(out);
    cv::VideoCapture cap(input.string());
    if (!cap.isOpened())
        throw std::runtime_error("Cannot decode video " + input.string() +
                                 "; OpenCV needs a suitable video backend");
    cv::Mat frame;
    if (!cap.read(frame))
        throw std::runtime_error("Video contains no decodable frames");
    Analyzer analyzer;
    analyzer.initialize(frame);
    Tracker tracker;
    cv::VideoWriter writer;
    double fps = fpsOverride > 0 ? fpsOverride : cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0)
        fps = 30;
    int width = std::min(frame.cols, 960),
        height = cvRound(double(frame.rows) * width / frame.cols);
    width += width % 2;
    height += height % 2;
    writer.open((out / "annotated.avi").string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps,
                {width, height});
    if (!writer.isOpened())
        throw std::runtime_error("MJPEG output codec unavailable");
    std::ofstream positions(out / "positions.csv");
    positions << "frame,time_s,track_id,class,x,y,map_x,map_y,confidence\n";
    int index = 0;
    cv::Mat map;
    do {
        auto result = analyzer.analyze(frame);
        tracker.update(result.balls, std::max(25.f, frame.cols * .045f));
        map = minimap(analyzer.table, result.balls, tracker.tracks());
        auto annotated = overlay(frame, analyzer.table, result);
        cv::resize(annotated, annotated, {width, height});
        int mw = width / 3, mh = mw * 420 / 800;
        cv::Mat small;
        cv::resize(map, small, {mw, mh});
        small.copyTo(annotated(cv::Rect(width - mw - 6, 6, mw, mh)));
        writer.write(annotated);
        for (auto b : result.balls) {
            auto p = toMap(b.center, analyzer.table.homography);
            positions << index << ',' << index / fps << ',' << b.trackId << ',' << b.category << ','
                      << b.center.x << ',' << b.center.y << ',' << p.x << ',' << p.y << ','
                      << b.confidence << '\n';
        }
        ++index;
    } while (cap.read(frame));
    writer.release();
    cv::imwrite((out / "final_minimap.png").string(), map);
    std::ofstream summary(out / "video_summary.txt");
    summary << "Processed frames: " << index << "\nFPS: " << fps
            << "\nDuration seconds: " << index / fps << '\n';
    std::cout << input.filename().string() << ": " << index << " frames\n";
}
void benchmark(const fs::path &data, const fs::path &out, bool runVideos) {
    if (!fs::is_directory(data))
        throw std::runtime_error("Dataset directory missing");
    fs::create_directories(out);
    std::ofstream metrics(out / "metrics.csv");
    metrics << "image,mAP50,mIoU,AP_cue,AP_black,AP_solid,AP_stripe,IoU_background,IoU_cue,IoU_"
               "black,IoU_solid,IoU_stripe,IoU_table\n";
    metrics << std::setprecision(8);
    std::vector<fs::path> clips;
    for (auto &e : fs::directory_iterator(data))
        if (e.is_directory() && fs::exists(e.path() / "frames/frame_first.png"))
            clips.push_back(e.path());
    std::sort(clips.begin(), clips.end());
    if (clips.empty())
        throw std::runtime_error("No clip folders found");
    std::vector<Box> allPred, allTruth;
    Confusion total{};
    int failed = 0;
    for (auto clip : clips) {
        try {
            std::string name = clip.filename().string();
            Analyzer analyzer;
            analyzer.initialize(readImage(clip / "frames/frame_first.png"));
            for (auto frameName : {"frame_first", "frame_last"}) {
                std::string stem = frameName, id = name + "/" + stem;
                auto image = readImage(clip / "frames" / (stem + ".png"));
                auto r = analyzer.analyze(image);
                saveResult((out / name).string(), stem, image, analyzer.table, r);
                auto pred = asBoxes(r, id);
                auto truth =
                    readBoxes((clip / "bounding_boxes" / (stem + "_bbox.txt")).string(), id);
                auto c = confusion(
                    r.labels, readImage(clip / "masks" / (stem + ".png"), cv::IMREAD_GRAYSCALE));
                auto d = detectionMetrics(pred, truth);
                writeMetrics(metrics, id, d, c);
                allPred.insert(allPred.end(), pred.begin(), pred.end());
                allTruth.insert(allTruth.end(), truth.begin(), truth.end());
                for (int i = 0; i < 6; ++i)
                    for (int j = 0; j < 6; ++j)
                        total[i][j] += c[i][j];
                std::cout << id << " mAP50=" << d.map << " mIoU=" << meanIoU(c) << '\n';
            }
            if (runVideos) {
                auto input = clip / (name + ".mp4");
                if (!fs::exists(input))
                    input = clip / (name + ".avi");
                video(input, out / name);
            }
        } catch (const std::exception &e) {
            ++failed;
            std::cerr << clip.filename().string() << ": " << e.what() << '\n';
        }
    }
    writeMetrics(metrics, "ALL", detectionMetrics(allPred, allTruth), total);
    if (failed)
        throw std::runtime_error(std::to_string(failed) +
                                 " clip(s) failed; results are incomplete");
}
} // namespace
int main(int argc, char **argv) {
    cv::setNumThreads(2);
    try {
        if (argc < 4) {
            std::cout
                << "Usage:\n  billiards image INPUT OUTPUT_DIR\n  billiards video INPUT "
                   "OUTPUT_DIR\n  billiards benchmark DATASET_DIR OUTPUT_DIR [--images-only]\n";
            return 1;
        }
        std::string mode = argv[1];
        fs::path input = argv[2], out = argv[3];
        if (mode == "image") {
            auto image = readImage(input);
            Analyzer a;
            a.initialize(image);
            auto r = a.analyze(image);
            saveResult(out.string(), input.stem().string(), image, a.table, r);
            cv::imwrite((out / "minimap.png").string(), minimap(a.table, r.balls, {}));
        } else if (mode == "video") {
            double fps = 0;
            if (argc == 5) {
                fps = std::stod(argv[4]);
                if (!std::isfinite(fps) || fps <= 0)
                    throw std::runtime_error("FPS must be positive");
            }
            if (argc > 5)
                throw std::runtime_error("Too many video arguments");
            video(input, out, fps);
        } else if (mode == "benchmark") {
            if (argc > 5 || (argc == 5 && std::string(argv[4]) != "--images-only"))
                throw std::runtime_error("Unknown benchmark option");
            benchmark(input, out, argc != 5);
        } else
            throw std::runtime_error("Unknown command: " + mode);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 2;
    }
}
