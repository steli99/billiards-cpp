// Author: OpenAI ChatGPT (AI-generated reference implementation).
#pragma once
#include <array>
#include <deque>
#include <opencv2/core.hpp>
#include <string>
#include <vector>
namespace billiards {
struct Ball {
    cv::Point2f center;
    float radius = 0;
    cv::Rect box;
    int category = 3;
    float confidence = 0;
    int trackId = -1;
};
struct Table {
    std::array<cv::Point2f, 4> corners{};
    cv::Mat mask, homography;
    cv::Vec3f cloth{};
    bool valid = false;
};
struct Result {
    std::vector<Ball> balls;
    cv::Mat labels;
};
struct Track {
    int id = 0, category = 0, missed = 0;
    cv::Point2f position, velocity{};
    std::deque<cv::Point2f> trail;
};
class Analyzer {
  public:
    Table table;
    void initialize(const cv::Mat &frame);
    Result analyze(const cv::Mat &frame) const;
};
class Tracker {
    std::vector<Track> tracks_;
    int nextId_ = 1;

  public:
    void update(std::vector<Ball> &balls, float gate);
    const std::vector<Track> &tracks() const {
        return tracks_;
    }
};
cv::Point2f toMap(const cv::Point2f &p, const cv::Mat &homography);
cv::Mat minimap(const Table &, const std::vector<Ball> &, const std::vector<Track> &);
cv::Mat overlay(const cv::Mat &, const Table &, const Result &);
cv::Mat colorize(const cv::Mat &labels);
void saveResult(const std::string &directory, const std::string &name, const cv::Mat &,
                const Table &, const Result &);
struct Box {
    cv::Rect2f rect;
    int category = 0;
    float score = 1;
    std::string image;
};
std::vector<Box> readBoxes(const std::string &path, const std::string &image);
double boxIoU(const cv::Rect2f &, const cv::Rect2f &);
struct DetectionMetrics {
    std::array<double, 4> ap{};
    std::array<int, 4> positives{};
    double map = 0;
};
DetectionMetrics detectionMetrics(const std::vector<Box> &predictions,
                                  const std::vector<Box> &truth);
using Confusion = std::array<std::array<unsigned long long, 6>, 6>;
Confusion confusion(const cv::Mat &prediction, const cv::Mat &truth);
std::array<double, 6> ious(const Confusion &);
double meanIoU(const Confusion &);
} // namespace billiards
