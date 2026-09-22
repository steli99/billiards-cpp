// Author: OpenAI ChatGPT (AI-generated reference implementation).
#include "billiards.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
namespace billiards {
std::vector<Box> readBoxes(const std::string &path, const std::string &image) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("Cannot open annotations: " + path);
    std::vector<Box> boxes;
    std::string line;
    while (std::getline(in, line)) {
        if (line.find_first_not_of(" \t\r") == std::string::npos)
            continue;
        std::istringstream row(line);
        Box b;
        b.image = image;
        if (!(row >> b.rect.x >> b.rect.y >> b.rect.width >> b.rect.height >> b.category) ||
            b.rect.width <= 0 || b.rect.height <= 0 || b.category < 1 || b.category > 4)
            throw std::runtime_error("Invalid bounding-box annotation: " + path);
        boxes.push_back(b);
    }
    return boxes;
}
double boxIoU(const cv::Rect2f &a, const cv::Rect2f &b) {
    double intersection = (a & b).area(), un = a.area() + b.area() - intersection;
    return un > 0 ? intersection / un : 0;
}
DetectionMetrics detectionMetrics(const std::vector<Box> &predictions,
                                  const std::vector<Box> &truth) {
    DetectionMetrics out;
    out.ap.fill(std::numeric_limits<double>::quiet_NaN());
    double sum = 0;
    int present = 0;
    for (int category = 1; category <= 4; ++category) {
        std::vector<Box> gt, pred;
        for (auto b : truth)
            if (b.category == category)
                gt.push_back(b);
        for (auto b : predictions)
            if (b.category == category)
                pred.push_back(b);
        out.positives[category - 1] = static_cast<int>(gt.size());
        if (gt.empty())
            continue;
        std::stable_sort(pred.begin(), pred.end(),
                         [](auto &a, auto &b) { return a.score > b.score; });
        std::vector<bool> matched(gt.size());
        std::vector<double> recall{0}, precision{0};
        int tp = 0, fp = 0;
        for (auto b : pred) {
            double best = 0;
            int index = -1;
            for (size_t j = 0; j < gt.size(); ++j)
                if (!matched[j] && b.image == gt[j].image) {
                    double iou = boxIoU(b.rect, gt[j].rect);
                    if (iou > best) {
                        best = iou;
                        index = static_cast<int>(j);
                    }
                }
            if (index >= 0 && best >= .5) {
                matched[index] = true;
                ++tp;
            } else
                ++fp;
            recall.push_back(double(tp) / gt.size());
            precision.push_back(double(tp) / (tp + fp));
        }
        recall.push_back(1);
        precision.push_back(0);
        for (int i = static_cast<int>(precision.size()) - 2; i >= 0; --i)
            precision[i] = std::max(precision[i], precision[i + 1]);
        double ap = 0;
        for (size_t i = 1; i < recall.size(); ++i)
            ap += (recall[i] - recall[i - 1]) * precision[i];
        out.ap[category - 1] = ap;
        sum += ap;
        ++present;
    }
    out.map = present ? sum / present : std::numeric_limits<double>::quiet_NaN();
    return out;
}
Confusion confusion(const cv::Mat &prediction, const cv::Mat &truth) {
    if (prediction.size() != truth.size() || prediction.type() != CV_8U || truth.type() != CV_8U)
        throw std::runtime_error("Expected matching uint8 label masks");
    Confusion c{};
    for (int y = 0; y < truth.rows; ++y)
        for (int x = 0; x < truth.cols; ++x) {
            int a = truth.at<uchar>(y, x), b = prediction.at<uchar>(y, x);
            if (a > 5 || b > 5)
                throw std::runtime_error("Mask labels must be 0..5");
            ++c[a][b];
        }
    return c;
}
std::array<double, 6> ious(const Confusion &c) {
    std::array<double, 6> out{};
    for (int i = 0; i < 6; ++i) {
        unsigned long long row = 0, col = 0;
        for (int j = 0; j < 6; ++j) {
            row += c[i][j];
            col += c[j][i];
        }
        auto un = row + col - c[i][i];
        out[i] = un ? double(c[i][i]) / un : std::numeric_limits<double>::quiet_NaN();
    }
    return out;
}
double meanIoU(const Confusion &c) {
    double sum = 0;
    int n = 0;
    for (double iou : ious(c))
        if (std::isfinite(iou)) {
            sum += iou;
            ++n;
        }
    return n ? sum / n : std::numeric_limits<double>::quiet_NaN();
}
} // namespace billiards
