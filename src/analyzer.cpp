// Author: OpenAI ChatGPT (AI-generated reference implementation).
#include "billiards.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
namespace billiards {
namespace {
const std::array<cv::Scalar, 6> colors = {cv::Scalar(25, 25, 25),  cv::Scalar(245, 245, 245),
                                          cv::Scalar(10, 10, 10),  cv::Scalar(30, 50, 235),
                                          cv::Scalar(0, 210, 255), cv::Scalar(95, 155, 55)};
float hueDistance(float a, float b) {
    float d = std::abs(a - b);
    return std::min(d, 180 - d);
}
std::array<cv::Point2f, 4> orderCorners(std::vector<cv::Point> poly) {
    cv::Point2f center{};
    for (auto p : poly)
        center += cv::Point2f(p);
    center *= .25f;
    std::sort(poly.begin(), poly.end(), [&](auto a, auto b) {
        return std::atan2(a.y - center.y, a.x - center.x) <
               std::atan2(b.y - center.y, b.x - center.x);
    });
    auto start = std::min_element(poly.begin(), poly.end(),
                                  [](auto a, auto b) { return a.x + a.y < b.x + b.y; });
    std::rotate(poly.begin(), start, poly.end());
    std::array<cv::Point2f, 4> q;
    for (int i = 0; i < 4; ++i)
        q[i] = poly[i];
    // Map the longer average pair of opposing sides horizontally.
    if (cv::norm(q[1] - q[0]) + cv::norm(q[2] - q[3]) <
        cv::norm(q[2] - q[1]) + cv::norm(q[3] - q[0]))
        std::rotate(q.begin(), q.begin() + 1, q.end());
    return q;
}
} // namespace
void Analyzer::initialize(const cv::Mat &frame) {
    if (frame.empty() || frame.type() != CV_8UC3)
        throw std::runtime_error("Expected a BGR image");
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    std::array<double, 180> histogram{};
    for (int y = frame.rows / 6; y < frame.rows * 5 / 6; y += 2)
        for (int x = frame.cols / 6; x < frame.cols * 5 / 6; x += 2) {
            auto p = hsv.at<cv::Vec3b>(y, x);
            if (p[1] > 65 && p[2] > 45)
                for (int d = -3; d <= 3; ++d)
                    histogram[(p[0] + d + 180) % 180] += 1;
        }
    int hue =
        static_cast<int>(std::max_element(histogram.begin(), histogram.end()) - histogram.begin());
    cv::Mat cloth(frame.size(), CV_8U, cv::Scalar(0));
    for (int y = 0; y < frame.rows; ++y)
        for (int x = 0; x < frame.cols; ++x) {
            auto p = hsv.at<cv::Vec3b>(y, x);
            if (hueDistance(p[0], static_cast<float>(hue)) < 13 && p[1] > 65 && p[2] > 40)
                cloth.at<uchar>(y, x) = 255;
        }
    int k = std::max(3, (std::min(frame.rows, frame.cols) / 100) | 1);
    cv::morphologyEx(cloth, cloth, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, {k, k}));
    cv::morphologyEx(cloth, cloth, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5}));
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(cloth, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty())
        throw std::runtime_error("No cloth region found");
    auto biggest = *std::max_element(contours.begin(), contours.end(), [](auto &a, auto &b) {
        return cv::contourArea(a) < cv::contourArea(b);
    });
    if (cv::contourArea(biggest) < .10 * frame.total())
        throw std::runtime_error("Table region is too small");
    std::vector<cv::Point> hull, poly;
    cv::convexHull(biggest, hull);
    for (double epsilon = .01; epsilon <= .12; epsilon += .005) {
        cv::approxPolyDP(hull, poly, epsilon * cv::arcLength(hull, true), true);
        if (poly.size() == 4)
            break;
    }
    if (poly.size() != 4)
        throw std::runtime_error("Cannot estimate four table boundaries");
    table.corners = orderCorners(poly);
    table.mask = cv::Mat::zeros(frame.size(), CV_8U);
    cv::fillConvexPoly(table.mask, poly, 255);
    cv::Mat valid;
    cv::bitwise_and(cloth, table.mask, valid);
    auto mean = cv::mean(hsv, valid);
    table.cloth = cv::Vec3f(static_cast<float>(hue), static_cast<float>(mean[1]),
                            static_cast<float>(mean[2]));
    std::array<cv::Point2f, 4> target = {cv::Point2f(20, 20), cv::Point2f(780, 20),
                                         cv::Point2f(780, 400), cv::Point2f(20, 400)};
    table.homography = cv::getPerspectiveTransform(table.corners.data(), target.data());
    table.valid = true;
}
Result Analyzer::analyze(const cv::Mat &frame) const {
    if (!table.valid || frame.size() != table.mask.size())
        throw std::runtime_error("Initialize table with matching image size first");
    cv::Mat hsv, gray;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::Mat inner;
    cv::erode(table.mask, inner, cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5}));
    cv::Mat fg = cv::Mat::zeros(frame.size(), CV_8U);
    for (int y = 0; y < frame.rows; ++y)
        for (int x = 0; x < frame.cols; ++x)
            if (inner.at<uchar>(y, x)) {
                auto p = hsv.at<cv::Vec3b>(y, x);
                bool different = hueDistance(p[0], table.cloth[0]) > 18 || p[1] < 65 ||
                                 p[2] < table.cloth[2] * .50 || p[2] > table.cloth[2] * 1.55;
                if (different)
                    fg.at<uchar>(y, x) = 255;
            }
    cv::morphologyEx(fg, fg, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3}));
    cv::morphologyEx(fg, fg, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3}));
    double area = cv::countNonZero(table.mask);
    float base = static_cast<float>(std::sqrt(area) / 55.0);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(fg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<std::pair<cv::Point2f, float>> candidates;
    for (auto &c : contours) {
        double a = cv::contourArea(c), per = cv::arcLength(c, true);
        cv::Point2f center;
        float r;
        cv::minEnclosingCircle(c, center, r);
        if (r < base * .38 || r > base * 1.85 || a < 10 || per <= 0)
            continue;
        if (4 * CV_PI * a / (per * per) < .40 || a / (CV_PI * r * r) < .40)
            continue;
        candidates.push_back({center, r});
    }
    // Hough candidates help recover balls whose color is close to the cloth.
    cv::Mat smooth;
    cv::GaussianBlur(gray, smooth, {5, 5}, 1.1);
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(smooth, circles, cv::HOUGH_GRADIENT, 1.2, base * 1.25, 100, 18,
                     std::max(3, int(base * .5)), std::max(5, int(base * 1.6)));
    for (auto c : circles) {
        int x = cvRound(c[0]), y = cvRound(c[1]);
        if (x < 0 || y < 0 || x >= inner.cols || y >= inner.rows || !inner.at<uchar>(y, x))
            continue;
        cv::Mat disk = cv::Mat::zeros(frame.size(), CV_8U);
        cv::circle(disk, {x, y}, cvRound(c[2] * .8), 255, -1);
        cv::Mat evidence;
        cv::bitwise_and(disk, fg, evidence);
        if (cv::countNonZero(evidence) < .22 * cv::countNonZero(disk))
            continue;
        bool duplicate = false;
        for (auto p : candidates)
            if (cv::norm(p.first - cv::Point2f(c[0], c[1])) < std::max(p.second, c[2]))
                duplicate = true;
        if (!duplicate)
            candidates.push_back({{c[0], c[1]}, c[2]});
    }
    Result result;
    result.labels = cv::Mat::zeros(frame.size(), CV_8U);
    result.labels.setTo(5, table.mask);
    for (auto candidate : candidates) {
        auto center = candidate.first;
        float r = candidate.second;
        cv::Rect box(cvFloor(center.x - r), cvFloor(center.y - r), cvCeil(2 * r + 1),
                     cvCeil(2 * r + 1));
        box &= cv::Rect(0, 0, frame.cols, frame.rows);
        int total = 0, white = 0, dark = 0, color = 0;
        cv::Mat ballmask = cv::Mat::zeros(frame.size(), CV_8U);
        for (int y = box.y; y < box.y + box.height; ++y)
            for (int x = box.x; x < box.x + box.width; ++x) {
                if (cv::norm(cv::Point2f(static_cast<float>(x), static_cast<float>(y)) - center) >
                        r * .85 ||
                    !inner.at<uchar>(y, x))
                    continue;
                auto p = hsv.at<cv::Vec3b>(y, x);
                ++total;
                if ((p[1] < 90 && p[2] > 145) ||
                    (p[0] >= 15 && p[0] <= 40 && p[1] < 150 && p[2] > 170))
                    ++white;
                if (p[2] < 70)
                    ++dark;
                if (p[1] > 80 && p[2] > 70)
                    ++color;
            }
        if (total == 0)
            continue;
        float wf = float(white) / total, df = float(dark) / total, cf = float(color) / total;
        int category = 3;
        float confidence = .55f;
        if (wf > .70) {
            category = 1;
            confidence = wf;
        } else if (df > .55) {
            category = 2;
            confidence = df;
        } else if (wf > .20 && cf > .12) {
            category = 4;
            confidence = std::min(.95f, .55f + wf * .5f);
        } else
            confidence = std::min(.95f, .5f + cf * .4f);
        Ball b{center, r, box, category, confidence, -1};
        result.balls.push_back(b);
        // Only foreground pixels inside the detected ball support are labeled as ball.
        cv::circle(ballmask, center, cvRound(r), 255, -1);
        cv::bitwise_and(ballmask, fg, ballmask);
        cv::bitwise_and(ballmask, table.mask, ballmask);
        result.labels.setTo(category, ballmask);
    }
    return result;
}
void Tracker::update(std::vector<Ball> &balls, float gate) {
    struct Edge {
        float cost;
        size_t t, b;
    };
    std::vector<Edge> edges;
    for (size_t t = 0; t < tracks_.size(); ++t)
        for (size_t b = 0; b < balls.size(); ++b) {
            if (tracks_[t].missed > 12)
                continue;
            float distance = static_cast<float>(
                cv::norm(tracks_[t].position + tracks_[t].velocity - balls[b].center));
            if (distance < gate)
                edges.push_back(
                    {distance + (tracks_[t].category == balls[b].category ? 0 : gate * .3f), t, b});
        }
    std::sort(edges.begin(), edges.end(), [](auto a, auto b) { return a.cost < b.cost; });
    std::vector<bool> usedT(tracks_.size()), usedB(balls.size());
    for (auto e : edges)
        if (!usedT[e.t] && !usedB[e.b]) {
            auto &t = tracks_[e.t];
            auto &b = balls[e.b];
            usedT[e.t] = usedB[e.b] = true;
            t.velocity = .5f * t.velocity + .5f * (b.center - t.position);
            t.position = b.center;
            t.missed = 0;
            b.trackId = t.id;
            if (t.trail.empty() || cv::norm(t.trail.back() - t.position) > 1.0)
                t.trail.push_back(t.position);
            if (t.trail.size() > 2000)
                t.trail.pop_front();
        }
    for (size_t t = 0; t < tracks_.size(); ++t)
        if (!usedT[t])
            ++tracks_[t].missed;
    // Retired tracks are not confused with confidently pocketed balls.
    // Keep retired histories for the final trajectory map; never reassociate them.
    for (size_t b = 0; b < balls.size(); ++b)
        if (!usedB[b]) {
            auto &ball = balls[b];
            Track t;
            t.id = nextId_++;
            t.category = ball.category;
            t.position = ball.center;
            t.trail.push_back(ball.center);
            ball.trackId = t.id;
            tracks_.push_back(t);
        }
}
cv::Point2f toMap(const cv::Point2f &p, const cv::Mat &h) {
    std::vector<cv::Point2f> a{p}, b;
    cv::perspectiveTransform(a, b, h);
    return b[0];
}
cv::Mat minimap(const Table &table, const std::vector<Ball> &balls,
                const std::vector<Track> &tracks) {
    cv::Mat map(420, 800, CV_8UC3, cv::Scalar(40, 70, 50));
    cv::rectangle(map, {20, 20}, {780, 400}, colors[5], -1);
    for (auto p :
         std::vector<cv::Point>{{20, 20}, {400, 20}, {780, 20}, {20, 400}, {400, 400}, {780, 400}})
        cv::circle(map, p, 12, {10, 10, 10}, -1);
    for (auto &t : tracks)
        for (size_t i = 1; i < t.trail.size(); ++i)
            cv::line(map, toMap(t.trail[i - 1], table.homography),
                     toMap(t.trail[i], table.homography), colors[t.category], 2, cv::LINE_AA);
    for (auto b : balls) {
        auto p = toMap(b.center, table.homography);
        cv::circle(map, p, 9, colors[b.category], -1, cv::LINE_AA);
        cv::circle(map, p, 9, {30, 30, 30}, 1);
        if (b.trackId >= 0)
            cv::putText(map, std::to_string(b.trackId), p + cv::Point2f(9, -7),
                        cv::FONT_HERSHEY_SIMPLEX, .35, {255, 255, 255}, 1);
    }
    return map;
}
cv::Mat colorize(const cv::Mat &labels) {
    cv::Mat out(labels.size(), CV_8UC3);
    for (int y = 0; y < labels.rows; ++y)
        for (int x = 0; x < labels.cols; ++x) {
            int k = labels.at<uchar>(y, x);
            auto c = colors.at(k);
            out.at<cv::Vec3b>(y, x) = cv::Vec3b(uchar(c[0]), uchar(c[1]), uchar(c[2]));
        }
    return out;
}
cv::Mat overlay(const cv::Mat &image, const Table &table, const Result &r) {
    cv::Mat out = image.clone();
    for (int i = 0; i < 4; ++i)
        cv::line(out, table.corners[i], table.corners[(i + 1) % 4], {0, 255, 255}, 2);
    for (auto b : r.balls) {
        cv::rectangle(out, b.box, colors[b.category], 2);
        cv::putText(out, std::to_string(b.category), b.box.tl() + cv::Point(0, -3),
                    cv::FONT_HERSHEY_SIMPLEX, .45, {255, 255, 255}, 1);
    }
    return out;
}
void saveResult(const std::string &dir, const std::string &name, const cv::Mat &image,
                const Table &table, const Result &r) {
    std::filesystem::create_directories(dir);
    std::string base = dir + "/" + name;
    cv::imwrite(base + "_boxes.jpg", overlay(image, table, r));
    cv::imwrite(base + "_labels.png", r.labels);
    cv::imwrite(base + "_segmentation.png", colorize(r.labels));
    std::ofstream f(base + "_predictions.txt");
    if (!f)
        throw std::runtime_error("Cannot create predictions file");
    for (auto b : r.balls)
        f << b.box.x << ' ' << b.box.y << ' ' << b.box.width << ' ' << b.box.height << ' '
          << b.category << ' ' << b.confidence << '\n';
}
} // namespace billiards
