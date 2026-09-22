// Author: OpenAI ChatGPT (AI-generated reference implementation).
#include "billiards.hpp"
#include <cmath>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
using namespace billiards;
void check(bool ok, const char *text) {
    if (!ok)
        throw std::runtime_error(text);
}
void near(double a, double b, const char *text) {
    check(std::abs(a - b) < 1e-6, text);
}
int main() {
    try {
        near(boxIoU({0, 0, 10, 10}, {5, 0, 10, 10}), 1.0 / 3, "IoU overlap");
        near(boxIoU({0, 0, 10, 10}, {20, 20, 10, 10}), 0, "IoU disjoint");
        std::vector<Box> gt = {{{0, 0, 10, 10}, 1, 1, "a"}, {{20, 0, 10, 10}, 1, 1, "a"}};
        auto perfect = detectionMetrics(gt, gt);
        near(perfect.map, 1, "Perfect AP");
        check(std::isnan(perfect.ap[1]), "Absent class is undefined");
        std::vector<Box> duplicate = {{{0, 0, 10, 10}, 1, .9f, "a"},
                                      {{0, 0, 10, 10}, 1, .8f, "a"},
                                      {{20, 0, 10, 10}, 1, .7f, "a"}};
        near(detectionMetrics(duplicate, gt).map, 5.0 / 6,
             "Duplicate detection must be a false positive");
        duplicate[0].image = "wrong";
        check(detectionMetrics(duplicate, gt).map < 1, "Matching must stay within an image");
        near(detectionMetrics({}, gt).map, 0, "Missing predictions AP");
        cv::Mat truth = (cv::Mat_<uchar>(2, 3) << 0, 1, 2, 3, 4, 5);
        near(meanIoU(confusion(truth, truth)), 1, "Perfect segmentation");
        cv::Mat bad = truth.clone();
        bad.at<uchar>(0, 0) = 6;
        bool throws = false;
        try {
            confusion(bad, truth);
        } catch (...) {
            throws = true;
        }
        check(throws, "Reject invalid labels");
        cv::Mat pred = truth.clone();
        pred.at<uchar>(0, 1) = 0;
        auto perClass = ious(confusion(pred, truth));
        near(perClass[0], .5, "Background IoU");
        near(perClass[1], 0, "Missed ball IoU");
        Tracker tracker;
        std::vector<Ball> balls = {{{10, 10}, 5, {5, 5, 10, 10}, 1, .9f, -1},
                                   {{100, 10}, 5, {95, 5, 10, 10}, 3, .9f, -1}};
        tracker.update(balls, 30);
        int first = balls[0].trackId, second = balls[1].trackId;
        std::swap(balls[0], balls[1]);
        balls[1].center.x += 8;
        tracker.update(balls, 30);
        check(balls[1].trackId == first && balls[0].trackId == second, "IDs survive reordering");
        std::vector<Ball> empty;
        tracker.update(empty, 30);
        tracker.update(balls, 30);
        check(balls[1].trackId == first, "Short missed detection preserves ID");
        cv::Mat scene(400, 700, CV_8UC3, cv::Scalar(25, 25, 25));
        cv::rectangle(scene, {70, 60}, {630, 340}, {160, 115, 30}, -1);
        cv::circle(scene, {200, 180}, 9, {240, 240, 240}, -1);
        cv::circle(scene, {450, 220}, 9, {10, 10, 10}, -1);
        Analyzer analyzer;
        analyzer.initialize(scene);
        check(analyzer.table.valid, "Synthetic table found");
        auto result = analyzer.analyze(scene);
        check(result.labels.at<uchar>(100, 100) == 5 && result.labels.at<uchar>(0, 0) == 0,
              "Table/background segmentation");
        bool cue = false, black = false;
        for (auto b : result.balls) {
            if (b.category == 1 && cv::norm(b.center - cv::Point2f(200, 180)) < 5)
                cue = true;
            if (b.category == 2 && cv::norm(b.center - cv::Point2f(450, 220)) < 5)
                black = true;
        }
        check(cue && black, "Synthetic cue and black balls detected");
        auto mapped = toMap(analyzer.table.corners[0], analyzer.table.homography);
        check(cv::norm(mapped - cv::Point2f(20, 20)) < .01, "Homography corner mapping");
        std::cout << "Passed: box IoU, AP matching/duplicates/absent classes, segmentation "
                     "IoU/validation, tracking/occlusion, synthetic table and balls, homography.\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
