//
// Created by jerry on 5/9/20.
//

#ifndef SIMPLESLAM_OPTICALFLOWTRACKER_H
#define SIMPLESLAM_OPTICALFLOWTRACKER_H
#include "myslam/common_include.h"
#include <mutex>


using namespace cv;
namespace simpleslam {

    struct Junction2D;

    class OpticalFlowTracker {
    public:
        OpticalFlowTracker(
                const cv::Mat &img1_,
                const cv::Mat &img2_,
                const std::vector<cv::KeyPoint> &kp1_,
                std::vector<cv::KeyPoint> &kp2_,
                std::vector<bool> &success_,
                bool inverse_ = true, bool has_initial_ = false, bool track_edge_ = true) :
                img1(img1_), img2(img2_), kp1(kp1_), kp2(kp2_), success(success_), inverse(inverse_),
                has_initial(has_initial_), track_edge(track_edge_) {}

        void calculateOpticalFlow(const cv::Range &range);

    private:
        const cv::Mat &img1;
        const cv::Mat &img2;
        const std::vector<cv::KeyPoint> &kp1;
        std::vector<cv::KeyPoint> &kp2;
        std::vector<bool> &success;
        bool inverse = true;
        bool has_initial = false;
        bool track_edge = false;

        std::mutex ostream_mux;
        std::mutex mux1;
    };

    void OpticalFlowSingleLevel(
            const Mat &img1,
            const Mat &img2,
            const std::vector<cv::KeyPoint> &kp1,
            std::vector<cv::KeyPoint> &kp2,
            std::vector<bool> &success,
            bool inverse = false,
            bool has_initial_guess = false
    );

    void OpticalFlowMultiLevel(
            const Mat &img1,
            const Mat &img2,
            const std::vector<cv::KeyPoint> &kp1,
            std::vector<cv::KeyPoint> &kp2,
            std::vector<bool> &success,
            bool inverse = false,
            bool is_edge = false
    );

}
#endif //SIMPLESLAM_OPTICALFLOWTRACKER_H
