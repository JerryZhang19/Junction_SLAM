//
// Created by gaoxiang on 19-5-2.
//
#pragma once

#ifndef MYSLAM_FEATURE_H
#define MYSLAM_FEATURE_H

#include <memory>
#include <opencv2/features2d.hpp>
#include "myslam/common_include.h"

namespace simpleslam {

struct Frame;
struct MapPoint;

/**
 * 2D Feature Point
 */
struct Feature {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Feature> Ptr;

    std::weak_ptr<Frame> frame_;         // frame that hold this feature
    cv::KeyPoint position_;              // 2D提取位置
    std::weak_ptr<MapPoint> map_point_;  // 关联地图点

    bool is_outlier_ = false;       // 是否为异常点
    double init_depth_=0;
    //bool is_on_left_image_ = true;  // 标识是否提在左图，false为右图

   public:
    Feature() {}

    Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint &kp)
        : frame_(frame), position_(kp) {}
    Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint &kp, double init_depth)
            : frame_(frame), position_(kp), init_depth_(init_depth){}
    Vec2 get_vec2(){return Vec2(position_.pt.x,position_.pt.y);}    
};
}  // namespace myslam


// TODO: fill in proper class elements
struct JunctionFeature{


};

#endif  // MYSLAM_FEATURE_H
