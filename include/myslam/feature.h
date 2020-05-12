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
struct Junction3D;

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

    bool is_outlier_ = false;
    double init_depth_=0;

   public:
    Feature() {}

    Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint &kp)
        : frame_(frame), position_(kp) {}
    Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint &kp, double init_depth)
            : frame_(frame), position_(kp), init_depth_(init_depth){}
    Vec2 get_vec2(){return Vec2(position_.pt.x,position_.pt.y);}
};

/**
 * 2D Junction Feature
 */
struct Junction2D{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Junction2D> Ptr;

    std::weak_ptr<Frame> frame_;         // frame that hold this feature
    Vec2 position_;
    std::vector<Vec2> endpoints_;
    std::weak_ptr<Junction3D> junction3D_;  // associated junction in map

    bool is_outlier_ = false;

public:
    Junction2D() {}

    Junction2D(std::shared_ptr<Frame> frame, const Vec2 &position, const std::vector<Vec2>& endpoints)
    : frame_(frame), position_(position), endpoints_(endpoints)  {}

    Vec2 get_vec2(){return position_;}
};

};

#endif
