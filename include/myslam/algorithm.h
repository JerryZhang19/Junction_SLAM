//
// Created by Jianwei Zhang
//
#pragma once
#ifndef MYSLAM_ALGORITHM_H
#define MYSLAM_ALGORITHM_H

// algorithms used in myslam
#include "myslam/common_include.h"
#include "myslam/feature.h"
#include  <map>
#include <vector>
#include "myslam/frame.h"
#include "myslam/camera.h"


namespace simpleslam {

    struct Vec2cmp
    {
        bool operator()(const Vec2& a, const Vec2& b) const {
            return a.x()<b.x()||(a.x()==b.x()&&a.y()<b.y());
        }
    };

    std::vector<std::shared_ptr<Junction2D>> EdgeToJunction(std::vector<std::pair<Vec2,Vec2>> edges);
    // converters
    inline Vec2 toVec2(const cv::Point2f p) { return Vec2(p.x, p.y); }
    Vec3 GetOrientation(std::shared_ptr<Frame> frame, Vec2 center, Vec2 endpoint, std::shared_ptr<Camera> camera);

}  // namespace myslam

#endif  // MYSLAM_ALGORITHM_H
