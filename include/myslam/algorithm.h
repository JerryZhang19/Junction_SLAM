//
// Created by gaoxiang on 19-5-4.
//

#ifndef MYSLAM_ALGORITHM_H
#define MYSLAM_ALGORITHM_H

// algorithms used in myslam
#include "myslam/common_include.h"
#include "myslam/feature.h"
#include  <map>
#include <vector>


namespace simpleslam {

    struct Vec2cmp
    {
        bool operator()(const Vec2& a, const Vec2& b) const {
            return a.x()<b.x()||(a.x()==b.x()&&a.y()<b.y());
        }
    };

    std::vector<std::shared_ptr<Junction2D>> EdgeToJunction(std::vector<std::pair<Vec2,Vec2>> edges)
    {
        std::map<Vec2,std::vector<Vec2>,Vec2cmp> vertexes;
        for (const auto &edge: edges)
        {
            if(vertexes.find(edge.first)==vertexes.end())
                vertexes[edge.first] = {edge.second};
            else
                vertexes[edge.first].push_back(edge.second);
            if(vertexes.find(edge.second)==vertexes.end())
                vertexes[edge.second] = {edge.first};
            else
                vertexes[edge.second].push_back(edge.first);
        }
        std::vector<std::shared_ptr<Junction2D>> Junctions;
        for(const auto& vertex:vertexes)
        {
            Junctions.push_back(Junction2D::Ptr (new Junction2D(NULL,vertex.first,vertex.second)));
        }
    }

    // converters
    inline Vec2 toVec2(const cv::Point2f p) { return Vec2(p.x, p.y); }

}  // namespace myslam

#endif  // MYSLAM_ALGORITHM_H
