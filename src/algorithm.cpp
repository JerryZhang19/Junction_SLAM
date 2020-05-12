//
// Created by jerry on 5/12/20.
//
#include"myslam/algorithm.h"

namespace simpleslam {

    std::vector<std::shared_ptr<Junction2D>> EdgeToJunction(std::vector<std::pair<Vec2, Vec2>> edges) {
        std::map<Vec2, std::vector<Vec2>, Vec2cmp> vertexes;
        for (const auto &edge: edges) {
            if (vertexes.find(edge.first) == vertexes.end())
                vertexes[edge.first] = {edge.second};
            else
                vertexes[edge.first].push_back(edge.second);
            if (vertexes.find(edge.second) == vertexes.end())
                vertexes[edge.second] = {edge.first};
            else
                vertexes[edge.second].push_back(edge.first);
        }
        std::vector<std::shared_ptr<Junction2D>> Junctions;
        for (const auto &vertex:vertexes) {
            Junctions.push_back(Junction2D::Ptr(new Junction2D(NULL, vertex.first, vertex.second)));
        }

        return Junctions;
    }

}