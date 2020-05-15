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
            if(vertex.second.size()>=2 || vertexes[vertex.second[0]].size()==1) //not just an endpoint
                Junctions.push_back(Junction2D::Ptr(new Junction2D(NULL, vertex.first, vertex.second)));
        }

        return Junctions;
    }





    Vec3 GetOrientation(std::shared_ptr<Frame> frame, Vec2 center, Vec2 endpoint, std::shared_ptr<Camera> camera)
    {
        Vec3 center_world;
        double depth  = 0.001*frame->depth_.at<unsigned short>(cv::Point2d(center[0], center[1]));
        if(depth==0)
        {
            LOG(INFO)<<"SHOULD HANDLE THIS SITUATION";
            return Vec3(0, 0, 0);
        }
        center_world = camera->pixel2world(center,
                                     camera->pose_,
                                     depth);

        std::vector<Vec3> points3d;
        Vec3 pworld;
        for(int i=0;i<10;i++)
        {
            Vec2 middle_point = center/10*(10-i)+endpoint/10*i;
            double depth  = 0.001*frame->depth_.at<unsigned short>(cv::Point2d(middle_point[0], middle_point[1]));
            if(depth ==0 || depth >12)
                continue;
            pworld = camera->pixel2world(middle_point,
                                 camera->pose_,
                                depth);
            points3d.push_back(pworld);
        }
        if(points3d.size()==0) {
            LOG(INFO)<<"NOT EXPECTING THIS, CORRECT ERROR";
            return Vec3(0, 0, 0);
        }

        Eigen::MatrixXd M;
        M.resize(3,points3d.size());
        for(int i=0;i<points3d.size();i++)
            M.col(i) = points3d[i] - center_world;

        Eigen::Matrix3d H  = M*M.transpose();
        Eigen::EigenSolver<Eigen::Matrix3d> es(H);
        auto eigenvalues = es.eigenvalues();
        double lambda1 = eigenvalues[0].real();
        double lambda2 = eigenvalues[1].real();
        double lambda3 = eigenvalues[2].real();

        int index_max = 0;
        if(lambda2>lambda1)
            index_max=1;
        if(lambda3>lambda2&lambda3>lambda1)
            index_max = 2;
        if(lambda1+lambda2+lambda3==0)
            return Vec3(0,0,0);
        double ratio = eigenvalues[index_max].real() / lambda1+lambda2+lambda3;
        if(ratio<0.99)
            return Vec3(0,0,0);
        Vec3 v = es.eigenvectors().col(index_max).real();
        if(v.dot(*( points3d.end()-1)-center_world)<0)
            v=-v;
        return v;
    }


}