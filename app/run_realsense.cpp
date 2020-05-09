//
// Created by gaoxiang on 19-5-4.
//

#include <gflags/gflags.h>
#include "myslam/visual_odometry.h"


std::string config_file = "../config/default.yaml";

int main(int argc, char **argv) {


    simpleslam::VisualOdometry::Ptr vo(
        new simpleslam::VisualOdometry(config_file));
    assert(vo->Init() == true);

    vo->Run();

    return 0;
}
