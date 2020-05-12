add_library(simpleslam SHARED
        frame.cpp
        map_element.cpp
        map.cpp
        camera.cpp
        config.cpp
        feature.cpp
        frontend.cpp
        backend.cpp
        viewer.cpp
        visual_odometry.cpp
        dataset.cpp algorithm.cpp)

target_link_libraries(simpleslam
        ${THIRD_PARTY_LIBS})
