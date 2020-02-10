cmake_minimum_required(VERSION 3.12)

ADD_EXECUTABLE(MatchHistograms MatchHistograms.cxx)

set(ITK_DIR "/home/alexandre.bone/Softwares/ITK")
find_package(ITK REQUIRED)
target_link_libraries(MatchHistograms ${ITK_LIBRARIES})

#TARGET_LINK_LIBRARIES(MatchHistograms
#        #/Users/alexandre.bone/Softwares/ITK_build_FFT
#        /home/alexandre.bone/Softwares/ITK
#        )

install (TARGETS MatchHistograms DESTINATION bin)
