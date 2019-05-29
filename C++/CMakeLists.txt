cmake_minimum_required (VERSION 2.8)

project(TestOpenVINO)

list (APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# environment variable InferenceEngine_DIR can be use instead of relaive path to specify location of configuration file
# get_filename_component(CMAKE_PREFIX_PATH "/home/johnfeng/intel/computer_vision_sdk/inference_engine/share" ABSOLUTE)
get_filename_component(CMAKE_PREFIX_PATH "/opt/intel/openvino/inference_engine/share" ABSOLUTE)

message(STATUS "Looking for inference engine configuration file at: ${CMAKE_PREFIX_PATH}")
find_package(InferenceEngine 1.4)

if (NOT InferenceEngine_FOUND)
    message(FATAL_ERROR "")
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE not defined, 'Release' will be used")
    set(CMAKE_BUILD_TYPE "Release")
endif()

if (NOT(BIN_FOLDER))
    if("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
        set (ARCH intel64)
    else()
        set (ARCH ia32)
    endif()

    set (BIN_FOLDER ${ARCH})
endif()

if (NOT (IE_MAIN_SOURCE_DIR))
    set(NEED_EXTENSIONS TRUE)
    if (WIN32)
        set (IE_MAIN_SOURCE_DIR ${CMAKE_SOURCE_DIR}/../bin/)
    else() 
        set (IE_MAIN_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}) # soure = project/build dir
    endif()
endif()

# message(STATUS "qq")
# message(STATUS ..${CMAKE_CURRENT_SOURCE_DIR})
# message(STATUS "qq")

if(NOT(UNIX))
    # here wont run
    set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${IE_MAIN_SOURCE_DIR}/${BIN_FOLDER})
    set (CMAKE_LIBRARY_PATH ${IE_MAIN_SOURCE_DIR}/${BIN_FOLDER})
    set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${IE_MAIN_SOURCE_DIR}/${BIN_FOLDER})
    set (CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${IE_MAIN_SOURCE_DIR}/${BIN_FOLDER})
    set (CMAKE_PDB_OUTPUT_DIRECTORY ${IE_MAIN_SOURCE_DIR}/${BIN_FOLDER})
    set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${IE_MAIN_SOURCE_DIR}/${BIN_FOLDER})
    set (LIBRARY_OUTPUT_DIRECTORY ${IE_MAIN_SOURCE_DIR}/${BIN_FOLDER})
    set (LIBRARY_OUTPUT_PATH ${LIBRARY_OUTPUT_DIRECTORY}) # compatibility issue: linux uses LIBRARY_OUTPUT_PATH, windows uses LIBRARY_OUTPUT_DIRECTORY
else ()
    set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${IE_MAIN_SOURCE_DIR})
    set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${IE_MAIN_SOURCE_DIR})
    set (CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${IE_MAIN_SOURCE_DIR})
    set (CMAKE_PDB_OUTPUT_DIRECTORY ${IE_MAIN_SOURCE_DIR})
    set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${IE_MAIN_SOURCE_DIR})
    set (LIBRARY_OUTPUT_DIRECTORY ${IE_MAIN_SOURCE_DIR})
    set (LIBRARY_OUTPUT_PATH ${LIBRARY_OUTPUT_DIRECTORY})
endif()

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
if (WIN32)
    if(NOT "${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
        message(FATAL_ERROR "Only 64-bit supported on Windows")
    endif()

    set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS _CRT_SECURE_NO_WARNINGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SCL_SECURE_NO_WARNINGS -DNOMINMAX")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc") #no asynchronous structured exception handling
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LARGEADDRESSAWARE")
    if (ENABLE_OMP)
        find_package(OpenMP)
        if (OPENMP_FOUND)
            set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        endif()
    endif()
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Werror=return-type ")
    if (APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=unused-command-line-argument")
    elseif(UNIX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wuninitialized -Winit-self -Wmaybe-uninitialized")
    endif()
endif()

include(feature_defs OPTIONAL)

####################################
## to use C++11
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
####################################

# set(GFLAGS_IS_SUBPROJECT TRUE)
# add_subdirectory(/home/johnfeng/inference_engine_samples/thirdparty/gflags)

# Properties->C/C++->General->Additional Include Directories
include_directories (
    ${InferenceEngine_INCLUDE_DIRS}
    ${InferenceEngine_INCLUDE_DIRS}/../samples/common
    ${InferenceEngine_INCLUDE_DIRS}/../samples/common/format_reader
    ${InferenceEngine_INCLUDE_DIRS}/../samples/common/samples
    # ${InferenceEngine_INCLUDE_DIRS}/../samples/build/thirdparty/gflags/include/gflags
    /home/john-server/inference_engine_samples_build/thirdparty/gflags/include/
    ${CMAKE_CURRENT_SOURCE_DIR}
)

include_directories(src) 
FILE(GLOB_RECURSE sourcefiles "src/*.cpp")

if (UNIX)
    SET(LIB_DL dl)
endif()

###################################################
# The list of samples which not require openCV.   #
# USE_OPENCV define is NOT set during their build #
# add_subdirectory(speech_sample)
###################################################

# Find OpenCV libray if exists
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})
if(OpenCV_FOUND)
    add_definitions(-DUSE_OPENCV)
endif()

# important part
link_directories(/home/john-server/inference_engine_samples_build/intel64/Release/lib)

LINK_LIBRARIES(${OpenCV_LIBRARIES}
               ${InferenceEngine_LIBRARIES}
               # format_reader
               # gflags
               gflags_nothreads
            #    ${LIB_DL}
            # dl 
               pthread
               IE::ie_cpu_extension
               )
            
# message(STATUS "qq")
# message(STATUS ${OpenCV_INCLUDE_DIRS})
# message(STATUS "qq")

add_executable(image_caption ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp ${sourcefiles})
