# Copyright (c) 2014 The Chromium Embedded Framework Authors. All rights
# reserved. Use of this source code is governed by a BSD-style license that
# can be found in the LICENSE file.

#
# Source files.
#

# cefsimple sources.
set(CEFSIMPLE_SRCS
        datacrawler_main.cpp
  )

APPEND_PLATFORM_SOURCES(CEFSIMPLE_SRCS)
source_group(cefsimple FILES ${CEFSIMPLE_SRCS})

APPEND_PLATFORM_SOURCES(CEFSIMPLE_RES_SRCS)
source_group(cefsimple\\\\res FILES ${CEFSIMPLE_RES_SRCS})

set(CEFSIMPLE_SRCS
  ${CEFSIMPLE_SRCS}
  ${CEFSIMPLE_RES_SRCS}
        datacrawler.cpp
        datacrawler.h
        DatacrawlerConfiguration.cpp
        DatacrawlerConfiguration.h
        DataModulesEnum.h
        datamodule-configs/DataModuleBaseConfiguration.cpp
        datamodule-configs/DataModuleBaseConfiguration.h
        datamodule-configs/ScreenshotConfiguration.cpp
        datamodule-configs/ScreenshotConfiguration.h
        datamodules/DataModuleBase.cpp
        datamodules/DataModuleBase.h
        graph/NodeElement.cpp
        graph/NodeElement.h
        datamodules/screenshot-datamodule/ScreenshotDataModule.cpp
        datamodules/screenshot-datamodule/ScreenshotDataModule.h
        util/Logger.cpp
        util/Logger.h
        util/LogLevel.h
        datamodules/screenshot-datamodule/ScreenshotClient.cpp
        datamodules/screenshot-datamodule/ScreenshotClient.h
        datamodules/screenshot-datamodule/ScreenshotHandler.cpp
        datamodules/screenshot-datamodule/ScreenshotHandler.h
        datamodules/DataBase.cpp
        datamodules/DataBase.h
        datamodules/screenshot-datamodule/ScreenshotData.cpp
        datamodules/screenshot-datamodule/ScreenshotData.h
        datamodules/screenshot-datamodule/Screenshot.cpp
        datamodules/screenshot-datamodule/Screenshot.h datamodules/screenshot-datamodule/ScreenshotRequestHandler.cpp datamodules/screenshot-datamodule/ScreenshotRequestHandler.h)

APPEND_PLATFORM_SOURCES(CEFSIMPLE_HELPER_SRCS)
source_group(cefsimple FILES ${CEFSIMPLE_HELPER_SRCS})

#
# Shared configuration.
#

# Target executable names.
set(CEF_TARGET "datacrawler")


# Logical target used to link the libcef library.
ADD_LOGICAL_TARGET("libcef_lib" "${CEF_LIB_DEBUG}" "${CEF_LIB_RELEASE}")

# Determine the target output directory.
SET_CEF_TARGET_OUT_DIR()


#
# Linux configuration.
#

if(OS_LINUX)
  # Executable target.
  add_executable(${CEF_TARGET} ${CEFSIMPLE_SRCS})
  SET_EXECUTABLE_TARGET_PROPERTIES(${CEF_TARGET})
  add_dependencies(${CEF_TARGET} libcef_dll_wrapper)
  target_link_libraries(${CEF_TARGET} libcef_lib libcef_dll_wrapper ${CEF_STANDARD_LIBS})

  # Add OpenCV-3.4.1
  SET(OpenCV_DIR "${CEF_ROOT}/OpenCV/installation/OpenCV-3.4.1/share/OpenCV/")
  find_package(OpenCV)
  target_link_libraries(${CEF_TARGET} ${OpenCV_LIBS})


  # Copy OpenCV resources to target
  SET(OPENCV_RESOURCE_DIR "${CEF_ROOT}/OpenCV/installation/OpenCV-3.4.1/lib/")
  SET(OPENCV_RESOURCE_FILES  libopencv_imgproc.so.3.4 libopencv_imgcodecs.so.3.4 libopencv_core.so.3.4)

  # Set rpath so that libraries can be placed next to the executable.
  set_target_properties(${CEF_TARGET} PROPERTIES INSTALL_RPATH "$ORIGIN")
  set_target_properties(${CEF_TARGET} PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
  set_target_properties(${CEF_TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CEF_TARGET_OUT_DIR})

  # Copy binary and resource files to the target output directory.
  COPY_FILES("${CEF_TARGET}" "${CEF_BINARY_FILES}" "${CEF_BINARY_DIR}" "${CEF_TARGET_OUT_DIR}")
  COPY_FILES("${CEF_TARGET}" "${CEF_RESOURCE_FILES}" "${CEF_RESOURCE_DIR}" "${CEF_TARGET_OUT_DIR}")
  COPY_FILES("${CEF_TARGET}" "${OPENCV_RESOURCE_FILES}" "${OPENCV_RESOURCE_DIR}" "${CEF_TARGET_OUT_DIR}")

  # Set SUID permissions on the chrome-sandbox target.
  SET_LINUX_SUID_PERMISSIONS("${CEF_TARGET}" "${CEF_TARGET_OUT_DIR}/chrome-sandbox")
endif()