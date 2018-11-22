#### Dependencies
This folder includes all dependencies required for successful compilation of CEF and projects consuming CEF.

#### Missing Dependencies
Due large binary dependencies and large amount of files following dependencies were removed and have to be installed manually:

##### CEF
`cmake/` - CMake files to compile the CEF application.

`include/`, `libcef_dll` and `Resources` - Operating System specific header-files to compile and execute CEF-based applications.
 
`Debug/` - Contains `libcef.so` and other components required to run the debug version of CEF-based applications. By default these files should be placed in the same directory as the executable and will be copied there as part of the build process.

`Release/` - Contains `libcef.so` and other components required to run the release version of CEF-based applications. By default these files should be placed in the same directory as the executable and will be copied there as part of the build process.

1) All those files can be be downloaded from [here](http://opensource.spotify.com/cefbuilds/index.html#linux64_builds) (Linux 64-Bit). Choose the `Standard Distribution`.

2) Extract the content somewhere on your computer and copy everything beside `cmake/cef_variables.cmake`

3) Open your project with an IDE and compile!

##### OpenCV
`OpenCV-3.4.1` - Contains OpenCV libary, which can/is consumed in the project. Please just run `OpenCV/installOpenCV-18-04.sh` (Caution: Ubuntu 18.04 only), beforehand make sure you have at least 10 GB of space. Depending on your computer, this may take several minutes. After installation you can remove following folders. 
