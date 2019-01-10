#include "ScreenshotHandler.h"

/**
 * ~ScreenshotHandler
 */
ScreenshotHandler::~ScreenshotHandler() {
    delete lastScreenshot;
    delete lastL1Norms;
}

/**
 * ScreenshotHandler - Initializies ScreenshotHandler
 *
 * @param countLastL1Norms represents the number of the last n screenshots for whom the average should
 * be calculated. The average is later used to calculate the deviation from the total average.
 * @param changePixelThreshold represents a threshold in percent. If the deviation between the average of the last n
 * and the total average of all screenshots falls below this threshold, a screenshot will be taken.
 * @param renderHeight represents the height of the screenshot.
 * @param renderWidth represents the width of the screenshot.
 *
 */
ScreenshotHandler::ScreenshotHandler(bool * quitMessageLoop, int countLastL1Norms, float changePixelThreshold, int renderHeight, int renderWidth) {
    logger = Logger::getInstance();

    this->renderHeight = renderHeight;
    this->renderWidth = renderWidth;

    mHasPainted = false;
    initialInvoke = true;
    this->quitMessageLoop = quitMessageLoop;

    lastScreenshot = nullptr;

    numInvokations = 0;
    sumL1Norm = 0;
    this->countLastL1Norms = countLastL1Norms;
    lastL1Norms = new int32_t[countLastL1Norms]{0};

    this->changePixelThreshold = changePixelThreshold;
}

/**
 * GetViewRect - Sets height and width of the given CefRect-instance
 * @param browser represents the current CefBrowser-instance
 * @param rect represents the CefRect-instance of the given CefBrowser-instance
 * @return This will return true.
 */
void ScreenshotHandler::GetViewRect(CefRefPtr<CefBrowser> browser, CefRect &rect) {
    rect = CefRect(0, 0, renderWidth, renderHeight);
}

/**
 * OnPaint - Logic for taking a screenshot of the website. Called if the @param buffer was updated. It is not longer called
 * being called. CefQuitMessageLoop() is called in following cases:
 *
 * (a) the deviation between total average of L1-norms and the last n screenshots falls below the specified threshold
 * (b) external: onPaint() was not called for over a specified time (ELAPSED_TIME_ONPAINT_TIMEOUT in ScreenshotDataModule.h)
 * (c) external: onPaint() timed-out (ONPAINT_TIMEOUT in ScreenshotDataModule.h)
 *
 * @param browser represents the current CefBrowser-instance.
 * @param type specifies if a pop-up or view has has been painted.
 * @param dirtyRects specifies the areas, which are 'dirty' and were repainted
 * @param buffer represents the raw BGRA buffer. The size of the buffer is 4 Bytes * height * width
 * @param width represents the width of the screenshot
 * @param height represents the height of the screenhot
 */
void ScreenshotHandler::OnPaint(CefRefPtr<CefBrowser> browser, PaintElementType type, const RectList &dirtyRects,
                                const void *buffer, int width, int height) {

    timeOnPaintInvoke = std::chrono::steady_clock::now();

    // hide vertical and horizontal scrollbars
    // we are doing it here, since the website has been loaded already
    if(initialInvoke) {
        CefRefPtr<CefFrame> mainFrame = browser.get()->GetMainFrame();
        mainFrame->ExecuteJavaScript("document.documentElement.style.overflow = 'hidden'; document.body.scroll = \"no\";",
                                 mainFrame->GetURL(), 0);
    }

    mHasPainted = true;

    if (!initialInvoke) {
        ++numInvokations;

        unsigned char *changeMatrix = calculateChangeMatrix(lastScreenshot, (unsigned char*) buffer, renderHeight, renderWidth);
        int32_t l1Norm = calculateL1Norm(changeMatrix, renderWidth, renderHeight);

        logger->info("Current L1-Norm: " + std::to_string(l1Norm));

        sumL1Norm += l1Norm;
        int64_t averageL1Norm = sumL1Norm / numInvokations;
        insertL1Norm(l1Norm);

        if (lastL1Norms[0] != 0) {
            int32_t averageLastL1Norms = 0;

            for (int i = 0; i < countLastL1Norms; i++)
                averageLastL1Norms += lastL1Norms[i];

            averageLastL1Norms = averageLastL1Norms / countLastL1Norms;

            if ((double) averageLastL1Norms / (double) averageL1Norm < changePixelThreshold) {
                logger->info("Less than " + std::to_string(changePixelThreshold * 100) +
                             "% of the pixels changed! Taking screenshot!");
                // lock variable from other threads such as timeout
                quitMessageLoopMutex.lock();
                *quitMessageLoop = true;
                quitMessageLoopMutex.unlock();
                return;
            }
        }

        delete lastScreenshot;
        delete changeMatrix;
    }

    lastScreenshot = new unsigned char[height * width * 4];
    memcpy(lastScreenshot, buffer, sizeof(unsigned char) * height * width * 4);
    initialInvoke = false;
}

/** calculateChangeMatrix - Calculates a change matrix between two matrices from R^(nxm). Each element of the change
 *  matrix represents a state and is mapped to 4 Bytes of each input matrices. The state is 1 if the selected 4 Bytes of
 *  both matrices differ from each other, else 0.
 *
 * @param firstMatrix represents the first matrix
 * @param secMatrix represents the second matrix
 * @param numCol represents the number of columns of the matrix (width of screenshot)
 * @param numRow represents the number of rows of the matrix (height of screenshot)
 * @return A matrix A ∈ R^(numCol x numRow), where a_i,j ∈ {0,1}.
 *         a_i,j = 1, if there was a change in this point between firstMatrix and secondMatrix
 *         a_i,j = 0, if there was no change.
 */
unsigned char *
ScreenshotHandler::calculateChangeMatrix(unsigned char *firstMatrix, unsigned char *secMatrix, int32_t numRow,
                                         int32_t numCol) {
    auto *changeMatrix = new unsigned char[numCol * numRow]{0};

    for (int32_t i = 0, k = 0; i < numRow * numCol * 4; i += 4, k++) {
        if (*(firstMatrix + i) != *(secMatrix + i) ||
            *(firstMatrix + i + 1) != *(secMatrix + i + 1) ||
            *(firstMatrix + i + 2) != *(secMatrix + i + 2) ||
            *(firstMatrix + i + 3) != *(secMatrix + i + 3)) {
            changeMatrix[k] = 1;
        }
    }

    return changeMatrix;
}

/**
 * calculateL1Norm - Calculates the L1-norm of a given matrix
 *
 * @param matrix represents the matrix the L1-norm to be calculated for
 * @param numCol represents the number of columns of the matrix
 * @param numRow represents the number of rows of the matrix
 * @return The L1-norm of the given matrix
 */
int32_t ScreenshotHandler::calculateL1Norm(unsigned char *matrix, int32_t numCol, int32_t numRow) {
    int64_t l1 = 0;

    for (int64_t i = 0; i < numRow * numCol; i++)
        l1 += abs((int32_t) *(matrix + i));

    return l1;
}

/**
 * insertL1Norm - Inserts into lastL1Norms in our case the last calculated L1-norm in a FiFo-manner.
 */
void ScreenshotHandler::insertL1Norm(int32_t value) {

    for (int i = 0; i < countLastL1Norms; i++) {
        lastL1Norms[i] = lastL1Norms[i + 1];
    }

    lastL1Norms[countLastL1Norms - 1] = value;
}

/**
 * getTimeSinceLastPaint - Calculate the elapsed time since the last onPain()-invoke
 * @return Elapsed time since the last onPaint()-invoke
 */
long ScreenshotHandler::getTimeSinceLastPaint(){
    std::chrono::steady_clock::time_point currentTime = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - timeOnPaintInvoke).count();
}

/**
 * hasPainted
 * @return True if onPaint() was invoked, else false.
 */
bool ScreenshotHandler::hasPainted() { return mHasPainted; }

/**
 * getScreenshot - Returns the last screenshot painted.
 *
 * @return Screenshot in size 4 Bytes * width of screenshot * height screenshot
 */
unsigned char* ScreenshotHandler::getScreenshot(){

    if(lastScreenshot == nullptr)
        throw "Fatal: The lastScreenshot is null! Stopping datacrawler!";

    auto* screenshot = new unsigned char[renderHeight * renderWidth * 4];

    // copying, since lastScreenshot will be cleaned after destructor call
    for(int i = 0; i < renderHeight * renderWidth * 4; i++){
        *(screenshot + i) = *(lastScreenshot + i);
    }

    return screenshot;
}

/** qetQuitMessageLoopMutex
 *
 * @return Mutex to lock variable for quitting the message loop.
 */
std::mutex& ScreenshotHandler::getQuitMessageLoopMutex() { return quitMessageLoopMutex;}