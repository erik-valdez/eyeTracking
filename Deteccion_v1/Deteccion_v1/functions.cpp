
#include "functions.hpp"
//constantes
float kFastEyeWidth=50;
const int kumbralGradiente=50;
const int kWeightBlurSize = 5;
const bool kEnableWeight = true;
const float kWeightDivisor = 1.0;
const bool kEnablePostProcess = true;
const bool kPlotVectorField = false;
const float kPostProcessThreshold = 0.97;

cv::Mat *LeftCornerKernel,*RightCornerKernel;
float kEyeCornerKernel[4][6] = {{-1,-1,-1, 1, 1, 1},{-1,-1,-1,-1, 1, 1},{-1,-1,-1,-1, 0, 3},{ 1, 1, 1, 1, 1, 1},};

void createCornerKernels(){//--------------------------------------------------------------------------------------------------
    RightCornerKernel=new cv::Mat(4,6,CV_32F,kEyeCornerKernel);
    LeftCornerKernel = new cv::Mat(4,6,CV_32F);
    flip(*RightCornerKernel, *LeftCornerKernel, 1);
}//createcornerkernels



cv::Mat calcularGradiente(const cv::Mat &mat) {//--------------------------------------------------------------------------------------------
    
    cv::Mat out(mat.rows,mat.cols,CV_64F);
    
    for (int y = 0; y < mat.rows; ++y) {
        const uchar *Mr = mat.ptr<uchar>(y);
        double *Or = out.ptr<double>(y);
        
        Or[0] = Mr[1] - Mr[0];
        for (int x = 1; x < mat.cols - 1; ++x) {
            Or[x] = (Mr[x+1] - Mr[x-1])/2.0;
        }
        Or[mat.cols-1] = Mr[mat.cols-1] - Mr[mat.cols-2];
    }
    return out;
}//calculargradiente

cv::Mat magnitudMatriz(const cv::Mat &matX, const cv::Mat &matY) {//-----------------------------------------------------------------------
    cv::Mat mags(matX.rows,matX.cols,CV_64F);
    for (int y = 0; y < matX.rows; ++y) {
        const double *Xr = matX.ptr<double>(y), *Yr = matY.ptr<double>(y);
        double *Mr = mags.ptr<double>(y);
        for (int x = 0; x < matX.cols; ++x) {
            double gX = Xr[x], gY = Yr[x];
            double magnitude = sqrt((gX * gX) + (gY * gY));
            Mr[x] = magnitude;
        }
    }
    return mags;
}//margnitudMatriz

double calcularUmbralDinamico(const cv::Mat &mat, double stdDevFactor) {//---------------------------------------------------
    cv::Scalar stdMagnGrad, meanMagnGrad;
    meanStdDev(mat, meanMagnGrad, stdMagnGrad);
    double stdDev = stdMagnGrad[0] / sqrt(mat.rows*mat.cols);
    return stdDevFactor * stdDev + meanMagnGrad[0];
}//calcularumbraldinamico

void testPossibleCentersFormula(int x, int y, const cv::Mat &weight,double gx, double gy,cv::Mat &out) {//-------------------------
    for (int cy = 0; cy < out.rows; ++cy) {
        double *Or = out.ptr<double>(cy);
        const unsigned char *Wr = weight.ptr<unsigned char>(cy);
        for (int cx = 0; cx < out.cols; ++cx) {
            if (x == cx && y == cy) {
                continue;
            }
            double dx = x - cx;
            double dy = y - cy;
            double magnitude = sqrt((dx * dx) + (dy * dy));
            dx = dx / magnitude;
            dy = dy / magnitude;
            double dotProduct = dx*gx + dy*gy;
            dotProduct = std::max(0.0,dotProduct);
            if (kEnableWeight) {
                Or[cx] += dotProduct * dotProduct * (Wr[cx]/kWeightDivisor);
            } else {
                Or[cx] += dotProduct * dotProduct;
            }
        }
    }
}//testepossiblecentersformula

bool inMat(cv::Point p,int rows,int cols) {//--------------------------------------------------------------------------------------
    return p.x >= 0 && p.x < cols && p.y >= 0 && p.y < rows;
}
bool floodShouldPushPoint(const cv::Point &np, const cv::Mat &mat) {//---------------------------------------------------------
    return inMat(np, mat.rows, mat.cols);
}

cv::Mat floodKillEdges(cv::Mat &mat) {//--------------------------------------------------------------------------------------------
    rectangle(mat,cv::Rect(0,0,mat.cols,mat.rows),255);
    
    cv::Mat mask(mat.rows, mat.cols, CV_8U, 255);
    std::queue<cv::Point> toDo;
    toDo.push(cv::Point(0,0));
    while (!toDo.empty()) {
        cv::Point p = toDo.front();
        toDo.pop();
        if (mat.at<float>(p) == 0.0f) {
            continue;
        }
        cv::Point np(p.x + 1, p.y);
        if (floodShouldPushPoint(np, mat)) toDo.push(np);
        np.x = p.x - 1; np.y = p.y;
        if (floodShouldPushPoint(np, mat)) toDo.push(np);
        np.x = p.x; np.y = p.y + 1;
        if (floodShouldPushPoint(np, mat)) toDo.push(np);
        np.x = p.x; np.y = p.y - 1;
        if (floodShouldPushPoint(np, mat)) toDo.push(np);
        mat.at<float>(p) = 0.0f;
        mask.at<uchar>(p) = 0;
    }
    return mask;
}//floodkillingedges

cv::Point unscalePoint(cv::Point p, cv::Rect origSize) {//-------------------------------------------------------------------------
    float ratio = (((float)kFastEyeWidth)/origSize.width);
    int x = round(p.x / ratio);
    int y = round(p.y / ratio);
    return cv::Point(x,y);
}
//unsacalepoint

cv::Point findEyesCenter(cv::Mat eye ){//***********************
    cv::Mat eyeRze,weight,out;
    cv::resize(eye,eyeRze, cv::Size(kFastEyeWidth,kFastEyeWidth*eye.rows/eye.cols));
    
    const cv::Mat eyeRzeT=eyeRze.t();
    cv::Mat gradienteX= calcularGradiente(eyeRzeT.t());
    cv::Mat gradienteY= calcularGradiente(eyeRzeT).t();
    cv::Mat Magnitudes= magnitudMatriz(gradienteX,gradienteY);
    double umbralGradiente= calcularUmbralDinamico(Magnitudes, kumbralGradiente);
   
    
    for (int y = 0; y < eyeRze.rows; ++y) {
        double *Xr = gradienteX.ptr<double>(y), *Yr = gradienteY.ptr<double>(y);
        const double *Mr = Magnitudes.ptr<double>(y);
        for (int x = 0; x < eyeRze.cols; ++x) {
            double gX = Xr[x], gY = Yr[x];
            double magnitude = Mr[x];
            if (magnitude > umbralGradiente) {
                Xr[x] = gX/magnitude;
                Yr[x] = gY/magnitude;
            } else {
                Xr[x] = 0.0;
                Yr[x] = 0.0;
            }
        }
    }

    GaussianBlur( eyeRze, weight, cv::Size( kWeightBlurSize, kWeightBlurSize ), 0, 0 );
    for (int y = 0; y < weight.rows; ++y) {
        unsigned char *row = weight.ptr<unsigned char>(y);
        for (int x = 0; x < weight.cols; ++x) {
            row[x] = (255 - row[x]);
        }
    }
    cv::Mat outSum =cv::Mat::zeros(eyeRze.rows,eyeRze.cols,CV_64F);
    for (int y = 0; y < weight.rows; ++y) {
        const double *Xr = gradienteX.ptr<double>(y), *Yr = gradienteY.ptr<double>(y);
        for (int x = 0; x < weight.cols; ++x) {
            double gX = Xr[x], gY = Yr[x];
            if (gX == 0.0 && gY == 0.0) {
                continue;
            }
            testPossibleCentersFormula(x, y, weight, gX, gY, outSum);
        }
    }
    double numGradients = (weight.rows*weight.cols);
    outSum.convertTo(out, CV_32F,1.0/numGradients);
    cv::Point maxP;
    double maxVal;
    minMaxLoc(out, NULL,&maxVal,NULL,&maxP);
    if(kEnablePostProcess) {
        cv::Mat floodClone;
        double floodThresh = maxVal * kPostProcessThreshold;
        cv::threshold(out, floodClone, floodThresh, 0.0f, cv::THRESH_TOZERO);
        if(kPlotVectorField) {
            imwrite("eyeFrame.png",eye);
        }
        cv::Mat mask = floodKillEdges(floodClone);
        cv::minMaxLoc(out, NULL,&maxVal,NULL,&maxP,mask);
    }
    
    float ratio = (((float)kFastEyeWidth)/eye.cols);
    int x = round(maxP.x / ratio);
    int y = round(maxP.y / ratio);
    return cv::Point(x,y);
    
}//findeyecenter
