
#include <iostream>
#include <opencv2/opencv.hpp>
#include "functions.hpp"
#include "crop.hpp"


//variables globales
std::string mainwindowName=" Vwntana Principal";
cv::Mat debugFrame,frame;

cv::Mat backgrund= cv::Mat::zeros(cv::Size(256,256),CV_8UC1);




int main() {
    cv::Mat grayFrame,testFrame;
    createCornerKernels();
    //frame=cv::imread("/Users/seddin/Documents/C++proyects/objetos proyecto de deteccion/im10.png");
    cv::ellipse(backgrund, cv::Point(113, 155.6), cv::Size(23.4, 15.2),43.0, 0.0, 360.0, cv::Scalar(255, 255, 255), -1);
    
    cv::VideoCapture video("/Users/seddin/Documents/C++proyects/objetos proyecto de deteccion/V3.mp4");
    if( !video.isOpened() ) {
        std::cout<<"no se abrio el video"<<std::endl;
    }
    video>>testFrame;
    cv::Rect trimp=crop(testFrame);
    while(true){
    video>>testFrame;
    frame=testFrame(trimp);
    std::vector<cv::Mat> RGBcha(3);
    cv::split(frame, RGBcha);
    grayFrame=RGBcha[2];
    debugFrame=grayFrame;
//----------------------------------------------------------
    cv::Rect eyeRegion(0,0,frame.cols,frame.rows);
    cv::Point pupil=findEyesCenter(grayFrame);
    
    cv::Rect leftRightCornerRegion(eyeRegion);
    leftRightCornerRegion.width -=pupil.x;
    leftRightCornerRegion.x += pupil.x;
    cv::Rect downCornerRegion(eyeRegion);
    downCornerRegion.height -=pupil.y;
    downCornerRegion.y += pupil.y;
    
    rectangle(frame,leftRightCornerRegion,cv::Scalar(255,255,255),1,8,0);
    rectangle(frame,eyeRegion,cv::Scalar(255,255,255),1,8,0);
    rectangle(frame,downCornerRegion,cv::Scalar(255,255,255),1,8,0);
    
    pupil.x += eyeRegion.x;
    pupil.y += eyeRegion.y;
        circle(frame, pupil, 7, cv::Scalar(255,255,255),1, 8, 0);
    cv::imshow("debugframe",frame);
//---------------------------------------------------
if(cv::waitKey(1) >= 0) break;
    
}//endwhile
    return 0;
}
