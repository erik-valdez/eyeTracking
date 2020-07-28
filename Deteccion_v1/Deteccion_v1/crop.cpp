
#include "crop.hpp"

cv::Rect crop(cv::Mat src) {
    
    
    int threshold_value = 3;
    int threshold_type = 0;
    int const max_binary_value = 255;
    cv::Mat graysrc,debugsrc,out;
    cv::cvtColor(src, graysrc, CV_BGR2GRAY);
    threshold(graysrc, debugsrc,threshold_value, max_binary_value, threshold_type );
    int Px=debugsrc.cols/2;
    int Py=debugsrc.rows/2;
    int value;
    int bValue=0;
    int up=0,down=debugsrc.cols,left=0,right=debugsrc.rows;
    
    std::cout<<"filas: "<<debugsrc.rows<<std::endl;
    std::cout<<"colum: "<<debugsrc.cols<<std::endl;
    
    
    for (int col = 0; col <debugsrc.cols; col++){
        
        value= (int)debugsrc.at<uchar>(Px,col);
        if (bValue==0&value==255) {
            up=col;
            //std::cout<<"up: "<<up<<std::endl;
        }
        
        if (bValue==255&value==0) {
            down=col;
            //std::cout<<"down: "<<down<<std::endl;
        }
        bValue=value;
    }
    bValue=0;
    for (int row = 0; row <debugsrc.rows; row++){
        
        value= (int)debugsrc.at<uchar>(row,Py);
        if (bValue==0&value==255) {
            left=row;
            //std::cout<<"left: "<<left<<std::endl;
        }
        
        if (bValue==255&value==0) {
            right=row;
            //std::cout<<"right: "<<right<<std::endl;
        }
        
        bValue=value;
    }
    
    
    cv::Rect outR=cv::Rect(up,left,down-up,right-left);
    
    //imshow("asdasd",src);//src(outR));
   // cvWaitKey(0);

    return outR;
}
