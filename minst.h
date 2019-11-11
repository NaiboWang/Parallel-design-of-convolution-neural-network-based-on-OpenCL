#ifndef __MINST_//如果没有编译过这个,编译,不然不编译
#define __MINST_
/*
MINST数据库是一个手写图像数据库，里面的结构数据详情请看：http://m.blog.csdn.net/article/details?id=53257185
*/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
//手写数字的图像的宽和高,还有他的内容
typedef struct MinstImg{
	int c;           // 图像宽,这里是28
	int r;           // 图像高,这里是28
	float* ImgData; // 图像数据二维动态数组,28X28的一维原始图像
}MinstImg;
//60000张原始图像
typedef struct MinstImgArr{
	int ImgNum;        // 存储图像的数目,这里训练集是60000,测试集为10000
	MinstImg* ImgPtr;  // 存储图像指针数组,每一张就是上面的28X28的结构体
}*ImgArr;              // 存储图像数据的数组,注意是指针类型
//用来存标签的结构体
typedef struct MinstLabel{
	int l;            // 输出标记的长,这里是10
	float* LabelData; // 输出标记数据,这里是10个元素,分别代表0到9,初始化的时候全部为0,这个图像对应的数字几就让相应位置的值为1.0
}MinstLabel;
//60000个标签
typedef struct MinstLabelArr{
	int LabelNum;//存储标签数目,这里训练集是60000,测试集为10000
	MinstLabel* LabelPtr;// 存储标签指针数组,每一张就是上面1个标签结构体
}*LabelArr;              // 存储图像标记的数组

LabelArr read_Lable(const char* filename); // 读入图像标记

ImgArr read_Img(const char* filename); // 读入图像

void save_Img(ImgArr imgarr,char* filedir); // 将图像数据保存成文件

#endif