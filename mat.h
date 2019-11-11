// 这里库文件主要存在关于二维矩阵数组的操作
#pragma once//只定义一次
#ifndef __MAT_
#define __MAT_

#include <stdlib.h>
#include <string.h>
#include <string>
#include <stdio.h>
#include <math.h>
#include <random>
#include <Windows.h>
#include <time.h>
#include "opencl.hpp"
#define full 0//完全卷积
#define same 1//输出相同大小的那种卷积,MATLAB CONV函数有这三种类型
#define valid 2//正常卷积
#define JS 1 //计时标记 1计时 0不计时
using namespace std;
typedef struct Mat2DSize {//定义矩阵大小的结构体,c和r表示列数和行数
	int c; // 列数（宽度）
	int r; // 行数（高度）
}nSize;

float** rotate180(float** mat, nSize matSize);// 矩阵翻转180度

											  // 两个矩阵各元素对应位置相加,mat1加mat2得到res,矩阵大小不变
void addmat(float** res, float** mat1, nSize matSize1, float** mat2, nSize matSize2);// 矩阵相加

float** correlation(float** map, nSize mapSize, float** inputData, nSize inSize, int type);// 互相关,协方差?

																						   // 卷积操作,map代表卷积核,mapSize为卷积核大小,inputData是要卷积的数据,inSize是要卷积数据的大小,type为卷积类型
float** cov(float** map, nSize mapSize, float** inputData, nSize inSize, int type); // 卷积操作

																					// 这个是矩阵的上采样（等值内插），upc及upr是内插倍数
float** UpSample(float** mat, nSize matSize, int upc, int upr);

// 给二维矩阵边缘扩大，增加addw大小的0值边,用于完全卷积
float** matEdgeExpand(float** mat, nSize matSize, int addc, int addr);

// 给二维矩阵边缘缩小，擦除shrinkc大小的边,用于完全卷积之后的还原
float** matEdgeShrink(float** mat, nSize matSize, int shrinkc, int shrinkr);

void savemat(float** mat, nSize matSize, const char* filename);// 保存矩阵数据

void multifactor(float** res, float** mat, nSize matSize, float factor);// 矩阵乘以系数

float summat(float** mat, nSize matSize);// 矩阵各元素的和

char * combine_strings(char *a, char *b);

char* intTochar(int i);

//定义一个时间计算类,用来计算某段过程发生的时间,其中str是用来记录提示信息的字符串
class GTime
{
	timeval start, end;
	LARGE_INTEGER beginticks;
	LARGE_INTEGER endticks;
	LARGE_INTEGER  frequency;//高性能计数器的频率：每秒357,9545个tick  我的INTEL T7500
	string str;
	FILE *f;
	double duration;
	int count;//写入次数限制
public:
	GTime(string s, FILE *fp)
	{
		beginticks.QuadPart = 0;
		endticks.QuadPart = 0;
		frequency.QuadPart = 0;
		QueryPerformanceFrequency(&frequency);
		f = fp;
		startT(s);//开始计时
		count = 0;
	}
	double getDu()
	{
		return duration;
	}
	//设置输出字符串内容
	void setStr(string s)
	{
		str = s;
	}
	void startT(string s)
	{
		//if (count < 300)
		//{
		setStr(s);
		QueryPerformanceCounter(&beginticks);
		//}
	}
	void startT()
	{
		QueryPerformanceCounter(&beginticks);
	}
	//结束计时
	void endT()
	{
		//if (count < 300)
		//{
		duration = GetCostMillisecond();
		fprintf(f, "%s %lf ms\n", str.c_str(), duration);
		count++;
		//}
	}
	double GetCostMillisecond()
	{
		QueryPerformanceCounter(&endticks);
		unsigned long long cost = (unsigned long long)(endticks.QuadPart - beginticks.QuadPart);
		double millsecond = (double)cost*1000.0 / (double)frequency.QuadPart;
		return millsecond;
	}
	unsigned long long  GetFrequency()
	{
		return (unsigned long long)frequency.QuadPart;
	}
};
#endif