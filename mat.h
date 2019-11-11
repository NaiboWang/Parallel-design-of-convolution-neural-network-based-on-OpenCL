// ������ļ���Ҫ���ڹ��ڶ�ά��������Ĳ���
#pragma once//ֻ����һ��
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
#define full 0//��ȫ���
#define same 1//�����ͬ��С�����־��,MATLAB CONV����������������
#define valid 2//�������
#define JS 1 //��ʱ��� 1��ʱ 0����ʱ
using namespace std;
typedef struct Mat2DSize {//��������С�Ľṹ��,c��r��ʾ����������
	int c; // ��������ȣ�
	int r; // �������߶ȣ�
}nSize;

float** rotate180(float** mat, nSize matSize);// ����ת180��

											  // ���������Ԫ�ض�Ӧλ�����,mat1��mat2�õ�res,�����С����
void addmat(float** res, float** mat1, nSize matSize1, float** mat2, nSize matSize2);// �������

float** correlation(float** map, nSize mapSize, float** inputData, nSize inSize, int type);// �����,Э����?

																						   // �������,map��������,mapSizeΪ����˴�С,inputData��Ҫ���������,inSize��Ҫ������ݵĴ�С,typeΪ�������
float** cov(float** map, nSize mapSize, float** inputData, nSize inSize, int type); // �������

																					// ����Ǿ�����ϲ�������ֵ�ڲ壩��upc��upr���ڲ屶��
float** UpSample(float** mat, nSize matSize, int upc, int upr);

// ����ά�����Ե��������addw��С��0ֵ��,������ȫ���
float** matEdgeExpand(float** mat, nSize matSize, int addc, int addr);

// ����ά�����Ե��С������shrinkc��С�ı�,������ȫ���֮��Ļ�ԭ
float** matEdgeShrink(float** mat, nSize matSize, int shrinkc, int shrinkr);

void savemat(float** mat, nSize matSize, const char* filename);// �����������

void multifactor(float** res, float** mat, nSize matSize, float factor);// �������ϵ��

float summat(float** mat, nSize matSize);// �����Ԫ�صĺ�

char * combine_strings(char *a, char *b);

char* intTochar(int i);

//����һ��ʱ�������,��������ĳ�ι��̷�����ʱ��,����str��������¼��ʾ��Ϣ���ַ���
class GTime
{
	timeval start, end;
	LARGE_INTEGER beginticks;
	LARGE_INTEGER endticks;
	LARGE_INTEGER  frequency;//�����ܼ�������Ƶ�ʣ�ÿ��357,9545��tick  �ҵ�INTEL T7500
	string str;
	FILE *f;
	double duration;
	int count;//д���������
public:
	GTime(string s, FILE *fp)
	{
		beginticks.QuadPart = 0;
		endticks.QuadPart = 0;
		frequency.QuadPart = 0;
		QueryPerformanceFrequency(&frequency);
		f = fp;
		startT(s);//��ʼ��ʱ
		count = 0;
	}
	double getDu()
	{
		return duration;
	}
	//��������ַ�������
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
	//������ʱ
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