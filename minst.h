#ifndef __MINST_//���û�б�������,����,��Ȼ������
#define __MINST_
/*
MINST���ݿ���һ����дͼ�����ݿ⣬����Ľṹ���������뿴��http://m.blog.csdn.net/article/details?id=53257185
*/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
//��д���ֵ�ͼ��Ŀ�͸�,������������
typedef struct MinstImg{
	int c;           // ͼ���,������28
	int r;           // ͼ���,������28
	float* ImgData; // ͼ�����ݶ�ά��̬����,28X28��һάԭʼͼ��
}MinstImg;
//60000��ԭʼͼ��
typedef struct MinstImgArr{
	int ImgNum;        // �洢ͼ�����Ŀ,����ѵ������60000,���Լ�Ϊ10000
	MinstImg* ImgPtr;  // �洢ͼ��ָ������,ÿһ�ž��������28X28�Ľṹ��
}*ImgArr;              // �洢ͼ�����ݵ�����,ע����ָ������
//�������ǩ�Ľṹ��
typedef struct MinstLabel{
	int l;            // �����ǵĳ�,������10
	float* LabelData; // ����������,������10��Ԫ��,�ֱ����0��9,��ʼ����ʱ��ȫ��Ϊ0,���ͼ���Ӧ�����ּ�������Ӧλ�õ�ֵΪ1.0
}MinstLabel;
//60000����ǩ
typedef struct MinstLabelArr{
	int LabelNum;//�洢��ǩ��Ŀ,����ѵ������60000,���Լ�Ϊ10000
	MinstLabel* LabelPtr;// �洢��ǩָ������,ÿһ�ž�������1����ǩ�ṹ��
}*LabelArr;              // �洢ͼ���ǵ�����

LabelArr read_Lable(const char* filename); // ����ͼ����

ImgArr read_Img(const char* filename); // ����ͼ��

void save_Img(ImgArr imgarr,char* filedir); // ��ͼ�����ݱ�����ļ�

#endif