#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include "mat.h"

float** rotate180(float** mat, nSize matSize)// ����ת180��,����matΪ��λ����,matSizeΪ�����С
{
	int i, c, r;
	int outSizeW = matSize.c;//������
	int outSizeH = matSize.r;//����߶�
							 //��������������һ���������,���Ĵ�С��WxH�ľ���,ע��������WxH,��ô��ת180�Ⱥ������ȻҲ��WxH
	float** outputData = (float**)malloc(outSizeH * sizeof(float*));//����H��һά����
	for (i = 0; i<outSizeH; i++)
		outputData[i] = (float*)malloc(outSizeW * sizeof(float));//ÿ�з���W������
																 //�����ȴ洢,��ô��Ȼforѭ�����ȸ߶Ⱥ���,����˵���к���
	for (r = 0; r<outSizeH; r++)
		for (c = 0; c<outSizeW; c++)
			outputData[r][c] = mat[outSizeH - r - 1][outSizeW - c - 1];//��ת180��,һĿ��Ȼ

	return outputData;
}

// ���ھ������ز��������ѡ��
// ���ﹲ������ѡ��full��same��valid���ֱ��ʾ
// fullָ��ȫ�����������Ĵ�СΪinSize+(mapSize-1)
// sameָͬ�����ͬ��С
// validָ��ȫ������Ĵ�С��һ��ΪinSize-(mapSize-1)��С���䲻��Ҫ��������0����
// �����,mapΪ�����,mapSizeΪ����˴�С,inputDataΪҪ�����ͼ��,inSize��Ҫ���ͼ���С,type������˵�ľ������
/*����������Խ��иĽ�,û��Ҫ���Ȱ���������ȫ���֮���ٽ�ȡ����,�������˷�ʱ��,����valid��ʱ��ֱ�Ӱ���valid�����OK��
ʵ���϶������CNN���绹���Լ����Ż������ж�������������,�Ժ���˵*/
float** correlation(float** map, nSize mapSize, float** inputData, nSize inSize, int type)
{
	int i, j, c, r;
	int outSizeW = inSize.c + (mapSize.c - 1); // ������������һ����,��ȫ����õ��ľ��MAP�Ŀ��/����
	int outSizeH = inSize.r + (mapSize.r - 1);// ������������һ����,��ȫ����õ��ľ��MAP�ĸ߶�/����
	int newSize = outSizeW - 1 + mapSize.c;//exInputData��С
	float** outputData = (float**)malloc(outSizeH * sizeof(float*)); // ����صĽ��������,����outSizeH������
	for (i = 0; i<outSizeH; i++)
		outputData[i] = (float*)calloc(outSizeW, sizeof(float));//ÿ�������СΪoutSizeW

																// Ϊ�˷�����㣬��inputData����һȦ
	float** exInputData = matEdgeExpand(inputData, inSize, mapSize.c - 1, mapSize.r - 1);//����full�����ʱ��,��Ҫ�����ͼ���������˴�С-1�Ŀ��,���ܽ�����ȫ���
																						 //�����ĸ�forѭ������full�������
	for (j = 0; j<outSizeH; j++)//�������MAP��ÿһ��
		for (i = 0; i<outSizeW; i++)//�������MAP��ÿһ��
			for (r = 0; r<mapSize.r; r++)//���ھ���˵�ÿһ��
				for (c = 0; c<mapSize.c; c++) {//���ھ���˵�ÿһ��
					outputData[j][i] = outputData[j][i] + map[r][c] * exInputData[j + r][i + c];
					//outputData�ĵ�j�е�i�е�ֵ,���ھ���˵�r�е�c�е�ֵ��������֮��ԭʼͼ��ĵ�j+r�е�i+c�еĽ�����ܺ�,������˾������
				}

	//float** outputData = ocorrelation(map, mapSize.r, exInputData, newSize, type);//���������ԭʼͼ��;�����Լ�������ͽ��о������OPENCL
	//�ͷŲ��õ�����֮���exInputData����
	for (i = 0; i<inSize.r + 2 * (mapSize.r - 1); i++)
		free(exInputData[i]);
	free(exInputData);
	nSize outSize = { outSizeW,outSizeH };//�������MAP�Ĵ�С,Ĭ��Ϊfull���֮�����MAP�Ĵ�С
	return outputData;//ֱ�ӷ���
}
// �������,map��������,mapSizeΪ����˴�С,inputData��Ҫ���������,inSize��Ҫ������ݵĴ�С,typeΪ�������
float** cov(float** map, nSize mapSize, float** inputData, nSize inSize, int type)
{
	// ���������������ת180�ȵ�����ģ���������
	float** flipmap = rotate180(map, mapSize); //��ת180�ȵ�����ģ��,���߳�Ϊ�����
											   //float** res = ocorrelation(flipmap, mapSize.r, inputData, inSize.r, type);//���������ԭʼͼ��;�����Լ�������ͽ��о������
											   //float **res;
	float** res = correlation(flipmap, mapSize, inputData, inSize, type);//���������ԭʼͼ��;�����Լ�������ͽ��о������
	int i;
	//�����ͷ���ʱ�洢�ľ��MAP�Ŀռ�
	for (i = 0; i<mapSize.r; i++)
		free(flipmap[i]);
	free(flipmap);
	return res;
}

// ����Ǿ�����ϲ�������ֵ�ڲ壩��upc��upr����X�����Y������ڲ屶��,��CNN��������ֵ����2
float** UpSample(float** mat, nSize matSize, int upc, int upr)
{
	int i, j, m, n;
	int c = matSize.c;//ԭ����
	int r = matSize.r;//ԭ��������
	float** res = (float**)malloc((r*upr) * sizeof(float*)); // ����ĳ�ʼ��,����r*upr�е�����
	for (i = 0; i<(r*upr); i++)
		res[i] = (float*)malloc((c*upc) * sizeof(float));//ÿ�������СΪc*upc

	for (j = 0; j<r*upr; j = j + upr) {//���з�����,ÿ�����upr����ͬ��ֵ,ע�������Ǹ߶�,����һ��j����ԭʼmapһ�е�����,һ��forѭ��ִ����,����һ�е����ݾ���������
		for (i = 0; i<c*upc; i = i + upc)// �������,��x������ÿ��upc��ֵ�ı�һ�θ�ֵ
			for (m = 0; m<upc; m++)//ÿ�ζ�������upc��Ԫ�ظ�ֵ
				res[j][i + m] = mat[j / upr][i / upc];//�����
													  /*��������forʵ���˵�һ�е��з���������,Ȼ�����������for����������ĵ�һ������ڶ���֮������е���*/
		for (n = 1; n<upr; n++)      //  �ߵ�����,�ڶ��е����һ��
			for (i = 0; i<c*upc; i++)//�з����л�
				res[j + n][i] = res[j][i];//���ղŵ�һ�еĽ��
	}
	return res;
}

// ����ά�����Ե��������addw��С��0ֵ��
float** matEdgeExpand(float** mat, nSize matSize, int addc, int addr)
{ // ������Ե����,����addc��addr�ֱ�ΪҪ�ڿ�Ⱥ͸߶�,Ҳ�����к�������ĸ���,�㷨��addc=filterSizeX-1,addr=filterSizeY-1,filterSize�Ǿ���˴�С
	int i, j;
	int c = matSize.c;//ԭ��������
	int r = matSize.r;//ԭ��������
	float** res = (float**)malloc((r + 2 * addr) * sizeof(float*)); // ����ĳ�ʼ��,����r+2*addr������,�ϱ�addr��,�±�addr��
	for (i = 0; i<(r + 2 * addr); i++)
		res[i] = (float*)malloc((c + 2 * addc) * sizeof(float));//ÿ���������Ϊc+2*addc��,�������addc��,�ұ�addc��

	for (j = 0; j<r + 2 * addr; j++) {
		for (i = 0; i<c + 2 * addc; i++) {
			if (j<addr || i<addc || j >= (r + addr) || i >= (c + addc))//�������������ı�Ե��,����Ϊ0
				res[j][i] = (float)0.0;
			else
				res[j][i] = mat[j - addr][i - addc]; // ��Ȼ,����ԭ����������
		}
	}
	return res;
}

// ����ά�����Ե��С������shrinkc��С�ı�,����һ����������ȫ�෴�Ĳ���,��������
float** matEdgeShrink(float** mat, nSize matSize, int shrinkc, int shrinkr)
{ // ��������С������Сaddw������Сaddh
	int i, j;
	int c = matSize.c;
	int r = matSize.r;
	float** res = (float**)malloc((r - 2 * shrinkr) * sizeof(float*)); // �������ĳ�ʼ��
	for (i = 0; i<(r - 2 * shrinkr); i++)
		res[i] = (float*)malloc((c - 2 * shrinkc) * sizeof(float));
	for (j = 0; j<r; j++) {
		for (i = 0; i<c; i++) {
			if (j >= shrinkr&&i >= shrinkc&&j<(r - shrinkr) && i<(c - shrinkc))
				res[j - shrinkr][i - shrinkc] = mat[j][i]; // ����ԭ����������
		}
	}
	return res;
}

void savemat(float** mat, nSize matSize, const char* filename)//�������filename
{
	FILE  *fp = NULL;
	fp = fopen(filename, "wb");//��������ʽ����
	if (fp == NULL)
		printf("write file failed\n");
	int i;
	for (i = 0; i<matSize.r; i++)
		fwrite(mat[i], sizeof(float), matSize.c, fp);//�����ȴ洢
	fclose(fp);
}
// ���������Ԫ�ض�Ӧλ�����,mat1��mat2�õ�res,�����С����
void addmat(float** res, float** mat1, nSize matSize1, float** mat2, nSize matSize2)
{
	int i, j;
	if (matSize1.c != matSize2.c || matSize1.r != matSize2.r)//Ҫ��֤��������Ĵ�С��ͬ
		printf("ERROR: Size is not same!");

	for (i = 0; i<matSize1.r; i++)
		for (j = 0; j<matSize1.c; j++)
			res[i][j] = mat1[i][j] + mat2[i][j];//���Ȼ�󷵻ظ�res
}
// ��������Ԫ�س���ͬһϵ��
void multifactor(float** res, float** mat, nSize matSize, float factor)
{
	int i, j;
	for (i = 0; i<matSize.r; i++)
		for (j = 0; j<matSize.c; j++)
			res[i][j] = mat[i][j] * factor;
}

float summat(float** mat, nSize matSize) // ���������Ԫ�صĺ�
{
	float sum = 0.0;
	int i, j;
	for (i = 0; i<matSize.r; i++)
		for (j = 0; j<matSize.c; j++)
			sum = sum + mat[i][j];
	return sum;
}