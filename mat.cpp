#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include "mat.h"

float** rotate180(float** mat, nSize matSize)// 矩阵翻转180度,输入mat为二位矩阵,matSize为矩阵大小
{
	int i, c, r;
	int outSizeW = matSize.c;//矩阵宽度
	int outSizeH = matSize.r;//矩阵高度
							 //下面三行是生成一个输出矩阵,他的大小是WxH的矩阵,注意输入是WxH,那么旋转180度后输出自然也是WxH
	float** outputData = (float**)malloc(outSizeH * sizeof(float*));//分配H行一维数组
	for (i = 0; i<outSizeH; i++)
		outputData[i] = (float*)malloc(outSizeW * sizeof(float));//每行分配W列数据
																 //行优先存储,那么自然for循环是先高度后宽度,或者说先行后列
	for (r = 0; r<outSizeH; r++)
		for (c = 0; c<outSizeW; c++)
			outputData[r][c] = mat[outSizeH - r - 1][outSizeW - c - 1];//旋转180度,一目了然

	return outputData;
}

// 关于卷积和相关操作的输出选项
// 这里共有三种选择：full、same、valid，分别表示
// full指完全，操作后结果的大小为inSize+(mapSize-1)
// same指同输出相同大小
// valid指完全操作后的大小，一般为inSize-(mapSize-1)大小，其不需要将输入添0扩大。
// 互相关,map为卷积核,mapSize为卷积核大小,inputData为要卷积的图像,inSize是要卷积图像大小,type是上面说的卷积类型
/*这个函数可以进行改进,没必要都先把他进行完全卷积之后再截取部分,这样很浪费时间,计算valid的时候直接按照valid计算就OK了
实际上对于这个CNN网络还可以继续优化到所有东西都是正方形,以后再说*/
float** correlation(float** map, nSize mapSize, float** inputData, nSize inSize, int type)
{
	int i, j, c, r;
	int outSizeW = inSize.c + (mapSize.c - 1); // 这里的输出扩大一部分,完全卷积得到的卷积MAP的宽度/列数
	int outSizeH = inSize.r + (mapSize.r - 1);// 这里的输出扩大一部分,完全卷积得到的卷积MAP的高度/行数
	int newSize = outSizeW - 1 + mapSize.c;//exInputData大小
	float** outputData = (float**)malloc(outSizeH * sizeof(float*)); // 互相关的结果扩大了,分配outSizeH行数组
	for (i = 0; i<outSizeH; i++)
		outputData[i] = (float*)calloc(outSizeW, sizeof(float));//每个数组大小为outSizeW

																// 为了方便计算，将inputData扩大一圈
	float** exInputData = matEdgeExpand(inputData, inSize, mapSize.c - 1, mapSize.r - 1);//进行full卷积的时候,把要卷积的图像扩大卷积核大小-1的宽度,才能进行完全卷积
																						 //以下四个for循环进行full卷积操作
	for (j = 0; j<outSizeH; j++)//对于输出MAP的每一行
		for (i = 0; i<outSizeW; i++)//对于输出MAP的每一列
			for (r = 0; r<mapSize.r; r++)//对于卷积核的每一行
				for (c = 0; c<mapSize.c; c++) {//对于卷积核的每一列
					outputData[j][i] = outputData[j][i] + map[r][c] * exInputData[j + r][i + c];
					//outputData的第j行第i列的值,等于卷积核第r行第c列的值乘以扩充之后原始图像的第j+r行第i+c列的结果的总和,即完成了卷积操作
				}

	//float** outputData = ocorrelation(map, mapSize.r, exInputData, newSize, type);//根据输入的原始图像和卷积核以及卷积类型进行卷积操作OPENCL
	//释放不用的扩充之后的exInputData数组
	for (i = 0; i<inSize.r + 2 * (mapSize.r - 1); i++)
		free(exInputData[i]);
	free(exInputData);
	nSize outSize = { outSizeW,outSizeH };//定义输出MAP的大小,默认为full卷积之后输出MAP的大小
	return outputData;//直接返回
}
// 卷积操作,map代表卷积核,mapSize为卷积核大小,inputData是要卷积的数据,inSize是要卷积数据的大小,type为卷积类型
float** cov(float** map, nSize mapSize, float** inputData, nSize inSize, int type)
{
	// 卷积操作可以用旋转180度的特征模板相关来求
	float** flipmap = rotate180(map, mapSize); //旋转180度的特征模板,或者称为卷积核
											   //float** res = ocorrelation(flipmap, mapSize.r, inputData, inSize.r, type);//根据输入的原始图像和卷积核以及卷机类型进行卷积操作
											   //float **res;
	float** res = correlation(flipmap, mapSize, inputData, inSize, type);//根据输入的原始图像和卷积核以及卷机类型进行卷积操作
	int i;
	//以下释放临时存储的卷积MAP的空间
	for (i = 0; i<mapSize.r; i++)
		free(flipmap[i]);
	free(flipmap);
	return res;
}

// 这个是矩阵的上采样（等值内插），upc及upr是在X方向和Y方向的内插倍数,本CNN网络两个值都是2
float** UpSample(float** mat, nSize matSize, int upc, int upr)
{
	int i, j, m, n;
	int c = matSize.c;//原列数
	int r = matSize.r;//原矩阵行数
	float** res = (float**)malloc((r*upr) * sizeof(float*)); // 结果的初始化,分配r*upr行的数组
	for (i = 0; i<(r*upr); i++)
		res[i] = (float*)malloc((c*upc) * sizeof(float));//每行数组大小为c*upc

	for (j = 0; j<r*upr; j = j + upr) {//在行方向上,每次填充upr个相同的值,注意这里是高度,这里一个j就是原始map一行的数据,一次for循环执行完,整个一行的数据就扩充完了
		for (i = 0; i<c*upc; i = i + upc)// 宽的扩充,即x方向上每隔upc个值改变一次赋值
			for (m = 0; m<upc; m++)//每次对连续的upc个元素赋值
				res[j][i + m] = mat[j / upr][i / upc];//填充行
													  /*上面两个for实现了第一行的列方向上扩充,然后接下来两个for将基于上面的第一行扩充第二行之后的所有的行*/
		for (n = 1; n<upr; n++)      //  高的扩充,第二行到最后一行
			for (i = 0; i<c*upc; i++)//列方向切换
				res[j + n][i] = res[j][i];//填充刚才第一行的结果
	}
	return res;
}

// 给二维矩阵边缘扩大，增加addw大小的0值边
float** matEdgeExpand(float** mat, nSize matSize, int addc, int addr)
{ // 向量边缘扩大,其中addc和addr分别为要在宽度和高度,也就是列和行扩充的个数,算法是addc=filterSizeX-1,addr=filterSizeY-1,filterSize是卷积核大小
	int i, j;
	int c = matSize.c;//原矩阵列数
	int r = matSize.r;//原矩阵行数
	float** res = (float**)malloc((r + 2 * addr) * sizeof(float*)); // 结果的初始化,分配r+2*addr行数组,上边addr个,下边addr个
	for (i = 0; i<(r + 2 * addr); i++)
		res[i] = (float*)malloc((c + 2 * addc) * sizeof(float));//每行数组个数为c+2*addc个,左边扩充addc个,右边addc个

	for (j = 0; j<r + 2 * addr; j++) {
		for (i = 0; i<c + 2 * addc; i++) {
			if (j<addr || i<addc || j >= (r + addr) || i >= (c + addc))//如果是在新扩充的边缘处,设置为0
				res[j][i] = (float)0.0;
			else
				res[j][i] = mat[j - addr][i - addc]; // 不然,复制原向量的数据
		}
	}
	return res;
}

// 给二维矩阵边缘缩小，擦除shrinkc大小的边,是上一个函数的完全相反的操作,不做解释
float** matEdgeShrink(float** mat, nSize matSize, int shrinkc, int shrinkr)
{ // 向量的缩小，宽缩小addw，高缩小addh
	int i, j;
	int c = matSize.c;
	int r = matSize.r;
	float** res = (float**)malloc((r - 2 * shrinkr) * sizeof(float*)); // 结果矩阵的初始化
	for (i = 0; i<(r - 2 * shrinkr); i++)
		res[i] = (float*)malloc((c - 2 * shrinkc) * sizeof(float));
	for (j = 0; j<r; j++) {
		for (i = 0; i<c; i++) {
			if (j >= shrinkr&&i >= shrinkc&&j<(r - shrinkr) && i<(c - shrinkc))
				res[j - shrinkr][i - shrinkc] = mat[j][i]; // 复制原向量的数据
		}
	}
	return res;
}

void savemat(float** mat, nSize matSize, const char* filename)//保存矩阵到filename
{
	FILE  *fp = NULL;
	fp = fopen(filename, "wb");//二进制形式保存
	if (fp == NULL)
		printf("write file failed\n");
	int i;
	for (i = 0; i<matSize.r; i++)
		fwrite(mat[i], sizeof(float), matSize.c, fp);//行优先存储
	fclose(fp);
}
// 两个矩阵各元素对应位置相加,mat1加mat2得到res,矩阵大小不变
void addmat(float** res, float** mat1, nSize matSize1, float** mat2, nSize matSize2)
{
	int i, j;
	if (matSize1.c != matSize2.c || matSize1.r != matSize2.r)//要保证两个矩阵的大小相同
		printf("ERROR: Size is not same!");

	for (i = 0; i<matSize1.r; i++)
		for (j = 0; j<matSize1.c; j++)
			res[i][j] = mat1[i][j] + mat2[i][j];//相加然后返回给res
}
// 矩阵所有元素乘以同一系数
void multifactor(float** res, float** mat, nSize matSize, float factor)
{
	int i, j;
	for (i = 0; i<matSize.r; i++)
		for (j = 0; j<matSize.c; j++)
			res[i][j] = mat[i][j] * factor;
}

float summat(float** mat, nSize matSize) // 求矩阵所有元素的和
{
	float sum = 0.0;
	int i, j;
	for (i = 0; i<matSize.r; i++)
		for (j = 0; j<matSize.c; j++)
			sum = sum + mat[i][j];
	return sum;
}