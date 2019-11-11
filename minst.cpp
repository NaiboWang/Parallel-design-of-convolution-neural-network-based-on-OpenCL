#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "minst.h"

/*英特尔处理器和其他低端机用户必须翻转头字节。例如原始数据集存的值为00006ae0(大端模式),那么X86结构默认会把他读成e06a0000,那么需要把这个转换成00006ae0才好*/
int ReverseInt(int i)
{  
	unsigned char *split = (unsigned char*)&i;
	return ((int)split[0])<<24 | split[1]<<16 | split[2]<<8 | split[3];
}

ImgArr read_Img(const char* filename) // 读入图像
{
	FILE  *fp=NULL;
	fp=fopen(filename,"rb");//二进制方式读取
	if(fp==NULL)
		printf("open file failed\n");
	assert(fp);//成功则继续向下执行,否则退出给编译器提示错误

	int magic_number = 0;  //魔数,用来检验是不是正确的数据集
	int number_of_images = 0;  //数据集个数,6W或1W
	int n_rows = 0;  //数据集行数
	int n_cols = 0;  //数据集列数
	//从文件中读取sizeof(magic_number) 个字符到 &magic_number  
	fread((char*)&magic_number,sizeof(magic_number),1,fp); //文件读写指针会自动跟着向下,第三个参数指定读取1个数据
	magic_number = ReverseInt(magic_number);  //转换成正确的数字
	//获取训练或测试image的个数number_of_images 
	fread((char*)&number_of_images,sizeof(number_of_images),1,fp);  
	number_of_images = ReverseInt(number_of_images);    
	//获取训练或测试图像的高度Heigh  
	fread((char*)&n_rows,sizeof(n_rows),1,fp); 
	n_rows = ReverseInt(n_rows);                  
	//获取训练或测试图像的宽度Width  
	fread((char*)&n_cols,sizeof(n_cols),1,fp); 
	n_cols = ReverseInt(n_cols);  
	//获取第i幅图像，保存到vec中 
	int i,r,c;

	// 图像数组的初始化
	ImgArr imgarr=(ImgArr)malloc(sizeof(MinstImgArr));//imgarr是保存的原始图像的最外层结构体
	imgarr->ImgNum=number_of_images;//指定数据集大小,60000或10000
	imgarr->ImgPtr=(MinstImg*)malloc(number_of_images*sizeof(MinstImg));//分配number_of_images个图像存储结构体

	for(i = 0; i < number_of_images; ++i)  //分别对图像进行赋值操作
	{  
		imgarr->ImgPtr[i].r=n_rows;//定义行数,28
		imgarr->ImgPtr[i].c=n_cols;//定义列数,28
		imgarr->ImgPtr[i].ImgData=(float*)malloc(n_rows*n_cols*sizeof(float));//分配28X28大小的一维数组,用来保存一行数据
		for(r = 0; r < n_rows; ++r)      
		{
			for(c = 0; c < n_cols; ++c)
			{ 
				unsigned char temp = 0; 
				fread((char*) &temp, sizeof(temp),1,fp);//从数据文件中读取一个字节到temp变量中 
				imgarr->ImgPtr[i].ImgData[r*n_cols+c]=(float)temp/255.0;//归一化成0到1之间的float
			}  
		}    
	}

	fclose(fp);
	return imgarr;
}

LabelArr read_Lable(const char* filename)// 读入图像
{
	FILE  *fp=NULL;
	fp=fopen(filename,"rb");
	if(fp==NULL)
		printf("open file failed\n");
	assert(fp);

	int magic_number = 0;  
	int number_of_labels = 0; 
	int label_long = 10;

	//从文件中读取sizeof(magic_number) 个字符到 &magic_number  
	fread((char*)&magic_number,sizeof(magic_number),1,fp); 
	magic_number = ReverseInt(magic_number);  
	//获取训练或测试image的个数number_of_images 
	fread((char*)&number_of_labels,sizeof(number_of_labels),1,fp);  
	number_of_labels = ReverseInt(number_of_labels);    

	int i,l;

	// 图像标记数组的初始化
	LabelArr labarr=(LabelArr)malloc(sizeof(MinstLabelArr));
	labarr->LabelNum=number_of_labels;
	labarr->LabelPtr=(MinstLabel*)malloc(number_of_labels*sizeof(MinstLabel));

	for(i = 0; i < number_of_labels; ++i)  
	{  
		labarr->LabelPtr[i].l=10;//定义标签数量为10个
		labarr->LabelPtr[i].LabelData=(float*)calloc(label_long,sizeof(float));//这里使用calloc,意味着分配10个浮点数,同时把这些浮点数初始化为0.0
		unsigned char temp = 0;  
		fread((char*) &temp, sizeof(temp),1,fp);//读取标签值,这个值是0~9的值 
		labarr->LabelPtr[i].LabelData[(int)temp]=1.0;    //因此,给LabelData[temp]赋值为1.0表示这个图像的正确标签为temp所指向的那一位
	}
	fclose(fp);
	return labarr;	
}

char* intTochar(int i)// 将数字转换成字符串
{
	int itemp=i;
	int w=0;
	while(itemp>=10){
		itemp=itemp/10;
		w++;
	}
	char* ptr=(char*)malloc((w+2)*sizeof(char));
	ptr[w+1]='\0';
	int r; // 余数
	while(i>=10){
		r=i%10;
		i=i/10;		
		ptr[w]=(char)(r+48);
		w--;
	}
	ptr[w]=(char)(i+48);
	return ptr;
}

char * combine_strings(char *a, char *b) // 将两个字符串相连
{
	char *ptr;
	int lena=strlen(a),lenb=strlen(b);
	int i,l=0;
	ptr = (char *)malloc((lena+lenb+1) * sizeof(char));
	for(i=0;i<lena;i++)
		ptr[l++]=a[i];
	for(i=0;i<lenb;i++)
		ptr[l++]=b[i];
	ptr[l]='\0';
	return(ptr);
}

void save_Img(ImgArr imgarr,char* filedir) // 将图像数据保存成文件
{
	int img_number=imgarr->ImgNum;
	int i,r;
	for(i=0;i<img_number;i++){
		const char* filename=combine_strings(filedir,combine_strings(intTochar(i),".gray"));
		FILE  *fp=NULL;
		fp=fopen(filename,"wb");//二进制方式保存图片
		if(fp==NULL)
			printf("write file failed\n");
		assert(fp);
			fwrite(imgarr->ImgPtr[i].ImgData,sizeof(float),imgarr->ImgPtr[i].c,fp);//把imgarr->ImgPtr[i].ImgData[r]整个数组写入fp中,为什么是整个数组,因为imgarr->ImgPtr[i].ImgData[r]代表数组第一个元素,然后个数指定的是imgarr->ImgPtr[i].c,就是写入了整个数组
		
		fclose(fp);
	}	
}