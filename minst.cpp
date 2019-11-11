#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "minst.h"

/*Ӣ�ض��������������Ͷ˻��û����뷭תͷ�ֽڡ�����ԭʼ���ݼ����ֵΪ00006ae0(���ģʽ),��ôX86�ṹĬ�ϻ��������e06a0000,��ô��Ҫ�����ת����00006ae0�ź�*/
int ReverseInt(int i)
{  
	unsigned char *split = (unsigned char*)&i;
	return ((int)split[0])<<24 | split[1]<<16 | split[2]<<8 | split[3];
}

ImgArr read_Img(const char* filename) // ����ͼ��
{
	FILE  *fp=NULL;
	fp=fopen(filename,"rb");//�����Ʒ�ʽ��ȡ
	if(fp==NULL)
		printf("open file failed\n");
	assert(fp);//�ɹ����������ִ��,�����˳�����������ʾ����

	int magic_number = 0;  //ħ��,���������ǲ�����ȷ�����ݼ�
	int number_of_images = 0;  //���ݼ�����,6W��1W
	int n_rows = 0;  //���ݼ�����
	int n_cols = 0;  //���ݼ�����
	//���ļ��ж�ȡsizeof(magic_number) ���ַ��� &magic_number  
	fread((char*)&magic_number,sizeof(magic_number),1,fp); //�ļ���дָ����Զ���������,����������ָ����ȡ1������
	magic_number = ReverseInt(magic_number);  //ת������ȷ������
	//��ȡѵ�������image�ĸ���number_of_images 
	fread((char*)&number_of_images,sizeof(number_of_images),1,fp);  
	number_of_images = ReverseInt(number_of_images);    
	//��ȡѵ�������ͼ��ĸ߶�Heigh  
	fread((char*)&n_rows,sizeof(n_rows),1,fp); 
	n_rows = ReverseInt(n_rows);                  
	//��ȡѵ�������ͼ��Ŀ��Width  
	fread((char*)&n_cols,sizeof(n_cols),1,fp); 
	n_cols = ReverseInt(n_cols);  
	//��ȡ��i��ͼ�񣬱��浽vec�� 
	int i,r,c;

	// ͼ������ĳ�ʼ��
	ImgArr imgarr=(ImgArr)malloc(sizeof(MinstImgArr));//imgarr�Ǳ����ԭʼͼ��������ṹ��
	imgarr->ImgNum=number_of_images;//ָ�����ݼ���С,60000��10000
	imgarr->ImgPtr=(MinstImg*)malloc(number_of_images*sizeof(MinstImg));//����number_of_images��ͼ��洢�ṹ��

	for(i = 0; i < number_of_images; ++i)  //�ֱ��ͼ����и�ֵ����
	{  
		imgarr->ImgPtr[i].r=n_rows;//��������,28
		imgarr->ImgPtr[i].c=n_cols;//��������,28
		imgarr->ImgPtr[i].ImgData=(float*)malloc(n_rows*n_cols*sizeof(float));//����28X28��С��һά����,��������һ������
		for(r = 0; r < n_rows; ++r)      
		{
			for(c = 0; c < n_cols; ++c)
			{ 
				unsigned char temp = 0; 
				fread((char*) &temp, sizeof(temp),1,fp);//�������ļ��ж�ȡһ���ֽڵ�temp������ 
				imgarr->ImgPtr[i].ImgData[r*n_cols+c]=(float)temp/255.0;//��һ����0��1֮���float
			}  
		}    
	}

	fclose(fp);
	return imgarr;
}

LabelArr read_Lable(const char* filename)// ����ͼ��
{
	FILE  *fp=NULL;
	fp=fopen(filename,"rb");
	if(fp==NULL)
		printf("open file failed\n");
	assert(fp);

	int magic_number = 0;  
	int number_of_labels = 0; 
	int label_long = 10;

	//���ļ��ж�ȡsizeof(magic_number) ���ַ��� &magic_number  
	fread((char*)&magic_number,sizeof(magic_number),1,fp); 
	magic_number = ReverseInt(magic_number);  
	//��ȡѵ�������image�ĸ���number_of_images 
	fread((char*)&number_of_labels,sizeof(number_of_labels),1,fp);  
	number_of_labels = ReverseInt(number_of_labels);    

	int i,l;

	// ͼ��������ĳ�ʼ��
	LabelArr labarr=(LabelArr)malloc(sizeof(MinstLabelArr));
	labarr->LabelNum=number_of_labels;
	labarr->LabelPtr=(MinstLabel*)malloc(number_of_labels*sizeof(MinstLabel));

	for(i = 0; i < number_of_labels; ++i)  
	{  
		labarr->LabelPtr[i].l=10;//�����ǩ����Ϊ10��
		labarr->LabelPtr[i].LabelData=(float*)calloc(label_long,sizeof(float));//����ʹ��calloc,��ζ�ŷ���10��������,ͬʱ����Щ��������ʼ��Ϊ0.0
		unsigned char temp = 0;  
		fread((char*) &temp, sizeof(temp),1,fp);//��ȡ��ǩֵ,���ֵ��0~9��ֵ 
		labarr->LabelPtr[i].LabelData[(int)temp]=1.0;    //���,��LabelData[temp]��ֵΪ1.0��ʾ���ͼ�����ȷ��ǩΪtemp��ָ�����һλ
	}
	fclose(fp);
	return labarr;	
}

char* intTochar(int i)// ������ת�����ַ���
{
	int itemp=i;
	int w=0;
	while(itemp>=10){
		itemp=itemp/10;
		w++;
	}
	char* ptr=(char*)malloc((w+2)*sizeof(char));
	ptr[w+1]='\0';
	int r; // ����
	while(i>=10){
		r=i%10;
		i=i/10;		
		ptr[w]=(char)(r+48);
		w--;
	}
	ptr[w]=(char)(i+48);
	return ptr;
}

char * combine_strings(char *a, char *b) // �������ַ�������
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

void save_Img(ImgArr imgarr,char* filedir) // ��ͼ�����ݱ�����ļ�
{
	int img_number=imgarr->ImgNum;
	int i,r;
	for(i=0;i<img_number;i++){
		const char* filename=combine_strings(filedir,combine_strings(intTochar(i),".gray"));
		FILE  *fp=NULL;
		fp=fopen(filename,"wb");//�����Ʒ�ʽ����ͼƬ
		if(fp==NULL)
			printf("write file failed\n");
		assert(fp);
			fwrite(imgarr->ImgPtr[i].ImgData,sizeof(float),imgarr->ImgPtr[i].c,fp);//��imgarr->ImgPtr[i].ImgData[r]��������д��fp��,Ϊʲô����������,��Ϊimgarr->ImgPtr[i].ImgData[r]���������һ��Ԫ��,Ȼ�����ָ������imgarr->ImgPtr[i].c,����д������������
		
		fclose(fp);
	}	
}