//#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <string.h>
#include <math.h>

int num = 0, open[100][7][7] = { 0 }, target[7][7] = { 0 }, fn[100] = { 0 }, father[100] = { 0 }, son[100] = {0}, zero[100][2] = { 0 };
int target_pos[100][2] = { 0 }, init_pos[100][2] = { 0 }, origin[100] = { 0 }, step=0;
// open是状态表，target是目标状态，fn是总代价函数，father是父状态索引，son是子状态索引，zero表示每个状态中0所在的位置，target_pos表示目标状态各个数字位置
// init_pos表示初始状态各个数字位置，origin表示排序后的状态顺序索引，step是步数



int g_n(int n)			//对比目前状态与初始状态的差距
{
	int i, j, cost = 0;

	for (i = 0; i < num; i++)
	{
		for (j = 0; j < num; j++)
		{
			cost += abs(init_pos[open[n][i][j]][0] - i) + abs(init_pos[open[n][i][j]][1] - j);		//应用曼哈顿距离
		}
	}

	return cost;
}



int h_n(int n)			//对比目前状态与目标状态的差距
{
	int i, j, cost=0;

	for (i = 0; i < num; i++)
	{
		for (j = 0; j < num; j++)
		{
			cost += abs(target_pos[open[n][i][j]][0] - i) + abs(target_pos[open[n][i][j]][1] - j);		//应用曼哈顿距离
		}
	}

	return cost;
}


void A_star()
{
	int move[4][2] = {1,0,  -1,0,  0,-1,  0,1};		// 行+列：上下左右移动
	int head = 0, tail = 0, least=0, x, y, xx, yy, hn[100], gn[100], i, j, k, tmp;

	head = origin[least];		//least指向origin的头部,head对应open的相应状态
	hn[head] = h_n(head);	//判断初始状态与目标状态的差距
	while (hn[head]>0)
	{
		for (i = 0; i < 4; i++)
		{
			x = zero[head][0];
			y = zero[head][1];		//0元素原坐标
			xx = x + move[i][0];
			yy = y + move[i][1];		//0元素新坐标
			if (hn[head]>0 && xx >= 0 && xx < num && yy >= 0 && yy < num)		//更改状态，四个方向尝试
			{
				tail++;
				for (j = 0; j < num; j++)
				{
					for (k = 0; k < num; k++)
					{
						open[tail][j][k] = open[head][j][k];
					}
				}
				
				open[tail][x][y] = open[head][xx][yy];		//新的0元素位置的值替换原0元素位置
				open[tail][xx][yy] = 0;
				zero[tail][0] = xx;
				zero[tail][1] = yy;

				father[tail] = head;		//将扩展的状态索引到它的父状态
				origin[tail] = tail;			//新加的状态索引为自己

				hn[tail] = h_n(tail);		//h(n)
				gn[tail] = g_n(tail);		//g(n)
				fn[tail] = hn[tail] + gn[tail];		//f(n)=g(n)+h(n)
			}
		}
		least++;
				
		for (i = tail-1; i >= least; i--)			//选择fn最小的状态继续扩展，放到head处
		{
			j = origin[i];
			k = origin[i + 1];
			if (fn[j] > fn[k])
			{
				tmp = fn[k];
				fn[k] = fn[j];
				fn[j] = tmp;
			}
		}

		head = origin[least];		//least指向origin的头部,head对应open的相应状态
		/*
		printf("father:");
		for (i = 0; i < 10; i++)
		{
			printf("%d ", father[i]);
		}
		printf("\n");
		*/
	}
	
	step = 0;
	while (head != 0)		//回溯父状态，转换得到子状态关系
	{
		tmp = head;		
		head = father[head];
		son[head] = tmp;

		step++;
	}
	
}



void read_in()
{
	FILE *f_in;
	int i, j, tmp;
	char str_in[100];

	f_in = fopen("npuzzle_in.txt", "r");		//打开输入文件
	if (f_in != NULL)		//文件非空
	{
		fscanf(f_in, "%s", str_in);		//n码
		num = str_in[0] - '0';

		for (i = 0; (i < num*num) && (feof(f_in) == 0); i++)		//读取原始数码分布
		{
			fscanf(f_in, "%s", str_in);
			tmp = 0;
			for (j = 0; j < strlen(str_in); j++)
			{
				tmp = tmp * 10 + str_in[j] - '0';
			}

			open[0][i / num][i % num] = tmp;
			init_pos[tmp][0] = i / num;		//数字tmp的行号
			init_pos[tmp][1] = i % num;		//数字tmp的列号
			
			if (tmp == 0)
			{
				zero[0][0] = i / num;		// 0元素行号
				zero[0][1] = i % num;		// 0元素列号
			}
		}

		for (i = 0; (i < num*num) && (feof(f_in) == 0); i++)		//读取目标数码分布
		{
			fscanf(f_in, "%s", str_in);
			tmp = 0;
			for (j = 0; j < strlen(str_in); j++)
			{
				tmp = tmp * 10 + str_in[j] - '0';
			}

			target[i / num][i % num] = tmp;			
			target_pos[tmp][0] = i / num;		//数字tmp的行号
			target_pos[tmp][1] = i % num;		//数字tmp的列号
		}

		fclose(f_in);
	}
}


void print_out()
{
	FILE  *f_out;
	int i, j, k, head=0;

	f_out = fopen("npuzzle_out.txt", "w");		//打开输出文件
	if (f_out != NULL)
	{
		fprintf(f_out, "共%d步\n", step);
		
		fprintf(f_out, "初始状态\n");
		for (i = 0; i < num; i++)
		{
			for (j = 0; j < num-1; j++)
			{
				fprintf(f_out, "%d ", open[0][i][j]);				
			}
			fprintf(f_out, "%d\n", open[0][i][j]);
		}

		for (k = 1; k < step; k++)
		{
			fprintf(f_out, "第%d步\n", k);
			head = son[head];
			for (i = 0; i < num; i++)
			{
				for (j = 0; j < num - 1; j++)
				{
					fprintf(f_out, "%d ", open[head][i][j]);
				}
				fprintf(f_out, "%d\n", open[head][i][j]);
			}
		}
		
		fprintf(f_out, "目标状态\n");
		for (i = 0; i < num; i++)
		{
			for (j = 0; j < num - 1; j++)
			{
				fprintf(f_out, "%d ", target[i][j]);
			}
			fprintf(f_out, "%d\n", target[i][j]);
		}
		
		fclose(f_out);
	}
}


int main()
{	
	read_in();
	A_star();
	print_out();
		
	return 0;
}