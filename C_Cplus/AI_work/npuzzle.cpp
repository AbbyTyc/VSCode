//#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <string.h>
#include <math.h>

int num = 0, open[100][7][7] = { 0 }, target[7][7] = { 0 }, fn[100] = { 0 }, father[100] = { 0 }, son[100] = {0}, zero[100][2] = { 0 };
int target_pos[100][2] = { 0 }, init_pos[100][2] = { 0 }, origin[100] = { 0 }, step=0;
// open��״̬��target��Ŀ��״̬��fn���ܴ��ۺ�����father�Ǹ�״̬������son����״̬������zero��ʾÿ��״̬��0���ڵ�λ�ã�target_pos��ʾĿ��״̬��������λ��
// init_pos��ʾ��ʼ״̬��������λ�ã�origin��ʾ������״̬˳��������step�ǲ���



int g_n(int n)			//�Ա�Ŀǰ״̬���ʼ״̬�Ĳ��
{
	int i, j, cost = 0;

	for (i = 0; i < num; i++)
	{
		for (j = 0; j < num; j++)
		{
			cost += abs(init_pos[open[n][i][j]][0] - i) + abs(init_pos[open[n][i][j]][1] - j);		//Ӧ�������پ���
		}
	}

	return cost;
}



int h_n(int n)			//�Ա�Ŀǰ״̬��Ŀ��״̬�Ĳ��
{
	int i, j, cost=0;

	for (i = 0; i < num; i++)
	{
		for (j = 0; j < num; j++)
		{
			cost += abs(target_pos[open[n][i][j]][0] - i) + abs(target_pos[open[n][i][j]][1] - j);		//Ӧ�������پ���
		}
	}

	return cost;
}


void A_star()
{
	int move[4][2] = {1,0,  -1,0,  0,-1,  0,1};		// ��+�У����������ƶ�
	int head = 0, tail = 0, least=0, x, y, xx, yy, hn[100], gn[100], i, j, k, tmp;

	head = origin[least];		//leastָ��origin��ͷ��,head��Ӧopen����Ӧ״̬
	hn[head] = h_n(head);	//�жϳ�ʼ״̬��Ŀ��״̬�Ĳ��
	while (hn[head]>0)
	{
		for (i = 0; i < 4; i++)
		{
			x = zero[head][0];
			y = zero[head][1];		//0Ԫ��ԭ����
			xx = x + move[i][0];
			yy = y + move[i][1];		//0Ԫ��������
			if (hn[head]>0 && xx >= 0 && xx < num && yy >= 0 && yy < num)		//����״̬���ĸ�������
			{
				tail++;
				for (j = 0; j < num; j++)
				{
					for (k = 0; k < num; k++)
					{
						open[tail][j][k] = open[head][j][k];
					}
				}
				
				open[tail][x][y] = open[head][xx][yy];		//�µ�0Ԫ��λ�õ�ֵ�滻ԭ0Ԫ��λ��
				open[tail][xx][yy] = 0;
				zero[tail][0] = xx;
				zero[tail][1] = yy;

				father[tail] = head;		//����չ��״̬���������ĸ�״̬
				origin[tail] = tail;			//�¼ӵ�״̬����Ϊ�Լ�

				hn[tail] = h_n(tail);		//h(n)
				gn[tail] = g_n(tail);		//g(n)
				fn[tail] = hn[tail] + gn[tail];		//f(n)=g(n)+h(n)
			}
		}
		least++;
				
		for (i = tail-1; i >= least; i--)			//ѡ��fn��С��״̬������չ���ŵ�head��
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

		head = origin[least];		//leastָ��origin��ͷ��,head��Ӧopen����Ӧ״̬
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
	while (head != 0)		//���ݸ�״̬��ת���õ���״̬��ϵ
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

	f_in = fopen("npuzzle_in.txt", "r");		//�������ļ�
	if (f_in != NULL)		//�ļ��ǿ�
	{
		fscanf(f_in, "%s", str_in);		//n��
		num = str_in[0] - '0';

		for (i = 0; (i < num*num) && (feof(f_in) == 0); i++)		//��ȡԭʼ����ֲ�
		{
			fscanf(f_in, "%s", str_in);
			tmp = 0;
			for (j = 0; j < strlen(str_in); j++)
			{
				tmp = tmp * 10 + str_in[j] - '0';
			}

			open[0][i / num][i % num] = tmp;
			init_pos[tmp][0] = i / num;		//����tmp���к�
			init_pos[tmp][1] = i % num;		//����tmp���к�
			
			if (tmp == 0)
			{
				zero[0][0] = i / num;		// 0Ԫ���к�
				zero[0][1] = i % num;		// 0Ԫ���к�
			}
		}

		for (i = 0; (i < num*num) && (feof(f_in) == 0); i++)		//��ȡĿ������ֲ�
		{
			fscanf(f_in, "%s", str_in);
			tmp = 0;
			for (j = 0; j < strlen(str_in); j++)
			{
				tmp = tmp * 10 + str_in[j] - '0';
			}

			target[i / num][i % num] = tmp;			
			target_pos[tmp][0] = i / num;		//����tmp���к�
			target_pos[tmp][1] = i % num;		//����tmp���к�
		}

		fclose(f_in);
	}
}


void print_out()
{
	FILE  *f_out;
	int i, j, k, head=0;

	f_out = fopen("npuzzle_out.txt", "w");		//������ļ�
	if (f_out != NULL)
	{
		fprintf(f_out, "��%d��\n", step);
		
		fprintf(f_out, "��ʼ״̬\n");
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
			fprintf(f_out, "��%d��\n", k);
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
		
		fprintf(f_out, "Ŀ��״̬\n");
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