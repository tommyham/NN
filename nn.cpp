#include <stdio.h>
#include<stdlib.h>
#include<fstream>
#include<string>
#include<sstream>
#define _USE_MATH_DEFINES
#include <iostream>
#include<complex>
#include<math.h>
#include<vector>
#include<time.h>
#include"nn.h"
using namespace std;

int main() {
	int input_element, output_element = 0;
	vector<double>input(0, 0);
	vector<vector<double>>input_data((0, 0));
	vector<double>label(0, 0);
	vector<vector<double>>label_data((0, 0));
	cout << "input_element=";
	scanf_s("%d", &input_element);


	ifstream file_d("input.txt");
	string line;
	vector<string>strvec;
	//入力データの読み込み
	while (getline(file_d, line)) {
		input.resize(0);
		strvec = split(line, ',');
		for (int i = 0; i < strvec.size(); i++) {
			input.push_back(stod(strvec.at(i)));
		}
		int input_size = input.size();
		//指定した入力値より多い要素分だけ削除
		for (int i = 0; i < input_size - input_element; i++) {
			input.pop_back();
		}
		input.push_back(1);//バイアス項の入力を追加
		input_data.push_back(input);
	}
	ifstream file_l("teacher.txt");
	//ラベルデータの読み込み
	while (getline(file_l, line)) {
		strvec = split(line, ',');
		label.resize(0);
		for (int i = 0; i < strvec.size(); i++) {
			label.push_back(stod(strvec.at(i)));
		}
		label_data.push_back(label);
	}

	//learn_online(input_data, label_data);
	learn_patch(input_data, label_data);
	//element_rand(input_data);

	return 0;
}