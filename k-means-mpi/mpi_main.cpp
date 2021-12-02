#include "mpi.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include <cmath>
#include <random>

using namespace std;

float getIndex(float** normalizedNumbers, float** clusterCenters, int* clusterPointsSize, int* clusterDataIndex);
float** readArrayAndNorm(string path);
float** alloc_2d_float(int rows, int cols);
float getDistance(float* point1, float* point2);
float* kmeans(float** normalizedItems);

// Стартовые значения
const int itemsCount = 517;
const int paramCount = 4;
const int clustersCount = 4;


int main(int argc, char* argv[])
{
    // Данные для передачи
    int rank;
    float indexResult[1];
    float indexResultMax[1];
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    srand(time(0) + rank);

    // Начальные данные
    float** normalizedItems = alloc_2d_float(itemsCount, paramCount);
	float* resultBest = new float[clustersCount + 1];

    // Поток с рангом 0 заполняет массив данными
    if (rank == 0) {
		string path = "dataBase.csv";
		normalizedItems = readArrayAndNorm(path);
    }
	// Отправка данных в другие потоки
	MPI_Bcast(&(normalizedItems[0][0]), itemsCount * paramCount, MPI_FLOAT, 0, MPI_COMM_WORLD);

    indexResult[0] = 999;
    //Первый процесс шлет другим, остальные получают
    if (rank != 0) {
		float* result = new float[clustersCount + 1];
		// Первая итерация
		resultBest = kmeans(normalizedItems);
		// Повторяем kmeans несколько раз и фиксируем лучший
		for (int i = 0; i < 5; i++) {
			result = kmeans(normalizedItems);
			if (result[0] < resultBest[0])
				for (int i = 0; i < clustersCount + 1; i++) {
					resultBest[i] = result[i];
				}
		}
		// Сохраняем результат
		indexResult[0] = resultBest[0];
    }

    // Ищем лучший результат и отправляем его остальным
    MPI_Reduce(indexResult, indexResultMax, 1, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Bcast(indexResultMax, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

	// Если поток имеет лучший результат то он выводит свои результаты
    if ((rank != 0) && (indexResult[0] == indexResultMax[0])) {
		cout << "Thread: " << rank << " My result index: " << indexResult[0] << endl;
		cout << "My clasters weight: ";
		for (int i = 1; i < clustersCount + 1; i++) {
			cout << resultBest[i] << " ";
		}
    }

	// Точка выхода
    MPI_Finalize();
    return 0;
}

// Считывание данных и их нормирование
float** readArrayAndNorm(string path) {
	// Открываем потоки чтения
	string line;
	ifstream in(path);
	float** characters = alloc_2d_float(itemsCount, paramCount);

	// Заполняем массив данными
	if (in.is_open())
	{
		int i = 0;
		while (getline(in, line))
		{
			string str1 = "";
			int j = 0;
			for (char& c : line) {
				if (c == ';') {
					characters[i][j] = stof(str1);
					str1 = "";
					j++;
				}
				else
					str1 += c;
			}
			i++;
		}
	}
	else
		cout << "file_not_found" << endl;

	// Поиск экстремумов
	float extremum[2][paramCount];
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < paramCount; j++) {
			extremum[i][j] = characters[0][j];
		}
	}
	for (int i = 1; i < itemsCount; i++) {
		for (int j = 0; j < paramCount; j++) {
			float number = characters[i][j];
			if (number > extremum[0][j])
				extremum[0][j] = number;
			else if (number < extremum[1][j])
				extremum[1][j] = number;
		}
	}

	// Нормирование
	for (int i = 0; i < itemsCount; i++) {
		for (int j = 0; j < paramCount; j++) {
			characters[i][j] = (characters[i][j] - extremum[1][j]) / (extremum[0][j] - extremum[1][j]);
		}
	}
	in.close();
	return characters;
}

// Функция для передачи матрицы в mpi
float** alloc_2d_float(int rows, int cols) {
	float* data = (float*)malloc(rows * cols * sizeof(float));
	float** array = (float**)malloc(rows * sizeof(float*));
	for (int i = 0; i < rows; i++)
		array[i] = &(data[cols * i]);
	return array;
}

// Расстояние между двумя векторами
float getDistance(float* point1, float* point2) {
	float distance = 0;
	for (int i = 0; i < paramCount; i++) {
		distance += pow((point1[i] - point2[i]), paramCount);
	}
	return pow(distance, (1.0 / paramCount));
}

// Алгоритм kmeans
float* kmeans(float** normalizedItems) {

	// Выделяем память 
	float clusterData[itemsCount];
	int clusterDataIndex[itemsCount];
	float** clusterVector = alloc_2d_float(itemsCount, paramCount);
	float index = 0;
	int* clusterWeight = new int[clustersCount];


	// Задаем начальные значения центров кластеров
	srand(time(0));
	std::random_device dev;
	std::mt19937 rng(dev());
	std::uniform_int_distribution<std::mt19937::result_type> dist6(0, itemsCount);
	for (int i = 0; i < clustersCount; i++) {
		int rnd = dist6(rng);
		for (int j = 0; j < paramCount; j++) {
			clusterVector[i][j] = normalizedItems[rnd][j];
		}
	}

	// Переменные для остановки
	float indexLast = 0;
	int stopCount = 0;

	// Начало цикла поиска
	while (stopCount != 3) {

		// Обнуляем переменные
		for (int i = 0; i < clustersCount; i++) {
			clusterWeight[i] = 0.0;
		}
		for (int i = 0; i < itemsCount; i++) {
			clusterData[i] = -1.0;
			clusterDataIndex[i] = -1;
		}

		// Распределяем значения по кластерам
		for (int i = 0; i < clustersCount; i++) {
			for (int j = 0; j < itemsCount; j++) {
				float distance = getDistance(normalizedItems[j], clusterVector[i]);
				if ((clusterData[j] < 0) || (clusterData[j] > distance)) {
					clusterData[j] = getDistance(normalizedItems[j], clusterVector[i]);
					clusterDataIndex[j] = i;
				}
			}
		}

		// Сбрасываем значения центров
		for (int i = 0; i < clustersCount; i++) {
			for (int j = 0; j < paramCount; j++) {
				clusterVector[i][j] = 0.0;
			}
		}


		// Находим новые центры
		for (int i = 0; i < itemsCount; i++) {
			clusterWeight[clusterDataIndex[i]]++;
			for (int j = 0; j < paramCount; j++) {
				clusterVector[clusterDataIndex[i]][j] += normalizedItems[i][j];
			}
		}
		for (int i = 0; i < clustersCount; i++) {
			for (int j = 0; j < paramCount; j++) {
				clusterVector[i][j] /= clusterWeight[i];
			}
		}

		// Проверка выхода
		index = getIndex(normalizedItems, clusterVector, clusterWeight, clusterDataIndex);
		if (indexLast == index)
			stopCount++;
		else {
			stopCount = 0;
			indexLast = index;
		}
	}

	// Передаем результат
	float* result = new float[clustersCount + 1];
	result[0] = index;
	for (int i = 1; i < clustersCount + 1; i++) {
		result[i] = clusterWeight[i - 1];
	}
	return result;



}

// Вычисление индекса
float getIndex(float** normalizedItems, float** clusterVector, int* clusterWeight, int* clusterDataIndex) {
	float dispersion[clustersCount];

	// Обнуляем массив с дисперсией
	for (int i = 0; i < clustersCount; i++)
		dispersion[i] = 0;

	// Ищем дисперсию S
	for (int i = 0; i < clustersCount; i++) {
		for (int j = 0; j < itemsCount; j++) {
			if (clusterDataIndex[j] == i) {
				dispersion[i] += getDistance(normalizedItems[j], clusterVector[i]);
			}
		}
		dispersion[i] /= clusterWeight[i];
	}

	// Вычислям индекс по формуле DB = (max((Si+Sj)/dij)) / c
	float DB = 0;
	for (int i = 0; i < clustersCount; i++) {
		float maxDivision = -1;
		for (int j = 0; j < clustersCount; j++) {
			if (i == j)
				continue;
			float distance = getDistance(clusterVector[i], clusterVector[j]);
			if (distance != 0)
				distance = (dispersion[i] + dispersion[j]) / distance;
			if (distance > maxDivision)
				maxDivision = distance;
		}
		DB += maxDivision;
	}
	DB /= clustersCount;
	return DB;
}