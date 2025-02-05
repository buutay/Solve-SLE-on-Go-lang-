package main

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

// Функция для генерации случайной матрицы n x n
func generateMatrix(n int) [][]float64 {
	matrix := make([][]float64, n)
	for i := 0; i < n; i++ {
		matrix[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			matrix[i][j] = rand.Float64() * 10 // случайные числа от 0 до 10
		}
	}
	return matrix
}

// Функция для генерации случайного вектора длиной n
func generateVector(n int) []float64 {
	vector := make([]float64, n)
	for i := 0; i < n; i++ {
		vector[i] = rand.Float64() * 10 // случайные числа от 0 до 10
	}
	return vector
}

// Функция для копирования матрицы
func copyMatrix(matrix [][]float64) [][]float64 {
	n := len(matrix)
	newMatrix := make([][]float64, n)
	for i := 0; i < n; i++ {
		newMatrix[i] = make([]float64, n)
		copy(newMatrix[i], matrix[i])
	}
	return newMatrix
}

// Функция для вычисления определителя матрицы
func determinant(matrix [][]float64) float64 {
	n := len(matrix)
	if n == 1 {
		return matrix[0][0]
	}
	if n == 2 {
		return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
	}

	det := 0.0
	for i := 0; i < n; i++ {
		submatrix := make([][]float64, n-1)
		for j := 0; j < n-1; j++ {
			submatrix[j] = make([]float64, n-1)
			for k := 0; k < n-1; k++ {
				submatrix[j][k] = matrix[j+1][(k+i+1)%n]
			}
		}
		det += math.Pow(-1, float64(i)) * matrix[0][i] * determinant(submatrix)
	}
	return det
}

// Функция для решения СЛАУ методом Крамера без параллелизма
func cramerSequential(A [][]float64, b []float64) ([]float64, error) {
	n := len(A)
	detA := determinant(A)

	if detA == 0 {
		return nil, fmt.Errorf("система уравнений не имеет решений или имеет бесконечно много решений")
	}

	x := make([]float64, n)
	for i := 0; i < n; i++ {
		Ai := copyMatrix(A)
		for j := 0; j < n; j++ {
			Ai[j][i] = b[j]
		}
		x[i] = determinant(Ai) / detA
	}

	return x, nil
}

// Функция для решения СЛАУ методом Крамера с параллелизмом
func cramerParallel(A [][]float64, b []float64) ([]float64, error) {
	n := len(A)
	detA := determinant(A)

	if detA == 0 {
		return nil, fmt.Errorf("система уравнений не имеет решений или имеет бесконечно много решений")
	}

	x := make([]float64, n)
	var wg sync.WaitGroup

	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			Ai := copyMatrix(A)
			for j := 0; j < n; j++ {
				Ai[j][i] = b[j]
			}
			x[i] = determinant(Ai) / detA
		}(i)
	}

	wg.Wait()

	return x, nil
}

func main() {
	n := 20 // Размерность системы уравнений

	A := generateMatrix(n)
	b := generateVector(n)

	// Замер времени решения без параллелизма
	start := time.Now()
	cramerSequential(A, b)
	durationSequential := time.Since(start)

	// Замер времени решения с параллелизмом
	start = time.Now()
	cramerParallel(A, b)
	durationParallel := time.Since(start)

	fmt.Printf("Время решения без параллелизма: %v\n", durationSequential)
	fmt.Printf("Время решения с параллелизмом: %v\n", durationParallel)
}
