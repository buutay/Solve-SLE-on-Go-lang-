package main

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

// parallelSLAU решает систему линейных уравнений методом Гаусса с параллелизмом
func parallelSLAU(a [][]float64, b []float64) ([]float64, error) {
	n := len(a)
	if n != len(b) {
		return nil, fmt.Errorf("размеры матрицы A и вектора b не совпадают")
	}

	// Объединяем матрицу A и вектор b в расширенную матрицу
	ab := make([][]float64, n)
	for i := 0; i < n; i++ {
		ab[i] = make([]float64, n+1)
		copy(ab[i][:n], a[i])
		ab[i][n] = b[i]
	}

	// Преобразование к треугольному виду
	for i := 0; i < n; i++ {
		// Поиск ведущего элемента
		maxRow := i
		for j := i + 1; j < n; j++ {
			if math.Abs(ab[j][i]) > math.Abs(ab[maxRow][i]) {
				maxRow = j
			}
		}
		// Перестановка строк
		ab[i], ab[maxRow] = ab[maxRow], ab[i]

		// Нормализация текущей строки (для избежания проблем с малой точностью)
		divisor := ab[i][i]

		if math.Abs(divisor) < 1e-10 {
			return nil, fmt.Errorf("матрица вырождена или близка к вырожденной")
		}
		for j := i; j <= n; j++ {
			ab[i][j] /= divisor
		}

		var wg sync.WaitGroup
		//Исключение элементов под главной диагональю
		for j := i + 1; j < n; j++ {
			wg.Add(1)
			go func(j int) {
				defer wg.Done()
				factor := ab[j][i]
				for k := i; k <= n; k++ {
					ab[j][k] -= factor * ab[i][k]
				}
			}(j)
		}
		wg.Wait()
	}

	// Обратный ход (нахождение решений)
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		x[i] = ab[i][n]
		for j := i + 1; j < n; j++ {
			x[i] -= ab[i][j] * x[j]
		}
	}
	return x, nil
}

// regularSLAU решает систему линейных уравнений методом Гаусса без параллелизмом
func regularSLAU(a [][]float64, b []float64) ([]float64, error) {
	n := len(a)
	if n != len(b) {
		return nil, fmt.Errorf("размеры матрицы A и вектора b не совпадают")
	}

	// Объединяем матрицу A и вектор b в расширенную матрицу
	ab := make([][]float64, n)
	for i := 0; i < n; i++ {
		ab[i] = make([]float64, n+1)
		copy(ab[i][:n], a[i])
		ab[i][n] = b[i]
	}

	// Преобразование к треугольному виду
	for i := 0; i < n; i++ {
		// Поиск ведущего элемента
		maxRow := i
		for j := i + 1; j < n; j++ {
			if math.Abs(ab[j][i]) > math.Abs(ab[maxRow][i]) {
				maxRow = j
			}
		}
		// Перестановка строк
		ab[i], ab[maxRow] = ab[maxRow], ab[i]

		// Нормализация текущей строки (для избежания проблем с малой точностью)
		divisor := ab[i][i]

		if math.Abs(divisor) < 1e-10 {
			return nil, fmt.Errorf("матрица вырождена или близка к вырожденной")
		}
		for j := i; j <= n; j++ {
			ab[i][j] /= divisor
		}

		//Исключение элементов под главной диагональю
		for j := i + 1; j < n; j++ {
			factor := ab[j][i]
			for k := i; k <= n; k++ {
				ab[j][k] -= factor * ab[i][k]
			}

		}
	}

	// Обратный ход (нахождение решений)
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		x[i] = ab[i][n]
		for j := i + 1; j < n; j++ {
			x[i] -= ab[i][j] * x[j]
		}
	}
	return x, nil
}

func CreateRandomMatrix(n int) [][]float64 {
	m := make([][]float64, n)
	for i := 0; i < n; i++ {
		m[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			m[i][j] = float64(rand.Intn(100))
		}
	}
	return m
}

func CreateRandomVector(n int) []float64 {
	v := make([]float64, n)
	for i := 0; i < n; i++ {
		v[i] = float64(rand.Intn(100))
	}
	return v
}

func main() {
	N := 2000
	a := CreateRandomMatrix(N)
	b := CreateRandomVector(N)

	start1 := time.Now()
	x1, err1 := parallelSLAU(a, b)
	if err1 != nil {
		fmt.Println("Ошибка:", err1)
		return
	}
	end1 := time.Since(start1)
	fmt.Println(end1)

	start2 := time.Now()
	x2, err2 := regularSLAU(a, b)
	if err2 != nil {
		fmt.Println("Ошибка:", err2)
		return
	}
	end2 := time.Since(start2)
	fmt.Println(end2)

	//fmt.Println("Решение:")
	//for i, val := range x {
	//        fmt.Printf("x%d = %f\n", i+1, val)
	//}

	_ = x1
	_ = x2
}
