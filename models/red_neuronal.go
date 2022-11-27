package models

type RedNeuronal struct {
	// Arreglo con la cantidad de neuronas que tendrá cada capa (sin contar la
	// capa de entrada). La cantidad de capas es len(NeuronasPorCapa)
	NeuronasPorCapa []int
	// Entradas de la red neuronal. La primera es el bias.
	Entradas []float64
	// Los pesos iniciales de cada una de las conexiones. En la siguiente estructura:
	// [nº de capa][nº de neurona de dicha capa][cada conexión de dicha neurona]
	// PesosIniciales debe tener length de [len(numeroCapas)] [numeroCapas[i]] [numeroCapas[i-1]]
	PesosIniciales [][][]float64
	// Coeficiente de aprendizaje de la red neuronal
	CAprendizaje float64
	// Vector de salidas deseadas
	SalidasDeseadas []float64
}

// Neuronas del perceptron. La estructura es la siguiente:
// [nº de capa][nº identificativo de la neurona en dicha capa]
var neuronas [][]*Neurona

// Perceptrones de la red. Su estructura es la siguiente:
// [nº de capa][nº identificativo de la neurona en dicha capa]

// Incializa todos las neuronas del perceptrón
func (r *RedNeuronal) InitPerceptron() {
	// Inicializamos la matriz de neuronas con len(r.NeuronasPorCapa) capas
	neuronas = make([][]*Neurona, len(r.NeuronasPorCapa))
	// Por cada capa hacemos r.NeuronasPorCapa[i] neuronas
	for i := range neuronas {
		neuronas[i] = make([]*Neurona, r.NeuronasPorCapa[i])
	}

	// Inicializamos las neuronas
	numEntradas := 0                        //Nº de entradas que tendrán las neuronas de cada capa
	for k, val := range r.NeuronasPorCapa { // k será el número de capa y val la cantidad de neuronas en esa capa
		for j := 0; j < val; j++ { //j será el nº identificativo de neurona en esa capa

			if k != 0 { // Si NO estamos en la primera capa oculta
				numEntradas = r.NeuronasPorCapa[k-1] + 1 //sumamos 1 para tener en cuenta el bias
			} else {
				numEntradas = len(r.Entradas)
			}

			neuronas[k][j] = &Neurona{
				Pesos:        r.PesosIniciales[k][j],
				DiferencialW: make([]float64, len(r.PesosIniciales[k][j])),
				Entradas:     make([]float64, numEntradas),
			}

			// Define el bias en cada neurona
			neuronas[k][j].Entradas[0] = r.Entradas[0]
		}
	}
}

// Inicializar entradas de la primer capa
func (r *RedNeuronal) inicializarEntradasPrimeraCapa() {
	for i := 0; i < r.NeuronasPorCapa[0]; i++ {
		neuronas[0][i].Entradas = r.Entradas
	}
}

// Propaga los resultados a las neuronas transladando a las entradas de cada capa las salidas de la capa anterior .
// Actualizando también la propiedad salida de cada neurona
// A la primera capa simplemente le asigna las entradas pasadas a la neurona.
func (r *RedNeuronal) Propagar() {

	r.inicializarEntradasPrimeraCapa()

	// Translada la salida de las neuronas de la capa k a las entradas de las neuronas de la capa k+1
	for k := 1; k < len(r.NeuronasPorCapa); k++ { // k es el nº de capa. Comienza en 1 porque las entradas de la capa 0 ya fueron incializados
		for j := 0; j < r.NeuronasPorCapa[k]; j++ { // j es el nº identificativo de neurona en esa capa
			for i, neurona := range neuronas[k-1] { // i es el nº identificativo de neurona de salida de la capa anterior y neurona es la neurona de salida
				// a partir de Entradas[i+1] ya que Entradas[0] es el bias
				neuronas[k][j].Entradas[i+1] = neurona.CalcularSalida()
			}
		}
	}

	// Actualiza los valores de la última capa
	for _, neurona := range neuronas[len(neuronas)-1] {
		neurona.CalcularSalida()
	}

}

// Devuelve un vector con el calculo de la salida de cada neurona
// de la última capa.
func (r *RedNeuronal) ObtenerSalida() []float64 {
	var res []float64
	for _, val := range neuronas[len(r.NeuronasPorCapa)-1] {
		res = append(res, val.Salida)
	}
	return res
}

// Modificación diferenciales de los pesos de la red siguiendo la siguiente formula:
// W(i,j,k) = CAprendizaje * Salida(i,k-1) * δ(j,k) = *CAprendizaje * Entrada(j,k)  δ(j,k)
// Siendo W(Salida, Entrada, Capa)
// Y luego actualiza todos los pesos.
// Se debe haber modificado las salidas previemente en la propagación

func (r *RedNeuronal) Retropropagar() {
	//Actualizamos los diferneciales de W
	for k := len(neuronas) - 1; k >= 0; k-- { // k es el nº de capa. Recorres las capas de la última a la primera
		for j := range neuronas[k] { // j es el nº identificativo de la neurona dentro de la capa k
			// Actualizamos los sigma de la neurona actual
			r.actualizarSigma(k, j)
			for i := range neuronas[k][j].Pesos { // i es la neurona de entrada de la conexión w

				// Modificación del peso
				if i == 0 { //Estamos modificando el peso de la conexión con el bias
					neuronas[k][j].DiferencialW[i] += r.CAprendizaje * neuronas[k][j].Sigma
				} else { //Estamos modifcando el peso de la conexión con otra nerurona
					neuronas[k][j].DiferencialW[i] += r.CAprendizaje * neuronas[k][j].Entradas[i] * neuronas[k][j].Sigma
				}
			}
		}
	}
	r.actualizarPesos()

}

// Actualiza los pesos de todas las neuronas utilizando el diferencial de peso
func (r *RedNeuronal) actualizarPesos() {
	for k := len(neuronas) - 1; k >= 0; k-- { // k es el nº de capa. Recorres las capas de la última a la primera
		for j := range neuronas[k] { // j es el nº identificativo de la neurona dentro de la capa k
			for i := range neuronas[k][j].Pesos { // i es la neurona de entrada de la conexión w
				// Modificación del peso
				neuronas[k][j].Pesos[i] += neuronas[k][j].DiferencialW[i]

			}
		}
	}
}

// Actualiza el sigma de la neurona que se encuentra en la capa k, posición j
func (r *RedNeuronal) actualizarSigma(k, j int) {
	if k == len(neuronas)-1 { // Para la capa de salida
		// Modificación del sigma
		neuronas[k][j].Sigma = (r.SalidasDeseadas[j] - neuronas[k][j].Salida) * neuronas[k][j].Salida * (1 - neuronas[k][j].Salida)
	} else { // Para las capas ocultas
		var sumatorio float64          // ∑ (w(k+1,j,q) * δ(k+1,q)) siendo k la capa, j la neurona de salida y q la neurona de entrada
		for q := range neuronas[k+1] { // q es el nº identificativo de neurona de la capa k+1
			sumatorio += neuronas[k+1][q].Pesos[j+1] * neuronas[k+1][q].Sigma
		}

		// Modificación del sigma
		neuronas[k][j].Sigma = sumatorio * neuronas[k][j].Salida * (1 - neuronas[k][j].Salida)
	}

}

// Entrena la red con n epoch
func (r *RedNeuronal) Entrenar(n int) {
	for i := 0; i < n; i++ {
		r.Propagar()
		r.Retropropagar()
	}
}
