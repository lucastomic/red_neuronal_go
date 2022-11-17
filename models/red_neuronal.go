package models

type RedNeuronal struct {
	// Arreglo con la cantidad de neuronas que tendrá cada capa (sin contar la
	// capa de entrada). La cantidad de capas es len(NeuronasPorCapa)
	NeuronasPorCapa []int
	// Entradas de la red neuronal. La primera es el bias.
	Entradas []float64
	// Los pesos de cada una de las conexiones. En la siguiente estructura:
	// [nº de capa][nº de neurona de dicha capa][cada conexión de dicha neurona]
	// Pesos debe tener length de [len(numeroCapas)] [numeroCapas[i]] [numeroCapas[i-1]]
	Pesos [][][]float64
	// Coeficiente de aprendizaje de la red neuronal
	CAprendizaje float64
}

// Neuronas del perceptron. La estructura es la siguiente:
// [nº de capa][nº identificativo de la neurona en dicha capa]
var neuronas [][]*Neurona

// Perceptrones de la red. Su estructura es la siguiente:
// [nº de capa][nº identificativo de la neurona en dicha capa]

// Incializa todos los perceptones de la red neuronal
func (r *RedNeuronal) InitPerceptrones() {
	// Inicializamos la matriz de neuronas
	neuronas = make([][]*Neurona, len(r.NeuronasPorCapa))
	for i := range neuronas {
		neuronas[i] = make([]*Neurona, r.NeuronasPorCapa[i])
	}

	numEntradas := 0                        //Nº de entradas que tendrán los perceptones de cada capa
	for i, val := range r.NeuronasPorCapa { // i será el número de capa y val la cantidad de neuronas en esa capa
		for j := 0; j < val; j++ { //j será el nº identificativo de neurona en esa capa

			if i != 0 {
				numEntradas = r.NeuronasPorCapa[i-1] + 1 //sumamos 1 para tener en cuenta el bias
			} else {
				numEntradas = len(r.Entradas)
			}

			neuronas[i][j] = &Neurona{
				Pesos:    r.Pesos[i][j],
				Entradas: make([]float64, numEntradas),
			}

			// Define el bias
			neuronas[i][j].Entradas[0] = r.Entradas[0]
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

	for i := 1; i < len(r.NeuronasPorCapa); i++ { // i es el nº de capa. Comienza en 1 porque las entradas de la capa 0 ya fueron incializados
		for j := 0; j < r.NeuronasPorCapa[i]; j++ { // j es el nº identificativo de neurona en esa capa
			for z, neurona := range neuronas[i-1] { // z es el nº identificativo de neurona de la capa anterior y neurona es la neurona anterior
				neuronas[i][j].Entradas[z+1] = neurona.CalcularSalida()
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
	indiceUltimaCapa := len(r.NeuronasPorCapa) - 1
	var res []float64
	for _, val := range neuronas[indiceUltimaCapa] {
		res = append(res, val.Salida)
	}
	return res
}
