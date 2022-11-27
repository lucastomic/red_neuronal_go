package models

import "math"

type Neurona struct {
	// Entradas de la neurona. El 0 es el bias
	Entradas []float64
	// Úlitma salida calculada de la neurona
	Salida float64
	// Pesos de todas las conexiones ENTRANTES de la nerurona. El 0 es el bias
	Pesos []float64
	// Sigma de la neurona
	Sigma float64
	// Diferencial de W que se aplicará respectivamente a cada peso
	DiferencialW []float64
}

// Calcula la salída de la neurona aplicando la función de activación
// a la neta. Tambien actualiza la propiedad Salida de la neurona
func (p *Neurona) CalcularSalida() float64 {
	neta := p.caclularNeta()
	p.Salida = sigmoide(neta)
	return p.Salida
}

// Función sigmoide con x = val
func sigmoide(val float64) float64 {
	cuadrado := math.Pow(math.E, -val)
	res := 1 / (1 + cuadrado)
	return res
}

// Devuelve la neta de la neurona
func (p *Neurona) caclularNeta() float64 {
	var res float64 = 0
	for i := range p.Entradas {
		res += p.Entradas[i] * p.Pesos[i]
	}
	return res
}
