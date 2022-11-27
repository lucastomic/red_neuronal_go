package main

import (
	"fmt"

	"github.com/lucastomic/red_neuronal_go/models"
)

var pesos [][][]float64 = [][][]float64{
	{
		{
			2.92120036669075,   //w(0,1)
			-0.892485496588051, //w(1,1)
			1.04047445114702,   //w(2,1)
		},
		{
			-1.10742220655084,
			-1.19463555514812,
			4.97026676312089,
		},
	},
	{
		{
			4.21899800654501,
			3.12101051444188,
			-1.57787738600746,
		},
	},
}

var bias float64 = 1

var entradas []float64 = []float64{
	bias,
	0.537188362330198,
	0.360478304326534,
}

func main() {
	red := models.RedNeuronal{
		NeuronasPorCapa: []int{2, 1},
		PesosIniciales:  pesos,
		Entradas:        entradas,
		CAprendizaje:    0.178136822069064,
		SalidasDeseadas: []float64{0.608886775095016},
	}
	red.InitPerceptron()

	for i := 0; i < 1; i++ {
		red.Propagar()
		fmt.Println(red.ObtenerSalida())
		red.Retropropagar()
	}
	// red.Entrenar(1000)

}
