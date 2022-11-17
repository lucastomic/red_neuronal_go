package main

import (
	"fmt"

	"github.com/lucastomic/red_neuronal_go/models"
)

var pesos [][][]float64 = [][][]float64{
	{
		{
			-2.089324442, //w(0,1)
			2.866413163,  //w(1,1)
			2.156065272,  //w(2,1)
		},
		{
			4.030276446,
			1.443913959,
			4.376297968,
		},
	},
	{
		{
			-1.895719238,
			-4.127540295,
			1.25208918,
		},
	},
}

var bias float64 = 1

var entradas []float64 = []float64{
	bias,
	0.196306987,
	0.905595995,
}

func main() {
	red := models.RedNeuronal{
		NeuronasPorCapa: []int{2, 1},
		Pesos:           pesos,
		Entradas:        entradas,
	}
	red.InitPerceptrones()
	red.Propagar()
	fmt.Println(red.ObtenerSalida())

}
