package paperNeuralNet;


public class Derivatives {
	//Backpropagation sirasinda aktivasyon fonksiyonlarinin turevleri kullanilacak
	
	static double dSigmoid(double x) {
		return ActivationFunctions.sigmoid(x)* (1 - ActivationFunctions.sigmoid(x));
		
	}

}
