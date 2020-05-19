package paperNeuralNet;

public class ActivationFunctions {
	//Problem icin hangi fonksiyonun kullanilmasi gerektigi bilinmemektedir. Deneme yoluyla en uygun fonksiyon bulunacaktir.
	
	static final double e = 2.71; //e sayisi tanimlaniyor, 
	//programin calismasi sirasinda degerinin degismesinin onune gecmek icin final anahtar kelimesi kullanildi
	
	
	static double sigmoid(double x) {
		
		return 1 / (1 + (Math.pow(e, -x)));
		
	}
	
	static double linear(double x) {
		return x;
	}
	static int step (double threshold, double x) {
		if(x > threshold) {
			return 1;
		}
		return 0;
	}
	static double sin(double x) {
		return Math.sin(x);
	}
	
	static double threshold(double x) {
		
		if(x<=0) {
			return 0;
		}
		
		else if(0 < x && x < 1 ) {
			return x;
		}
		
		else {  //  if(x>=1)
			return 1;
			
		}
	}
	
     static int _threshold(double x, double threshold) {
		
		if( x <= threshold) {
			return 0;
		}
		

		
		else {  
			return 1;
			
		}
	}
	
	static double hyperbolicTangent(double x) {
		return (Math.pow(e, x) + Math.pow(e, -x)) / (Math.pow(e, x) - Math.pow(e, -x));
	}

}