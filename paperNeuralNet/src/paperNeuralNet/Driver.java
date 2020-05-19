package paperNeuralNet;

//import java.util.Random;

public class Driver {
	
	//Yapay sinir aglarinda, aga verilecek girdi degerleri sayisal olmalilar
	//Makaleler title, abstract ve references bolumerindeki kelime sayilarina gore siniflandiriliyor
	//Buna gore kurallar asagida verilmistir:
	
	
	//titledaki kelime sayisi x<=10 ise bu deger 0, x>10 ise 1 olarak aliniyor
	
	//abstractdeki kelime sayisi y<200 ise 0, y>=200 ise 1 aliniyor
	
	//referencesdaki kelime sayisi z<500 ise 0, z>=500 ise 1 aliniyor
	
	//cikis degerinin yani makalenin ait oldugu sinifin belirlenmesi asagidaki gibidir
	
	//title degeri 0 ise cikis(sinif) 0 olur
	
	//title ve abstract degerlerinin her ikisi 1 ise cikis(sinif) 2 olur
	
	//title 1 ve abstract 0 ise references degerine bakilir
	
		//--references 0 ise cikis 0
	    //--references 1 ise cikis 1 olur
	 
	
	static void NeuralNet(double B1, double G1[]) {
		/*Random r = new Random();
		
		double[] agirlikGA = {r.nextDouble(), - r.nextDouble(), r.nextDouble(), - r.nextDouble(), r.nextDouble(), -r.nextDouble(), r.nextDouble(), - r.nextDouble(), r.nextDouble()};
		double[] agirlikAC = {r.nextDouble(), r.nextDouble(), r.nextDouble()};
		
		double[] esikA = {r.nextDouble(), -r.nextDouble(), r.nextDouble()};*/
		//agirliklari ve esik degerlerini rastgele atamayi denedim fakat iyi bir sonuc vermiyor
		
		double[] agirlikGA = {0.129952, -0.923123, 0.570345, - 0.328932, 0.129952, -0.923123, 0.570345, - 0.328932, 0.570345};
		double[] agirlikAC = {0.164732, 0.752621, 0.570345};
		
		double[] esikA = {0.341332, -0.115223, 0.164732};
		double[] esikC = {-0.993423};
		
		//ogrenme katsayisi ve momentum, en iyi degerlere gore optimize edildi
		double ogrenmeKS = 0.2; 
		double momentum = 0.3; 
		
		double hata = 0;
		double cikti = 0 , ciktiyeni =0;
		
		//Bu agin yapisi:
		//Giris katmani: 3 noron
		//Ara katman: 3 noron
		//Cikis katmani: 1 noron
		
		
		//System.out.println("Egitiliyor...");
	    
		for (int i = 0; i<400; i++) {// en iyi sonuc dongu sayisi= 400 icin aliniyor
			
			
			
			double net1 = (B1* agirlikGA[0]) + (B1 * agirlikGA[1]) + (B1 * agirlikGA[2])  + (1* esikA[0]);
			double net2 = (B1* agirlikGA[3]) + (B1 * agirlikGA[4]) + (B1 * agirlikGA[5])+ (1* esikA[1]);
			double net3 = (B1* agirlikGA[6]) + (B1 * agirlikGA[7]) + (B1 * agirlikGA[8])+ (1* esikA[2]);
			
			
			//sigmoid fonksiyonu kullanildi
			double cikis1 = ActivationFunctions.sigmoid(net1);
			double cikis2 = ActivationFunctions.sigmoid(net2);
			double cikis3 = ActivationFunctions.sigmoid(net3);
			
			double net = (1*esikC[0])+ (cikis1* agirlikAC[0]) + (cikis2 * agirlikAC[1]) + (cikis3 * agirlikAC[2]);
		     cikti= ActivationFunctions.sigmoid(net);
			hata = hata- cikti;
		
		//Backpropagation
		//Ara katman - cikti katmani arasi
		
		double degisim1= Derivatives.dSigmoid(cikti) * hata;
		
		
		double degisimMiktari1 = ogrenmeKS * degisim1* cikis1* momentum* G1[0];
		double degisimMiktari2 = ogrenmeKS * degisim1* cikis2* momentum* G1[1];
		double degisimMiktari3 = ogrenmeKS * degisim1* cikis3* momentum* G1[2];
		double esikCdegisim= ogrenmeKS * B1 *degisim1;
		
		
		double agirlikTemp1 = agirlikAC[0];
		agirlikAC[0]= agirlikAC[0]- degisimMiktari1;
		double agirlikTemp2 = agirlikAC[1];
		agirlikAC[1]= agirlikAC[1]- degisimMiktari2;
		double agirlikTemp3 = agirlikAC[2];
		agirlikAC[2]= agirlikAC[2]- degisimMiktari3;
		esikC[0]= esikC[0] + esikCdegisim; 
		
		//Girdi katmani - ara katman
		
		double degisimA1 = Derivatives.dSigmoid(cikti) *degisim1* agirlikTemp1;
		double degisimA2 = Derivatives.dSigmoid(cikti) *degisim1* agirlikTemp2;
		double degisimA3 = Derivatives.dSigmoid(cikti) *degisim1* agirlikTemp3;
		
		double agirlikDegisim1= ogrenmeKS*degisimA1*B1+ momentum* G1[0];
		double agirlikDegisim2= ogrenmeKS*degisimA1*B1+ momentum* G1[1];
		double agirlikDegisim12= ogrenmeKS*degisimA1*B1+ momentum* G1[2];
		
		double agirlikDegisim3= ogrenmeKS*degisimA2*B1+ momentum* G1[0];
		double agirlikDegisim4= ogrenmeKS*degisimA2*B1+ momentum* G1[1];
		double agirlikDegisim34= ogrenmeKS*degisimA2*B1+ momentum* G1[2];
		
		double agirlikDegisim5= ogrenmeKS*degisimA3*B1+ momentum* G1[0];
		double agirlikDegisim6= ogrenmeKS*degisimA3*B1+ momentum* G1[1];
		double agirlikDegisim56= ogrenmeKS*degisimA3*B1+ momentum* G1[2];
		
		
		
		double esikDegisim1 = ogrenmeKS*1 + degisimA1;
		double esikDegisim2 = ogrenmeKS*1 + degisimA2;
		double esikDegisim3 = ogrenmeKS*1 + degisimA3;
		
		
		agirlikGA[0]= agirlikGA[0] + agirlikDegisim1;
		agirlikGA[1]= agirlikGA[1] + agirlikDegisim2;
		agirlikGA[2]= agirlikGA[2] + agirlikDegisim12;
		agirlikGA[3]= agirlikGA[3] + agirlikDegisim3;
		agirlikGA[4]= agirlikGA[4] + agirlikDegisim4;
		agirlikGA[5]= agirlikGA[5] + agirlikDegisim34;
		agirlikGA[6]= agirlikGA[6] + agirlikDegisim5;
		agirlikGA[7]= agirlikGA[7] + agirlikDegisim6;
		agirlikGA[8]= agirlikGA[8] + agirlikDegisim56;
		
		
		
		esikA[0] = esikA[0] + esikDegisim1;
		esikA[1] = esikA[1] + esikDegisim2;
		esikA[2] = esikA[2] + esikDegisim3;
		
		
		
		double netyeni1 = (B1* agirlikGA[0]) + (B1 * agirlikGA[1])+ (B1 * agirlikGA[2]) + (1* esikA[0]);
		double netyeni2 = (B1* agirlikGA[3]) + (B1 * agirlikGA[4]) + (B1 * agirlikGA[5])+ (1* esikA[1]);
		double netyeni3 = (B1* agirlikGA[6]) + (B1 * agirlikGA[7]) + (B1 * agirlikGA[8])+ (1* esikA[2]);
		
		double cikisyeni1 = ActivationFunctions.sigmoid(netyeni1);
		double cikisyeni2 = ActivationFunctions.sigmoid(netyeni2);
		double cikisyeni3 = ActivationFunctions.sigmoid(netyeni3);
		
		double netyeni = (1*esikC[0])+ (cikisyeni1* agirlikAC[0]) + (cikisyeni2 * agirlikAC[1])+ (cikisyeni3 * agirlikAC[2]);
		ciktiyeni= ActivationFunctions.sigmoid(netyeni);
		
		
		
		
		}
		
        //System.out.println("Tahmin: ");
		
		System.out.println(ciktiyeni);
		
	}
	
	public static void main(String[] args) {
		
		//8 farkli durum olusur
		double[] G1 = {0, 1,1};
		double[] G2 = {1, 0 ,0};
		double[] G3 = {1, 0, 1};
		double[] G4 = {0,0,1};
		double[] G5 = {0,1,0};
		double[] G6 = {0,0,0};
		double[] G7 = {1,1,1};
		double[] G8 = {1,1,0};
		
		
		
		
		double B1= 0;
		double B2= 0;
		double B3= 1;
		double B4= 0;
		double B5 = 0;
		double B6 = 0;
		double B7 =2;
		double B8 = 2;
		
		System.out.print("COVID-19: Challenges to GIS with Big Data ");
		NeuralNet(B1, G1);
		System.out.println();
		System.out.print("Application of Data Mining Classification for Covid-19 Infected Status Using Algortima Naïve Method ");
		NeuralNet(B2, G2);
		System.out.println();
		System.out.print("Automatic Detection of Coronavirus Disease (COVID-19) Using X-ray Images and Deep Convolutional Neural Networks ");
		NeuralNet(B3, G3);
		System.out.println();
		System.out.print("Mapping the landscape of Artificial Intelligence applications against COVID-19 ");
		NeuralNet(B4, G4);
		System.out.println();
		System.out.print("SEIR and Regression Model based COVID-19 outbreak predictions in India ");
		NeuralNet(B5, G5);
		System.out.println();
		System.out.print("Detection of Covid-19 From Chest X-ray Images Using Artificial Intelligence: An Early Review ");
		NeuralNet(B3, G3);
		System.out.println();
		System.out.print("A Study of Knowledge Sharing related to Covid-19 Pandemic in Stack Overflow ");
		NeuralNet(B3, G3);
		System.out.println();
		System.out.print("CORD-19: The COVID-19 Open Research Dataset ");
		NeuralNet(B4, G4);
		System.out.println();
		System.out.print("Rapidly Bootstrapping a Question Answering Dataset for COVID-19 ");
		NeuralNet(B4, G4);
		System.out.println();
		System.out.print("Target specific mining of COVID-19 scholarly articles using one-class approach ");
		NeuralNet(B1, G1);
		System.out.println();
		System.out.print("Integrated Time Series Summarization and Prediction Algorithm and its Application to COVID-19 Data Mining ");
		NeuralNet(B3, G3);
		System.out.println();
		System.out.print("deepMINE - Natural Language Processing based Automatic Literature Mining and Research Summarization for Early-Stage Comprehension in Pandemic Situations specifically for COVID-19 ");
		NeuralNet(B8, G8);
		System.out.println();
		System.out.print("A Novel Approach of CT Images Feature Analysis and Prediction to Screen for Corona Virus Disease (COVID-19) ");
		NeuralNet(B3, G3);
		System.out.println();
		System.out.print("Drugs and the renin-angiotensin system in covid-19 ");
		NeuralNet(B5, G5);
		System.out.println();
		System.out.print("Clinical features of covid-19 ");
		NeuralNet(B4, G4);
		System.out.println();
		System.out.print("COVID-19 Related Research by Data Mining in Single Cell Transcriptome Profiles ");
		NeuralNet(B2, G2);
		System.out.println();
		System.out.print("Acute myocardial injury is common in patients with covid-19 and impairs their prognosis ");
		NeuralNet(B7, G7);
		System.out.println();
		System.out.print("LABORATORY INFORMATION SYSTEM REQUIREMENTS TO MANAGE THE COVID-19 PANDEMIC : A REPORT FROM THE B ELGIAN NATIONAL REFERENCE TESTING CENTER ");
		NeuralNet(B8, G8);
		System.out.println();
		System.out.print("Is traditional Chinese medicine useful in the treatment of COVID-19? ");
		NeuralNet(B8, G8);
		System.out.println();
		System.out.print("Potential COVID-19 protease inhibitors: Repurposing FDA-approved drugs ");
		NeuralNet(B6, G6);
		
		
		
		
		
	}

}
