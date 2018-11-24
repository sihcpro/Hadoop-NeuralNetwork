import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

import org.neuroph.core.*;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.input.WeightedSum;
import org.neuroph.core.transfer.*;
//import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.comp.neuron.InputNeuron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.ConnectionFactory;

public class Trial {
	private static List< List<Integer> > data = new ArrayList< List<Integer> >();
	private static List< List<Integer> > test = new ArrayList< List<Integer> >();
	private static List<String> name = new ArrayList<String>();
	
	private static double[][] truthValue = new double[][] {
//		{1,0,0,0,0,0,1,0,1,1}, // 0
//		{0,1,0,0,0,0,0,0,0,0},
//		{0,0,0,1,0,0,0,1,1,0},
//		{0,0,0,0,1,0,0,1,1,0},
//		{0,1,0,1,0,0,0,0,0,0},
//		{0,0,1,0,1,1,1,1,1,0}, // 5
//		{1,0,0,0,0,0,1,1,0,1},
//		{0,1,0,1,0,0,0,0,1,0},
//		{1,0,1,0,0,1,1,1,1,1},
//		{1,0,1,0,1,0,1,0,1,0}, // 9

	
		{1,0,0,0,0,0,1,1}, // 0
		{0,1,0,0,0,0,0,0},
		{0,0,0,1,0,0,1,0},
		{0,0,0,0,1,0,1,0},
		{0,1,0,1,0,0,0,0},
		{0,0,1,0,1,1,1,0}, // 5
		{1,0,0,0,0,0,0,1},
		{0,1,0,1,0,0,1,0},
		{1,0,1,0,0,1,1,1},
		{1,0,1,0,1,0,1,0}, // 9
	};
	
	private static Map<String, Integer> value = new HashMap<String, Integer>();

	private static void init() {
		for(int i = 0; i < 10; i++) {
			value.put(toString(truthValue[i]), i+1);
		}
	}
	
	private static String toString(double[] value) {
		String tmp = "";
		for(double i : value) {
			tmp = tmp + " " + i;
		}
		return tmp;
	}
	
	private static Neuron chooseNeural(int i, int c) {
		Neuron n = new Neuron();
		// Linear, Log, Ramp, Sgn, Sigmoid, Sin, Step, Tanh, Trapezoid
		switch( i%6 ) {
		case 0:
			// 200 0.15
			n.setTransferFunction(new Linear(1.0));
			break;
		case 1:
			// 200 0.3
			n.setTransferFunction(new Ramp());
			break;
		case 2:
			// 200 0
			n.setTransferFunction(new Sgn());
			break;
		case 3:
			// 200 0.20
			n.setTransferFunction(new Sigmoid(1.0));
			break;
		case 4:
			// 200 0.35
			n.setTransferFunction(new Step());
			break;
		case 5:
			// 200 0.1
			n.setTransferFunction(new Tanh(1.0));
			break;
		default:
			// 200 0.3
		}
		n.setInputFunction(new WeightedSum());
		return n;
	}
	
	public static void main(String[] args) {
		init();
		
		int kindNeuron[] = new int[] {0,0,0,10,0,0,0};
		
		int inputSize = 784;
		int hiddenSize = 10;
		int outputSize = 10;
		DataSet ds = new DataSet(inputSize, outputSize);

		int maxLen = kindNeuron.length;
		Layer inputLayer = new Layer();
		Layer hiddenLayerOne = new Layer();
		Layer outputLayer = new Layer();
		
		if( maxLen == 0 ) {
			kindNeuron = new int[] {0};
		}
		for(int i = 0; i < inputSize; i++) {
			InputNeuron in = new InputNeuron();
			in.setInputFunction(new WeightedSum());
			inputLayer.addNeuron(in);
		}
		int amountNeuronHidden = 0;
		for(int i = 0; i < 5; i++) {
			for(int j = 0; j < kindNeuron[i%maxLen]; j++) {
				if( amountNeuronHidden >= 40 )
					continue;
				hiddenLayerOne.addNeuron(chooseNeural(i, hiddenSize));
				amountNeuronHidden++;
			}
		}
		for(; amountNeuronHidden < hiddenSize; amountNeuronHidden++) {
			hiddenLayerOne.addNeuron(chooseNeural(5, hiddenSize));
		}
		for(int i = 0; i < outputSize; i++) {
			outputLayer.addNeuron(chooseNeural(3, outputSize));
		}

		try {
			System.out.println("Read file");
//			readfile("data_small.csv", 2000);
			readfile("train.csv", 1000, 25);
			System.out.println("Data size: "+data.size()+" * "+data.get(0).size());

//			System.out.println(data.toString());
//			String a[] = data.toString().split("],");
//			List< List<Integer> > data2 = new ArrayList< List<Integer> >();
//			for(String s : a) {
//				s = s.replace("[", "");
//				s = s.replace("]", "");
//				s = s.replace(" ", "");
//				System.out.println(s);
//				String s2[] = s.split(",");
//				List<Integer> list = new ArrayList<Integer>();
//				for(String i : s2) {
//					System.out.print(i+" ");
//					list.add(Integer.parseInt(i));
//				}
//				System.out.println();
//				data2.add(list);
//			}
//			System.out.println(data2);

			System.out.println("Trainfer Data");
			Prob data_train = changeTypeProblem(data);
//			for(int i = 0; i < 28; i++) {
//				for(int j = 0; j < 28; j++) {
//					System.out.print(data_train.X[0][i*28+j]+"\t");
//				}
//				System.out.println();
//			}
			System.out.println("Data_train size: "+data_train.X.length+" * "+data_train.X[0].length);
			System.out.println("Add data set");
//			double data_train_y[] = new double[outputSize];
//			for(int i = 0; i < outputSize; i++) {
//				data_train_y[i] = 0;
//			}
			
			double data_train_y[] = new double[outputSize];
			for(int i = 0; i < outputSize; i++) {
				data_train_y[i] = 0;
			}
			int value = 0;
			for(int i = 0; i < data_train.X.length; i++) {
				value = (int)data_train.y[i];
				data_train_y[ value ] = 10.0;
//				for(int k = 0; k < outputSize; k++) {
	//				System.out.print(data_train_y[k] + " ");
	//			}
	//			System.out.println("---");
				for(int k = 0; k < 784; k++) {
					data_train.X[i][k] /= 255;
				}
	//			System.out.println("---");
				ds.addRow(data_train.X[i], data_train_y);
				data_train_y[ value ] = 0;
			}
		} catch (Exception e) {
			System.out.println("Error in Adding");
			e.printStackTrace();
		}
		System.out.println("Training data");

		Sigmoid ss = new Sigmoid(1);
		System.out.println("--- " + ss.getOutput(1.0) + " ---");
		NeuralNetwork<BackPropagation> ann = new NeuralNetwork<BackPropagation>();
		ann.addLayer(0, inputLayer);
		ann.addLayer(1, hiddenLayerOne);
		ConnectionFactory.fullConnect(ann.getLayerAt(0), ann.getLayerAt(1));
//		ann.addLayer(2, hiddenLayerTwo);
		ann.addLayer(2, outputLayer);
		ConnectionFactory.fullConnect(ann.getLayerAt(1), ann.getLayerAt(2));
//		ann.addLayer(3, outputLayer);
//		ConnectionFactory.fullConnect(ann.getLayerAt(2), ann.getLayerAt(3));
		ConnectionFactory.fullConnect(ann.getLayerAt(0), ann.getLayerAt(ann.getLayersCount()-1), false);
		ann.setInputNeurons(inputLayer.getNeurons());
		ann.setOutputNeurons(outputLayer.getNeurons());
		ann.randomizeWeights(-1.0, 1.0);
		BackPropagation bP = new BackPropagation();
		bP.setMaxIterations(1000);
		System.out.println(bP.getLearningRate());
		long startTime = System.nanoTime();
		ann.learn(ds, bP);
		long endTime = System.nanoTime();
		System.out.println("Running time = "+(endTime - startTime)/1000000000+" s");
//		for(int i = 0; i < 40; i++)
//			System.out.println(ann.getLayerAt(2).getNeuronAt(5).getInputConnections()[i].getWeight());
		
		
//		System.out.println(ann.);
		

//		MultiLayerPerceptron ann = new MultiLayerPerceptron(TransferFunctionType.SIGMOID, inputSize, inputSize/16, outputSize);
//		MomentumBackpropagation bp = new MomentumBackpropagation();
//		
//		double LEARINING_RATE = 0.01;
//		double MOMENTUM = 0.025;
//		bp.setLearningRate(LEARINING_RATE);
//		bp.setMomentum(MOMENTUM);
//		
//		ann.setLearningRule(bp);
//		long startTime = System.nanoTime();
//		ann.learn(ds);
//		long endTime = System.nanoTime();
//		System.out.println("Running time = "+(endTime - startTime)/1000000000+" s");
//		String NNET_NAME = "sihc";
//		ann.save(NNET_NAME + "last");

//		23
        
        
        
        
        
		try {
			System.out.println("Trainfer Test");
			Prob data_test = changeTypeProblem(test);
			System.out.println("Start get results");
			boolean result[] = new boolean[data_test.X.length];
			int count = 0;
//			boolean result2[] = new boolean[data_test.X.length];
//			int count2 = 0;
			for(int i = 0; i < data_test.X.length; i++) {
				ann.setInput(data_test.X[i]);
				ann.calculate();
				if(i == 0) {
					System.out.print("\nNetInput0\t");
					for(int k = 0; k < 784; k++) {
						System.out.print(ann.getLayerAt(0).getNeuronAt(k).getNetInput() + "\t");
					}
					System.out.print("\nOutput 0\t");
					for(int k = 0; k < 784; k++) {
						System.out.print(ann.getLayerAt(0).getNeuronAt(k).getOutput() + "\t");
					}
					System.out.print("\nWeight\t\t");
					for(int k = 0; k < 784; k++) {
						System.out.print(ann.getLayerAt(1).getNeuronAt(0).getWeights()[k] + "\t");
					}
					System.out.print("\nNetInput1\t");
					for(int k = 0; k < hiddenSize; k++) {
						System.out.print(ann.getLayerAt(1).getNeuronAt(k).getNetInput() + "\t");
					}
					System.out.print("\nOutput 1\t");
					for(int k = 0; k < hiddenSize; k++) {
						System.out.print(ann.getLayerAt(1).getNeuronAt(k).getOutput() + "\t");
					}
					System.out.print("\nNetInput2\t");
					for(int k = 0; k < 10; k++) {
						System.out.print(ann.getLayerAt(2).getNeuronAt(k).getNetInput() + "\t");
					}
					System.out.print("\nOutput 2\t");
					for(int k = 0; k < 10; k++) {
						System.out.print(ann.getLayerAt(2).getNeuronAt(k).getOutput() + "\t");
					}
					System.out.println();
				}
				double[] networkOutputOne = ann.getOutput();
				
				int answer = maxi(networkOutputOne);
				System.out.print("Result: "+data_test.y[i]+" vs "+answer+"  ");
				result[i] = answer == data_test.y[i];
				if(result[i])
					count++;
				for(int j= 0; j < 10; j++) {
					System.out.format("%.6f ", networkOutputOne[j]);
				}
				System.out.println();
				
				
				
				
				
//				System.out.println("Result: "+data_test.y[i]+" vs "+networkOutputOne[0]);

//				System.out.print("Result: "+data_test.y[i]+"      vs      ");
//				if( !value.containsKey(toString(networkOutputOne)) ) {
//					System.out.print("non  |  ");
//					System.out.print(toString(truthValue[(int)data_test.y[i]])+" vs ");
//					System.out.print(toString(networkOutputOne));
//				} else {
//					double answer = value.get(toString(networkOutputOne)) - 1;
//					System.out.print(answer);
//					result[i] = answer == data_test.y[i];
//					if( result[i] ) count++;
//				}
//				result2[i] = nearest(networkOutputOne) == data_test.y[i];
//				if( result2[i] ) count2++;
//				System.out.println();
				
//				double max_predict = networkOutputOne[0];
//				int output = 0;
//				System.out.print(networkOutputOne[0]);
//				for(int j = 1; j < networkOutputOne.length; j++) {
//					System.out.print(" ; "+networkOutputOne[j]);
//					if( networkOutputOne[j] > max_predict ) {
//						max_predict = networkOutputOne[j];
//						output = j;
//					}
//				}
//				System.out.println();
//								
//				result[i] = data_test.y[i] == output;
//				System.out.println("     Result: "+data_test.y[i]+" vs "+output);
			}
			System.out.println("Done");
			System.out.println("Result : "+((float)count / result.length));
//			System.out.println("Result2: "+((float)count2/ result.length));
		} catch (Exception e) {
			System.out.println("Error in get result");
			e.printStackTrace();
		}
	}
	
	
	
	
	
	
	
	
	private static int maxi(double []result) {
		int num = 0;
		double proba = 0;
		for(int i= 0; i < 10; i++) {
			if(proba < result[i]) {
				proba = result[i];
				num = i;
			}
		}		
		return num;
	}
	
	
//	private static int nearest(double []result) {
//		int num = 0, count = 0, bestmatch = 0;
//		double x, y;
//		try {
//			for(int i = 0; i < 10; i++) {
//				count = result.length;
//				for(int j = 0; j < result.length; j++) {
//					x = result[j];
//					y = truthValue[i][j];
//					if(x * y == 0 && x + y != 0) {
//						count--;
//					}
//				}
//				if( count > bestmatch ) {
//					bestmatch = count;
//					num = i;
//				}
//			}
//			return num;
//		} catch (Exception e) {
//			return num;
//		}
//	}	
	
	
	public static void readfile(String file_link, int train_leng, int test_leng) throws Exception{
//		List<Integer> list = new ArrayList<Integer>();
		int limit = train_leng + test_leng;
		File file = new File(file_link);
		BufferedReader reader = null;
		
		try {
		    reader = new BufferedReader(new FileReader(file));
		    String text = null;
		    
		    int count = -1;
		    while ((text = reader.readLine()) != null && count < limit) {
		    	if(count == -1) {
		    		name.add(text);
		    		count++;
		    	} else {
		    		String[] tmp = text.split(",");
//			    		System.out.println("tmp ");
//			    		System.out.println(tmp.length);
		    		data.add(new ArrayList<Integer>());
		    		for(String i : tmp) {
		    			data.get(count).add(Integer.parseInt(i));
//			    			((Object) data).elementAt(count).add(Integer.parseInt(i));
		    		}
		    		count++;
		    	}
		    }

		    
			for(int i = 0; i < test_leng; i++) {
				test.add(data.get(data.size()-1));
				data.remove(data.size()-1);
			}
		} catch (FileNotFoundException e) {
		    e.printStackTrace();
		} catch (IOException e) {
		    e.printStackTrace();
		} finally {
		    try {
		        if (reader != null) {
		            reader.close();
		        }
		    } catch (IOException e) {
		    }
		}
	}
	
	public static Prob changeTypeProblem(List< List<Integer> > data) throws Exception{
		Prob prob = new Prob();

		try {
			if(data == null || data.size() == 0) {
				System.out.println("Error data is empty!");
				return new Prob();
			} else if(data.get(0).size() == 0){
				System.out.println("Error data[0] is empty!");
				return new Prob();
			}
			
			int leng = data.get(0).size() - 1;
			prob.X = new double[data.size()][leng];
			prob.y = new double[data.size()];
			
			for(int i = 0; i < data.size(); i++) {
				prob.y[i] = data.get(i).get(0);

				for(int j = 0; j < leng; j++) {
					prob.X[i][j] = data.get(i).get(j+1);
				}
			}
		} catch (Exception e) {
			System.out.println("Error in make Problem");
			System.out.println(e.toString());
			e.printStackTrace();
			throw e;
		}
		
		return prob;
	}
	
	public static class Prob {
		public String[] name;
		public double[][] X;
		public double[] y;
		
//		Prob Prob(int lengX, int lengY) {
//			this.name = new String[lengY];
//			this.X = new float[lengX][lengY];
//			this.y = new float[lengX];
//			return this;
//		}
	}
}
