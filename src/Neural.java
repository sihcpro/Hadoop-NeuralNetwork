import java.io.BufferedReader;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.neuroph.core.*;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.input.WeightedSum;
import org.neuroph.core.transfer.*;
import org.neuroph.nnet.comp.neuron.InputNeuron;
import org.neuroph.nnet.comp.neuron.ThresholdNeuron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.ConnectionFactory;

public class Neural implements LearningEventListener {
	private static Prob data_train = null;
	private static Prob data_test = null;
	private static List<String> name = new ArrayList<String>();
	private static int totalLine = 950;
	private static double testRatio = 0.9;
	private static int testCount = (int)(totalLine * testRatio);
	private static int validateCount = totalLine - testCount;
	private static int maxIterations = 10;
	private static FileSystem fs = null;
	private static boolean isHadoop = false;
	private static int loopCheck = 1;
	
	public static boolean readyData() {
		if( data_train != null )
			return data_train.X.length > 0;
		else
			return false;
	}
	
	Neural(){}
	
	Neural(int total_line, double test_ratio){
		totalLine = total_line;
		testRatio = test_ratio;
	}

	Neural(int total_line, double test_ratio, int max_interation, int loop){
		totalLine = total_line;
		testRatio = test_ratio;
		maxIterations = max_interation;
		loopCheck = loop;
	}

	Neural(int total_line, double test_ratio, int max_interation, int loop, FileSystem fs_in){
		totalLine = total_line;
		testRatio = test_ratio;
		maxIterations = max_interation;
		loopCheck = loop;
		fs = fs_in;
		isHadoop = true;
	}

	public static void main(String[] args) {
		int neuronSet[][] = new int[][] {	{ 0,0,0,0,0,0,0,0,0,0 },
											{ 0,4,0,2,3,0,3,0,1,5 },
											{ 2,1,2,2,4,5,2,1,2,2 },
										};
		Neural a = new Neural(9000, 0.9, 20, 1);
		try {
			System.out.println("Read data");
			init("train.csv");
		} catch (Exception e) {
			System.out.println("Read file error!");
		}
		Result output = a.run(neuronSet[0]);
		Result output2 = a.run(neuronSet[1]);
		Result output3 = a.run(neuronSet[2]);
		System.out.println(output.result);
		System.out.println(output2.result);
		System.out.println(output3.result);
	}
	
	public static void init(String file_link) throws Exception {
		if( readyData() )
			return;
		List<List<Double>> data = new ArrayList<List<Double>>();
		List<List<Double>> test = new ArrayList<List<Double>>();
		try {
			readfile(file_link, data, test);
			data_train = changeTypeProblem(data);
			data = null;
			data_test = changeTypeProblem(test);
			test = null;
		} catch (Exception e) {
			System.out.println("Error in init data");
			e.printStackTrace();
			throw e;
		}
		System.out.println("Data size: " + data_train.X.length + " * " + data_train.X[0].length);
		System.out.println("Convert Data");
	}
	
	public Result run(int neuronSet[]) {
		Result best = new Result(-2, "");
		for(int i = 0; i < loopCheck; i++) {
			Result tmp = runSingle(neuronSet);
			if( tmp.point > best.point ) {
				best = tmp;
			}
		}
		return best;
	}
	
	public Result runSingle(int neuronSet[]) {

		Result output = new Result();
		output.result = "";

		int inputSize = 6;
		int hiddenSize = 10;
		int outputSize = 1;
		DataSet ds = new DataSet(inputSize, outputSize);

		int maxLen = neuronSet.length;
		Layer inputLayer = new Layer();
		Layer hiddenLayerOne = new Layer();
		Layer outputLayer = new Layer();

		if (maxLen == 0) {
			neuronSet = new int[] { 0 };
		}
		for (int i = 0; i < inputSize; i++) {
			InputNeuron in = new InputNeuron();
			in.setInputFunction(new WeightedSum());
			inputLayer.addNeuron(in);
		}
		int amountNeuronHidden = 0;
		for (int i = 0; i < maxLen - 1; i++) {
			for (int j = 0; j < neuronSet[i]; j++) {
				if (amountNeuronHidden >= hiddenSize)
					break;
				hiddenLayerOne.addNeuron(chooseNeuron(i, hiddenSize));
				amountNeuronHidden++;
			}
		}
		for (; amountNeuronHidden < hiddenSize; amountNeuronHidden++) {
			hiddenLayerOne.addNeuron(chooseNeuron(5, hiddenSize));
		}
		for (int i = 0; i < outputSize; i++) {
			outputLayer.addNeuron(chooseNeuron(3, outputSize));
		}

		try {
			System.out.println("Train data: " + testCount + " | Validation data: " + validateCount);

			for (int i = 0; i < testCount; i++) {
				double x = data_train.X[i][0] / 10.0;
				double y = data_train.X[i][1] / 1000.0;
				double[] inp = { x, y, x * x, y * y, Math.sin(x), Math.sin(y) };
				double[] oup = { data_train.y[i] };
				ds.addRow(inp, oup);
			}
		} catch (Exception e) {
			System.out.println("Error in Adding");
			e.printStackTrace();
			output.result += "Error in Adding: " + e.getMessage() + "\n";
			output.result += "Train data: " + ds.size() + "\n";
		}
		System.out.println("Start training");
		
		NeuralNetwork<BackPropagation> ann = new NeuralNetwork<BackPropagation>();
		try {
			ann.addLayer(0, inputLayer);
			ann.addLayer(1, hiddenLayerOne);
			ConnectionFactory.fullConnect(ann.getLayerAt(0), ann.getLayerAt(1));
			ann.addLayer(2, outputLayer);
			ConnectionFactory.fullConnect(ann.getLayerAt(1), ann.getLayerAt(2));
			ConnectionFactory.fullConnect(ann.getLayerAt(0), ann.getLayerAt(ann.getLayersCount() - 1), false);
			ann.setInputNeurons(inputLayer.getNeurons());
			ann.setOutputNeurons(outputLayer.getNeurons());
			ann.randomizeWeights(-1, 1);
	
			BackPropagation bP = new BackPropagation();
			bP.setMaxIterations(maxIterations);
			bP.setLearningRate(0.01);
			bP.addListener(this);
	
			long startTime = System.nanoTime();
			ann.learn(ds, bP);
			long endTime = System.nanoTime();
			System.out.println("Finished in: " + (endTime - startTime) / 1000000000 + "s");
		} catch (Exception e) {
			System.out.println("Error in Training");
			e.printStackTrace();
			output.result += "Error in Training: " + e.getMessage() + "\n";
		}

		try {
			System.out.println("Start testing");
			boolean result[] = new boolean[data_test.X.length];
			double threshold = 0.5;
			int count = 0;
			for (int i = 0; i < validateCount; i++) {
				double x = data_test.X[i][0] / 10.0;
				double y = data_test.X[i][1] / 1000.0;
				double[] tinp = { x, y, x * x, y * y, Math.sin(x), Math.sin(y) };
				ann.setInput(tinp);
				ann.calculate();

				int answer = ann.getOutput()[0] > threshold ? 1 : 0;
				System.out.print("Result: " + data_test.y[i] + " vs " + answer + " (" + ann.getOutput()[0] + ")\n");
				result[i] = answer == data_test.y[i];
				if (result[i])
					count++;
			}
			System.out.println("Finished testing");
			System.out.println("Result: Test with " + result.length + " cases | correct in " + count + " cases | ratio " + ((float) count / result.length));
			
			/*===========*/
			ANNResult aresult = new ANNResult(ann, ((float) count / result.length));
			output.result += aresult.getNeuralNetworkInformation();
			output.point += aresult.getRatio();
			/*===========*/
		} catch (Exception e) {
			System.out.println("Error in get result");
			e.printStackTrace();
			output.result += "Error in get result: " + e.getMessage() + "\n";
		}
//		System.out.println(output.result);
		return output;
	}
	
	private static Neuron chooseNeuron(int i, int c) {
		ThresholdNeuron n;
		switch (i % 6) {
		case 0:
			n = new ThresholdNeuron(new WeightedSum(), new Linear(1.0));
			break;
		case 1:
			n = new ThresholdNeuron(new WeightedSum(), new Ramp());
			break;
		case 2:
			n = new ThresholdNeuron(new WeightedSum(), new Sgn());
			break;
		case 3:
			n = new ThresholdNeuron(new WeightedSum(), new Sigmoid(1.0));
			break;
		case 4:
			n = new ThresholdNeuron(new WeightedSum(), new Step());
			break;
		case 5:
			n = new ThresholdNeuron(new WeightedSum(), new Tanh(1.0));
			break;
		default:
			n = new ThresholdNeuron(new WeightedSum(), new Sigmoid(1.0));
			break;
		}
		n.setThresh(0.3);
		return n;
	}

	public static void readfile(String file_link, List<List<Double>> data, List<List<Double>> test) throws Exception {
		BufferedReader reader = null;

		try {
			if( isHadoop ) {
				reader = new BufferedReader(new InputStreamReader(fs.open(new Path(file_link))));
			} else {
				reader = new BufferedReader(new FileReader(new File(file_link)));				
			}
			String text = null;

			int count = -1;
			while ((text = reader.readLine()) != null && count < totalLine) {
				if (count == -1) {
					name.add(text);
					count++;
				} else {
					String[] tmp = text.split(",");
					data.add(new ArrayList<Double>());
					for (String i : tmp) {
						data.get(count).add(Double.parseDouble(i));
					}
					count++;
				}
			}
			totalLine = data.size();
			testCount = (int)(totalLine * testRatio);
			validateCount = totalLine - testCount;
			
			for (int i = 0; i < validateCount; i++) {
				test.add(data.get(data.size() - 1));
				data.remove(data.size() - 1);
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

	public static Prob changeTypeProblem(List<List<Double>> data) throws Exception {
		Prob prob = new Prob();

		try {
			if (data == null || data.size() == 0) {
				System.out.println("Error data is empty!");
				return new Prob();
			} else if (data.get(0).size() == 0) {
				System.out.println("Error data[0] is empty!");
				return new Prob();
			}

			int leng = data.get(0).size() - 1;
			prob.X = new double[data.size()][leng];
			prob.y = new double[data.size()];

			for (int i = 0; i < data.size(); i++) {
				prob.y[i] = data.get(i).get(0);

				for (int j = 0; j < leng; j++) {
					prob.X[i][j] = data.get(i).get(j + 1);
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

	@Override
	public void handleLearningEvent(LearningEvent event) {
		BackPropagation bp = (BackPropagation) event.getSource();
		if (event.getEventType() != LearningEvent.Type.LEARNING_STOPPED)
			System.out.println(bp.getCurrentIteration() + ". iteration : " + bp.getTotalNetworkError());
	}
	
	public class ANNResult {
		private NeuralNetwork<BackPropagation> ann;
		private double ratio;
		
		public double getRatio() {
			return ratio;
		}

		ANNResult(NeuralNetwork<BackPropagation> ann, double ratio){
			this.ann = ann;
			this.ratio = ratio;
		}
		
		public String getNeuralNetworkInformation() {
			int layerCount = ann.getLayersCount();
			String result = "", output = "";
			for(int i = 0; i < layerCount; i++) {
				result = "";
				Layer layer = ann.getLayerAt(i);
				if(i == 0) {
					System.out.println("Input layer: " + layer.getNeuronsCount() + " neuron(s)");
				} else if(i == layerCount- 1) {
					System.out.println("Output layer: " + layer.getNeuronsCount() + " neuron(s)");
				} else
					System.out.println("Hidden layer " + i + ": " + layer.getNeuronsCount() + " neuron(s)");
				
				for(int j = 0; j < layer.getNeuronsCount(); j++) {
					Neuron neuron = layer.getNeuronAt(j);
					Weight[] weight = neuron.getWeights();
					String ntype = neuron.getTransferFunction().toString();
					if(ntype.contains("Linear")) {
						result += "LINEAR : ";
					} else if(ntype.contains("Ramp")) {
						result += "RAMP   : ";
					} else if(ntype.contains("Sgn")) {
						result += "SGIN   : ";
					} else if(ntype.contains("Sigmoid")) {
						result += "SIGMOI : ";
					} else if(ntype.contains("Step")) {
						result += "STEP   : ";
					} else if(ntype.contains("Tanh")) {
						result += "TANH   : ";
					} else {
						result += "LINEAR : ";
					}
					for(int k = 0; k < weight.length; k++) {
						result += String.format("%10.5f", weight[k].value);
					}
					result += "\n";
				}
				output += result;
			}
			return output;
		}
	}
}

class Prob {
	public String[] name;
	public double[][] X;
	public double[] y;
}

class Result implements Writable {
	public int leng = 0;
	public double point = 0;
	public String result = "";
	
	Result(){
		result = "";
	}
	
	Result(double p, String r) {
		point = p;
		result = r;
		leng = r.length();
	}
	
	public String toString() {
		return point + "|" + result;
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		point = in.readDouble();
		result = in.readUTF();
		leng = result.length();
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeDouble(point);
		out.writeUTF(result);
	}
	
	public static Result read(DataInput in) throws IOException {
		Result r = new Result();
		r.readFields(in);
		return r;
	}
}
