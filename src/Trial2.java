import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

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

public class Trial2 implements LearningEventListener {
	private static List<List<Double>> data = new ArrayList<List<Double>>();
	private static List<List<Double>> test = new ArrayList<List<Double>>();
	private static List<String> name = new ArrayList<String>();

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

	public static void main(String[] args) {
		new Trial2().run();
	}

	public Result run() {

		int neuronSet[] = new int[] { 1, 1, 1, 4, 1, 2 };

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

		Prob data_train = null;
		int totalLine = 950;
		double testRatio = 0.9;
		int testCount = (int)(totalLine * testRatio);
		int validateCount = totalLine - testCount;
		try {
			System.out.println("Read data");
			readfile("train.csv", testCount, validateCount);
			System.out.println("Data size: " + data.size() + " * " + data.get(0).size());

			System.out.println("Convert Data");
			data_train = changeTypeProblem(data);
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
		}
		System.out.println("Start training");

		NeuralNetwork<BackPropagation> ann = new NeuralNetwork<BackPropagation>();
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
		bP.setMaxIterations(20);
		bP.setLearningRate(0.01);
		bP.addListener(this);

		long startTime = System.nanoTime();
		ann.learn(ds, bP);
		long endTime = System.nanoTime();
		System.out.println("Finished in: " + (endTime - startTime) / 1000000000 + "s");
		Result output = new Result();

		try {
			System.out.println("Start testing");
			Prob data_test = changeTypeProblem(test);
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
			output.result = aresult.getNeuralNetworkInformation();
			output.point = aresult.getRatio();
			/*===========*/
		} catch (Exception e) {
			System.out.println("Error in get result");
			e.printStackTrace();
		}
		return output;
	}

	public static void readfile(String file_link, int train_leng, int test_leng) throws Exception {
		int limit = train_leng + test_leng;
		File file = new File(file_link);
		BufferedReader reader = null;

		try {
			reader = new BufferedReader(new FileReader(file));
			String text = null;

			int count = -1;
			while ((text = reader.readLine()) != null && count < limit) {
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

			for (int i = 0; i < test_leng; i++) {
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

	public static class Prob {
		public String[] name;
		public double[][] X;
		public double[] y;
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
			String result = "";
			for(int i = 0; i < layerCount; i++) {
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
//						System.out.print("LINEAR:\t\t");
						result += "LINEAR :\t";
					} else if(ntype.contains("Ramp")) {
//						System.out.print("RAMP:\t\t");
						result += "RAMP   :\t";
					} else if(ntype.contains("Sgn")) {
//						System.out.print("SGIN:\t\t");
						result += "SGIN   :\t";
					} else if(ntype.contains("Sigmoid")) {
//						System.out.print("SIGMOID:\t");
						result += "SIGMOI :\t";
					} else if(ntype.contains("Step")) {
//						System.out.print("STEP:\t\t");
						result += "STEP   :\t";
					} else if(ntype.contains("Tanh")) {
//						System.out.print("TANH:\t\t");
						result += "TANH   :\t";
					} else {
//						System.out.print("");
						result += "LINEAR :\t";
					}
					for(int k = 0; k < weight.length; k++) {
//						System.out.format("%10.5f", weight[k].value);
						result += String.format("%10.5f", weight[k].value);
					}
//					System.out.println();
					result += "\n";
				}
				System.out.println(result);
			}
			System.out.println();
			System.out.format("Ratio: %.2f/1.0", ratio);
			return result;
		}
	}
}

class Result {
	public double point = 0;
	public String result = "";
}
