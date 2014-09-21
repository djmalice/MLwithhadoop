package lRegression;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.*;

import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.util.*;

public class LogReg {

	public static class Map extends MapReduceBase implements Mapper<LongWritable, Text, Text, DoubleWritable> {
		private Text word = new Text();
		private double[] readTheta(){
			String thetaFN = "thetaF";
			Path thetaFP = new Path(thetaFN);
			double theta[] = new double[61188 + 1];
			Configuration configuration = new Configuration();
			FileSystem fs;
			try {
				fs = FileSystem.get(configuration);
				FSDataInputStream in = fs.open(thetaFP);
				BufferedReader br = new BufferedReader(new InputStreamReader(in));
				String line = null;
				while((line = br.readLine()) != null){
					//reducer's output is not sorted
					StringTokenizer stokenizer = new StringTokenizer(line);
					int index = -1;
					double thetaVal = 0;
					if(stokenizer.hasMoreTokens()){
						index = Integer.parseInt(stokenizer.nextToken());
					}else{
						System.err.println("Error when reading theta");
						System.exit(2);
					}
					if(stokenizer.hasMoreElements()){
						thetaVal = Double.parseDouble(stokenizer.nextToken());
					}else{
						System.err.println("Error when reading theta");
						System.exit(2);
					}
					theta[index] = thetaVal;
				}
				br.close();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			return theta;
		}
		
		private double g(double []theta, Integer[] X, int y){
			//dimensions of array x and array theta are not the same.
			//x is a sparse vector, with indices representing
			//values that were set.
			double sum = 0.0;
			if(X == null || theta == null){
				System.out.print("Null value for X | theta");
				System.exit(3);
			}
			for(int i=0; i<X.length; i++)
			{
				sum += theta[X[i]];
			}
			sum = -y * sum;
			
			double exp = (1/(1 + Math.exp(sum))) * y;
			return  exp;
		}
		
		public void map(LongWritable key, Text value, OutputCollector<Text, DoubleWritable> output, Reporter reporter) throws IOException {
			String line = value.toString();
			StringTokenizer tokenizer = new StringTokenizer(line);
			String classlbl = "";
			String docid = "";
			Integer X[] = null;
			List<Integer> wordIDs = new ArrayList<Integer>();
			if(tokenizer.hasMoreTokens()){
				docid = tokenizer.nextToken();
			}
			if(tokenizer.hasMoreTokens()){
				classlbl = tokenizer.nextToken();
			}
			if(!classlbl.equalsIgnoreCase("")){
				while(tokenizer.hasMoreTokens()){
					String wordID = tokenizer.nextToken();
					wordIDs.add(new Integer(wordID));
				}
				if(wordIDs.size() > 0){
					X = new Integer[wordIDs.size()];
					for(int n=0; n<wordIDs.size(); n++){
						X[n] = new Integer(wordIDs.get(n));
					}
				}
				double theta[] = readTheta();
				double dotProduct = g(theta, X, Integer.parseInt(classlbl));
				System.out.print(dotProduct);
				for(int m=0; m<X.length; m++){
					word.set(String.valueOf(X[m]));
					output.collect(word, new DoubleWritable(dotProduct));
				}
			}
			else{
				System.err.println("Format of input is not correct");
				System.exit(5);
			}
		}
	}

	public static class Reduce extends MapReduceBase implements Reducer<Text, DoubleWritable,Text,DoubleWritable> {
		public void reduce(Text key, Iterator<DoubleWritable> values, OutputCollector<Text, DoubleWritable> output, Reporter reporter) throws IOException {
			double sum = 0;
			while(values.hasNext()){
				sum += values.next().get();
			}
			output.collect(key, new DoubleWritable(sum));
		}
	}

	public static void main(String[] args) throws Exception {
		
		final int lenVocab = 61188;
		final int nIters  = 2;
		final double alpha = 0.1f; //training rate.
	
		Random rGen = new Random();
		
		//Initializing the filesystem
		Configuration configuration = new Configuration();
		FileSystem fs = FileSystem.get(configuration);
		Path outputF = new Path(args[1]);
		
		//assign initial vlaue of theta
		double[] theta = new double[lenVocab + 1];
		for(int i=0; i<lenVocab+1; i++){
			theta[i] = rGen.nextFloat();
		}
		
		//assign a vector for the gradient
		double[] gradient = new double[lenVocab + 1];
		
		//iterations of map reduce
		for(int i=0; i<nIters; i++){
			//delete output folder, if present
			if(fs.exists(outputF)){
				fs.delete(outputF, true);
			}
			
			JobConf conf = new JobConf(LogReg.class);
			conf.setJobName("logreg"+i);
			
			conf.setOutputKeyClass(Text.class);
			conf.setOutputValueClass(DoubleWritable.class);

			conf.setMapOutputKeyClass(Text.class);
			conf.setMapOutputValueClass(DoubleWritable.class);

			conf.setMapperClass(Map.class);
			conf.setReducerClass(Reduce.class);

			conf.setInputFormat(TextInputFormat.class);
			conf.setOutputFormat(TextOutputFormat.class);

			FileInputFormat.setInputPaths(conf, new Path(args[0]));
			FileOutputFormat.setOutputPath(conf, new Path(args[1]));
			
			//map-reduce job - gradient descent
			JobClient.runJob(conf);
			
			//read gradient
			Path gradientFile = new Path(args[1] + "/part-00000");
			if(!fs.exists(gradientFile)){
				System.err.println("File "+args[1]+ "/part-0000 not found");
				System.exit(1);
			}
			
			FSDataInputStream in = fs.open(gradientFile);
			BufferedReader br = new BufferedReader(new InputStreamReader(in));
			String line = null;
			while((line = br.readLine()) != null){
				//reducer's output is not sorted
				StringTokenizer stokenizer = new StringTokenizer(line);
				int index = -1;
				double gradVal = 0;
				if(stokenizer.hasMoreTokens()){
					index = Integer.parseInt(stokenizer.nextToken());
				}else{
					System.err.println("Error when reading the gradient");
					System.exit(2);
				}
				if(stokenizer.hasMoreElements()){
					gradVal = Double.parseDouble(stokenizer.nextToken());
				}else{
					System.err.println("Error when reading the gradient");
					System.exit(2);
				}
				gradient[index] = gradVal;
			}
			br.close();
			//update theta - learning rate = alpha
			for(int j=0; j<lenVocab + 1; j++){
				theta[j] = theta[j] - alpha * gradient[j];
			}
			//write theta
			//delete theta file and write it
			
			Path thetaF = new Path("thetaF");
			if(fs.exists(thetaF)){
				fs.delete(thetaF, true);
			}
			FSDataOutputStream outputStream = fs.create(thetaF, true);
			BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(outputStream));
			for(int m=1; m<lenVocab + 1; m++){
				bw.write(String.valueOf(m));
				bw.write(("\t"));
				bw.write(String.valueOf(theta[m]));
				bw.write("\n");
			}
			bw.close();
		}
	}
}