using System;
using System.Collections.Generic;
using System.Data.SqlTypes;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNeuralNetwork
{
    public class Neuron
    {
        public List<double> Weights { get; }

        public List<double> Inputs { get; }

        public NeuronType NeuronType { get; }

        public double Output { get; private set; }
        public double Delta { get; private set; }


        
        public Neuron(int inputCount, NeuronType type = NeuronType.Normal)
        {

            NeuronType = type;
            Weights = new List<double>();
            Inputs = new List<double>();

            InitWeigthsRandomValue(inputCount);
        }

        private void InitWeigthsRandomValue(int inputCount)
        {
            var rnd = new Random();

            for (int i = 0; i < inputCount; i++)
            {
                if(NeuronType == NeuronType.Input)
                { 
                    Weights.Add(1);
                }
                else { 
                
                    Weights.Add(rnd.NextDouble());
                }
                Inputs.Add(0);
            }
        }

        public double FeedForward(List<double> inputs)
        {
            for(int i = 0; i < inputs.Count; i++)
            {
                Inputs[i] = inputs[i];
            }

            var sum = 0.0;
            for (int i = 0; i < inputs.Count; i++)
            {
                sum += inputs[i] * Weights[i];
            }

            if (NeuronType != NeuronType.Input)
            { 
                Output = Sigmoid(sum);
            }
            else
            {
                Output = sum;
            }

            return Output;

        }

        private double Sigmoid(double x)
        {
            var result = 1.0 / (1.0 + Math.Pow(Math.E, -x));
            return result;
        }
        private double SigmoidDx( double x)
        {
            var sigmoid = Sigmoid(x);
            var result =  sigmoid * (1 - sigmoid);
            return result;
        }

        public void Learn(double error, double learningRate)
        {
            if(NeuronType == NeuronType.Input)
            {
                return;
            }

            Delta = error * SigmoidDx(Output);
            for(int i = 0; i < Weights.Count; i++)
            {
                var weiht = Weights[i];
                var Input = Inputs[i];

                var newWeight = weiht - Input * Delta * learningRate;
                Weights[i] = newWeight;
            }

        }

        public override string ToString()
        {
            return Output.ToString();
        }
        public void SetDelta(double delta)
        {
            Delta = delta;
        }
        public void SetWeights(List<double> weights)
        {
            if (weights.Count != Weights.Count)
            {
                throw new ArgumentException("Неверное количество весов для обновления нейрона.");
            }

            for (int i = 0; i < weights.Count; i++)
            {
                Weights[i] = weights[i];
            }
        }

      

    }
}
