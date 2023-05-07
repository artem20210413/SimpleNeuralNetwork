using Microsoft.VisualStudio.TestTools.UnitTesting;
using SimpleNeuralNetwork;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNeuralNetwork.Tests
{
    [TestClass()]
    public class NeuronNetworksTests
    {
        [TestMethod()]
        public void FeedForwardTest()
        {
            var dataset = new List<Tuple<double, double[]>>
            {
                /*
                 * Результат - Пациент болен - 1
                 *             Пыйиент здоров - 0
                 *             
                 * Неправильная температура Т
                 * Хорошая температура А
                 * Курит S
                 * Правильно питается F
                 * 
                 *                                            T A S F
                 */
                new Tuple<double, double[]> (0, new double[] {0,0,0,0}),
                new Tuple<double, double[]> (0, new double[] {0,0,0,1}),
                new Tuple<double, double[]> (1, new double[] {0,0,1,0}),
                new Tuple<double, double[]> (0, new double[] {0,0,1,1}),
                new Tuple<double, double[]> (0, new double[] {0,1,0,0}),
                new Tuple<double, double[]> (0, new double[] {0,1,0,1}),
                new Tuple<double, double[]> (1, new double[] {0,1,1,0}),
                new Tuple<double, double[]> (0, new double[] {0,1,1,1}),
                new Tuple<double, double[]> (1, new double[] {1,0,0,0}),
                new Tuple<double, double[]> (1, new double[] {1,0,0,1}),
                new Tuple<double, double[]> (1, new double[] {1,0,1,0}),
                new Tuple<double, double[]> (1, new double[] {1,0,1,1}),
                new Tuple<double, double[]> (1, new double[] {1,1,0,0}),
                new Tuple<double, double[]> (0, new double[] {1,1,0,1}),
                new Tuple<double, double[]> (1, new double[] {1,1,1,0}),
                new Tuple<double, double[]> (1, new double[] {1,1,1,1})
            };

            var topology = new Topology(4, 1, 0.1, 2);
            var neuronNetwork = new NeuronNetworks(topology);

            var difference =  neuronNetwork.Learn(dataset, 1000);
            var results = new List<double>();
            foreach(var data in dataset)
            {
                var res =  neuronNetwork.FeedForward(data.Item2).Output;
                results.Add(res);

            }

            for(int i = 0; i < results.Count; i++)
            {
                var expected = Math.Round(dataset[i].Item1, 4);
                var actual = Math.Round(results[i], 4);
                Assert.AreEqual(expected, actual);
            }

            //var result = neuronNetwork.FeedForward(new List<double> { 1,0,0,0});
        }
    }
}