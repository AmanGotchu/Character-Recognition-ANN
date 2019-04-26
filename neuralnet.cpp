#include <iostream>
#include <unordered_map>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <random>
using namespace std;


int colSize = 9;
int rowSize = 14;

string getBinary(char c){
    switch(c){
        case '0': 
            return "0000"; 
        case '1': 
            return "0001"; 
        case '2': 
            return "0010"; 
        case '3': 
            return "0011"; 
        case '4': 
            return "0100"; 
        case '5': 
            return "0101"; 
        case '6': 
            return "0110"; 
        case '7': 
            return "0111"; 
        case '8': 
            return "1000"; 
        case '9': 
            return "1001"; 
        case 'A': 
            return "1010"; 
        case 'B': 
            return "1011"; 
        case 'C': 
            return "1100"; 
        case 'D': 
            return "1101"; 
        case 'E': 
            return "1110"; 
        case 'F': 
            return "1111"; 
        default:
            return "";
    }
}

string hexToBinary(string hex){
    string binaryStr = "";

    char first = hex[0];
    char second = hex[1];

    return getBinary(first) + getBinary(second);
}

double logisticSigmoid(double input){
    double eVal = exp(-1 * input);
    return 1.0 / (1.0 + eVal);
}

double hyperbolicTangentSigmoid(double input){
    return tanh(input);
}

double randomVal(double minVal, double maxVal){
    double f = (double)rand() / RAND_MAX;
    double ans = minVal + f * (maxVal - minVal);

    return ans;
}

class network{
    public:
        int inputLayerSize = 127;
        int layerTwoSize = 127;
        int layerThreeSize = 26;

        vector< vector<double> > networkNodes;
        vector< vector< vector<double> > > weights;

        vector<double> inputLayerVals;
        vector<double> layerTwoVals;
        vector<double> layerThreeVals;

        void resetNodeVals(){
            networkNodes = vector< vector<double> >(3); // 3 Layers

            vector<double> layerOne(127, 0);
            layerOne[126] = 1;
            networkNodes[0] = layerOne;

            vector<double> layerTwo(127, 0);
            layerTwo[126] = 1;
            networkNodes[1] = layerTwo;

            vector<double> outputLayer(26, 0);
            networkNodes[2] = outputLayer;
        }

        static vector< vector<double> > errorVector(){
            vector< vector<double> > deltas;

            vector<double> layerOne(127, 0);
            deltas.push_back(layerOne);

            vector<double> layerTwo(127, 0);
            deltas.push_back(layerTwo);

            vector<double> outputLayer(26, 0);
            deltas.push_back(outputLayer);

            return deltas;
        }

        network(){
            // Initialize networkNodes
            resetNodeVals();

            double minInitWeight = -.1;
            double maxInitWeight = .1;

            vector< vector<double> > inputLayerWeights; // 127 x 127
            vector< vector<double> > layerTwoWeights; // 127 x 26

            // Initialize inputLayerWeights
            for(int i = 0; i<inputLayerSize; i++){
                vector<double> temp;
                for(int j = 0; j<layerTwoSize; j++){

                    // Get random value between -.1 and .1
                    double rVal = randomVal(minInitWeight, maxInitWeight);
                    while(rVal == 0){
                        rVal = randomVal(minInitWeight, maxInitWeight);
                    }
                    temp.push_back(rVal);
                }
                inputLayerWeights.push_back(temp);
            }

            // Initialize layerTwoWeights
            for(int i = 0; i<layerTwoSize; i++){
                vector<double> temp;
                for(int j = 0; j<layerThreeSize; j++){

                    // Get random value between -.1 and .1
                    double rVal = randomVal(minInitWeight, maxInitWeight);
                    while(rVal == 0){
                        rVal = randomVal(minInitWeight, maxInitWeight);
                    }
                    temp.push_back(rVal);
                }
                layerTwoWeights.push_back(temp);
            }

            weights.push_back(inputLayerWeights);
            weights.push_back(layerTwoWeights);
        }
};

void printWeights(vector< vector< vector<double> > > &weights, string fileName){
    ofstream weightFile;
    weightFile.open(fileName);

    weightFile << weights.size() << endl;

    for(int layer = 0; layer < weights.size(); layer++){
        weightFile << weights[layer].size() << " " <<weights[layer][0].size() << endl;
       for(int node = 0; node < weights[layer].size(); node++){
           for(int nextNode = 0; nextNode < weights[layer][node].size(); nextNode++){
               weightFile << node << " " << nextNode << " " << weights[layer][node][nextNode] << endl;
           }
       }
    }
}

void readWeights(network &nn, string readFile){
    ifstream weightFile;
    weightFile.open(readFile);

    int numLayers;
    weightFile >> numLayers;

    vector< vector< vector<double> > > weights;
    
    for(int i = 0; i<numLayers; i++){
        int layerOneSize, layerTwoSize;
        weightFile >> layerOneSize >> layerTwoSize;
        vector< vector< double> > layerWeights(layerOneSize, vector<double>(layerTwoSize));

        for(int iter = 0; iter < layerOneSize * layerTwoSize; iter++){
            int prev, next;
            double weight;
            weightFile >> prev >> next >> weight;
            layerWeights.at(prev).at(next) = weight;
        }
        
        weights.push_back(layerWeights);
    }

    nn.weights = weights;
}

network backPropogation(vector< vector<double> > charToInputMap, network &nn){
    ofstream tFile;
    tFile.open("trainingfile.txt");

    vector< vector<double> > errorVector = nn.errorVector();
    int numLayers = nn.networkNodes.size();

    double a = 0.1;
    double epoch = 0;

    nn.resetNodeVals();
    while(true){ //Figure out stopping criterion
        epoch++;
        double learningRate = a;
        int numCorrect = 0;

        for(int c = 0; c<charToInputMap.size(); c++){
            vector<double> binaryVector = charToInputMap[c]; //input
            
            char character = 'A' + c;
            vector<double> outputVector(26, 0); //output
            outputVector[c] = 1;

            /*                FORWARD PROPOGATION                 */
            for(int i = 0; i<binaryVector.size(); i++){ // Initialize input layer with binary vector
                nn.networkNodes[0][i] = binaryVector[i];
            }

            // layer == numLayers - 1 ? logistic sigmoid function : hyperbolic tangent function
            for(int layer = 1; layer < numLayers; layer++){
                for(int currLayerNode = 0; currLayerNode < nn.networkNodes[layer].size(); currLayerNode++){
                    double inVal = 0;
                    for(int prevLayerNode = 0; prevLayerNode < nn.networkNodes[layer-1].size(); prevLayerNode++){
                        inVal += nn.weights[layer-1][prevLayerNode][currLayerNode] * nn.networkNodes[layer-1][prevLayerNode];
                    }                   

                    double activationVal, logisticVal, hyperbolicVal;
                    logisticVal = logisticSigmoid(inVal);
                    hyperbolicVal = hyperbolicTangentSigmoid(inVal);

                    if(layer == numLayers-1){
                        activationVal = logisticVal;
                    }else{
                        activationVal = hyperbolicVal;
                    }

                    nn.networkNodes[layer][currLayerNode] = activationVal;
                }
            }

            /*                BACKWARD PROPOGATION                 */
            for(int outputNode = 0; outputNode < nn.networkNodes[numLayers-1].size(); outputNode++){
                // logistic sigmoid function derivative is g(x) * (1 - g(x))
                double derivVal = nn.networkNodes[numLayers-1][outputNode] * (1 - nn.networkNodes[numLayers-1][outputNode]);
                double diffVal = outputVector[outputNode] - nn.networkNodes[numLayers-1][outputNode];
                errorVector[numLayers-1][outputNode] = derivVal * diffVal;
            }

            for(int layer = numLayers-2; layer >= 1; layer--){
                for(int node = 0; node < nn.networkNodes[layer].size(); node++){
                    // hyperbolic tangent sigmoid function derivative is 1 - ( g(x) ^ 2)
                    double derivVal = 1 - (nn.networkNodes[layer][node] * nn.networkNodes[layer][node]);

                    double summationVal = 0;
                    for(int nextLayerNode = 0; nextLayerNode < nn.weights[layer][node].size(); nextLayerNode++){
                        summationVal += nn.weights[layer][node][nextLayerNode] * errorVector[layer+1][nextLayerNode];
                    }

                    errorVector[layer][node] = summationVal * derivVal;
                }
            }

            double maxOutput = 0;
            char testOut = '0';
            
            for(int outputNode = 0; outputNode < nn.networkNodes[numLayers-1].size(); outputNode++){
                if(nn.networkNodes[numLayers-1][outputNode] >= maxOutput){
                    maxOutput = nn.networkNodes[numLayers-1][outputNode];
                    testOut = 'A' + outputNode;
                }
            }

            for(int layer = 0; layer < numLayers-1; layer++){
                for(int currLayerNode = 0; currLayerNode < nn.weights[layer].size(); currLayerNode++){
                    for(int nextLayerNode = 0; nextLayerNode < nn.weights[layer][currLayerNode].size(); nextLayerNode++){
                        double currWeight = nn.weights[layer][currLayerNode][nextLayerNode];
                        double newWeight = currWeight + (learningRate * nn.networkNodes[layer][currLayerNode] * errorVector[layer+1][nextLayerNode]);

                        nn.weights[layer][currLayerNode][nextLayerNode] = newWeight;
                    }
                }
            }

            if(testOut == character && maxOutput >= .95){
                numCorrect++;
            }
            
            tFile << character << " : " << testOut << " : " << maxOutput << endl;
        }

        double avgCorrect= numCorrect / 26.0;
        tFile << "Current Epoch: " << epoch << endl;
        tFile << "Average Correct: " << avgCorrect << endl;
        tFile << "<-------------------------------->" << endl << endl;

        cout << "Current Epoch: " << epoch << endl;
        cout << "Average Correct: " << avgCorrect << endl;
        cout << "<-------------------------------->" << endl << endl;

        nn.resetNodeVals();
        if(avgCorrect >= .98){
            break;
        }
    }

    return nn;
}

char evaluate(vector<double> binaryInput, network &nn){
    nn.resetNodeVals();
    int numLayers = nn.networkNodes.size();

    /* Prep Input Layer */
    for(int i = 0; i<binaryInput.size(); i++){
        nn.networkNodes[0].at(i) = binaryInput[i];
    }

    for(int layer = 1; layer < numLayers; layer++){
        for(int currLayerNode = 0; currLayerNode < nn.networkNodes[layer].size(); currLayerNode++){
            double inVal = 0;
            for(int prevLayerNode = 0; prevLayerNode < nn.networkNodes[layer-1].size(); prevLayerNode++){
                inVal += nn.weights[layer-1][prevLayerNode][currLayerNode] * nn.networkNodes[layer-1][prevLayerNode];
            }                   

            double activationVal;
            if(layer == numLayers-1){ // Output Layer Activation Function
                activationVal = logisticSigmoid(inVal);
            }else{
                activationVal = hyperbolicTangentSigmoid(inVal);
            }

            nn.networkNodes[layer][currLayerNode] = activationVal;
        }
    }

    double maxOutputVal = 0;
    char actual = '0';

    // Evaluating output layer
    for(int outputNode = 0; outputNode<nn.networkNodes[numLayers-1].size(); outputNode++){
        if(nn.networkNodes[numLayers-1][outputNode] > maxOutputVal){
            maxOutputVal = nn.networkNodes[numLayers-1][outputNode];
            actual = 'A' + outputNode;
        }
    }

    return actual;
}

void evaluateNoise(network &neuralnet, vector< vector<int> > dottedIndexes, vector< vector<double> > charToInputMap, string outFile){
    ofstream noiseOut, rawNoiseOut;
    noiseOut.open(outFile);

    for(int i = 0; i<26; i++){
        char expected = 'A' + i;
        int flippedBits = 0;

        vector<int> currIndexes = dottedIndexes[i];
        int totalFilledBits = currIndexes.size();
        vector<double> binaryVector = charToInputMap[i];

        char actualVal = evaluate(binaryVector, neuralnet);

        while(actualVal == expected && currIndexes.size() > 0){
            int removeVal = rand() % currIndexes.size();
            binaryVector[currIndexes[removeVal]] = 0;
            currIndexes.erase(currIndexes.begin() + removeVal);

            actualVal = evaluate(binaryVector, neuralnet);
            flippedBits++;
        }

        if(flippedBits == totalFilledBits){
            noiseOut << expected << " was evaluated correctly for all " << flippedBits << "/" << flippedBits << " flipped bits" << endl;
        }else{
            noiseOut << expected << " failed with " << flippedBits << " flipped bits. Evaluated to: " << actualVal << endl;
        }
    }
}

bool parseBDF(vector< vector<double> > &charToInputMap, vector< vector<int> > &dottedIndexes){
    //Open the BDF file
    ifstream bdfFile;
    bdfFile.open("im9x14u.bdf");

    for(int i = 0; i<26; i++){
        char character = 'A' + i;
        vector<double> charInput(126, 0.0);
        vector<int> charIndexes;

        //Receive 1-d input buffer based on the BDF file format
        int bbx, bby, bbxOff, bbyOff;
        bdfFile >> bbx >> bby >> bbxOff >> bbyOff;

        for(int j = 0; j<bby; j++){
            int currRow = j;
            string hex; bdfFile >> hex;
            string binary = hexToBinary(hex);

            for(int k = 0; k<binary.length(); k++){
                if(binary[k] == '1'){
                    int index = j * colSize + k;
                    charIndexes.push_back(index);
                    charInput[index] = 1.0;
                }
            }
        }

        string fin; bdfFile >> fin;
        if(fin != "ENDCHAR"){ //Make sure parsing is correct
            cout << "ERROR" << endl;
            return false;
        }
        charToInputMap[i] = charInput;
        dottedIndexes.push_back(charIndexes);
    }

    return true;
}

/*
    Available Functions:
        parseBDF() - parses a bdf file
        readWeights() - read weights from a text file and inputs it to a neural network
        printWeights() - prints out the weights of a neural network to a given text file
        backPropogation() - trains a neural net 
        (not really back propagation but also includes forward propagation, named only because we were meant to follow pseudo-code in book)

        evaluate() - evaluates a given binary input vector with a given neural network
        checkNoise() - checks how many "noise iterations" or flippedBits it takes until expected value != returned value
*/

int main(){
    srand(NULL);

    /*                      REQUIRED                        */
    vector< vector<double> > charToInputMap(26, vector<double>(126, 0.0)); // Map capital letters to 126 size one-dimensional input buffer
    vector< vector<int> > dottedIndexes; // Maps all the dotted indexes for a character
    bool parsed = parseBDF(charToInputMap, dottedIndexes); // Parse the BDF File
    if(!parsed){
        cout << "Error parsing" << endl;
        return 0;
    }
    network neuralnet; //Untrained neural net with random weights between -.1 and .1
    

    /*                      TEST IT YOURSELF!               */

    // Trains an untrained neural net given BDF Format data
    network adjustedNet = backPropogation(charToInputMap, neuralnet);
    
    //Uses inputted weights from a text file
    readWeights(neuralnet, "adjustedweights.txt"); 

    // Print weights of a neural net to a text file after training
    printWeights(adjustedNet.weights, "adjustedweights.txt");

    // Evaluate how many noise is introduced to get a wrong value
    evaluateNoise(neuralnet, dottedIndexes, charToInputMap, "noisedata.txt");

    return 0;
}