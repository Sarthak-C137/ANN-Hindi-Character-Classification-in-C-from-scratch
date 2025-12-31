// Your First C++ Program

#include <iostream>
#include <cmath>
#include <iostream>
#include <fstream>
#define lrn_rate 0.007
#define mmntm_coff 0.75
#define leak_constant 0.001
#define orj 1200 // epoch counter for the 
using namespace std;

//**********data structure*************************** */
int size;

double bacth_error[10000]; // evry epoch error
double se;                 // store instance error
double data[3609][227] ={};
// bias and wts

double ol_grdnt[10][15];   // stores calculated gradient of output layer
double hl2_grdnt[14][15];  // calculated gradient of hidden 2 layer
double hl1_grdnt[14][226]; // stores calculated gradient of hidden 1 layer
// momentum base learning velocity component
double ol_vel[10][15];  // stores calculated gradient of output layer
double hl2_vel[14][15]; // calculated gradient of hidden 2 layer
double hl1_vel[14][226];
// ******************function module************************************************************************************


void store_weights(double hl1_wts[14][226], double hl2_wts[14][15], double ol_wts[10][15], const char* filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Error opening file for writing.\n";
        return;
    }
// store calculated gradient in the file
    out.write(reinterpret_cast<char*>(hl1_wts), sizeof(double) * 14 * 226);
    out.write(reinterpret_cast<char*>(hl2_wts), sizeof(double) * 14 * 15);
    out.write(reinterpret_cast<char*>(ol_wts), sizeof(double) * 10 * 15);
    
    out.close();
}


double leak_relu(double x)
{
    if (x>=0)
    {return x;}
    else
    {
        return (0.001*x);
    }
}
double diff_l_relu (double x)

{
    if (x>=0)
    {return 1.0;}
    else
    {
        return 0.001;
    }
}
double sigmoid(double x)
{
  if (x >= 0.0)
  {
    double y = 1.0 / (1.0 + exp(-x));
    return y;
  }
  else
  {
    double ex = exp(x);
    return ex / (1.0 + ex);
  }
}
void softmax()
{
  double sum = 0;
  for (int i = 0; i < 10; ++i)
  {
    sum += output_layer[i];
  }
  for (int i = 0; i < 10; ++i)
  {
    output_layer[i] = output_layer[i] / sum;
  }
}
void forward()
{
  for (int i = 0; i < 14; i++)
  {
    hl1_n[i + 1] = 0;
    for (int j = 1; j < 227; j++)
    {
      hl1_n[i + 1] += data[size][j] * hl1_wts[i][j];
    }

    hl1_n[i + 1] = leak_relu(hl1_n[i + 1]); // Sigmoid fun
  }

  for (int i = 0; i < 14; i++)
  {
    hl2_n[i + 1] = 0;
    for (int j = 0; j < 15; j++)
    {
      // int r = m - 1;
      hl2_n[i + 1] += hl1_n[j] * hl2_wts[i][j];
    }

    hl2_n[i + 1] = leak_relu(hl2_n[i + 1]); // using relu leaky as non linear fiunction
  }

  for (int i = 0; i < 10; i++)
  {
    output_layer[i] = 0;
    for (int j = 0; j < 15; j++)
    {
      // int s = q - 1;
      output_layer[i] += hl2_n[j] * ol_wts[i][j];
    }

    output_layer[i] = exp(output_layer[i]); //// soft max compatible
  }
  // calculate softmax for last layer
  softmax();
}

void Log_error()
{

  double true_op[10] = {0};
  int true_indx = data[size][0] - 1;
  true_op[true_indx] = 1; // setting intended output to be 1
  se = 0;
  for (int i = 0; i < 10; ++i)
  {
    se += (0 - true_op[i]) * log(output_layer[i]);
  }
  se = se / 10;
}

void grdnt_lyr_2()
{
  double grdnt_temp_sum[14] = {0};
  for (int j = 1; j < 15; ++j) // colum wise multiplication and sum  with the n+1 gradient and n+1 respective weights excluding the bias so j start from the 1
  {
    for (int i = 0; i < 10; ++i)
    {
      grdnt_temp_sum[j - 1] += (ol_grdnt[i][j]/hl2_n[j] )* ol_wts[i][j]; // save column sum of point to point multiplication
    }
  }
  for (int i = 0; i < 14; ++i) // colum wise multiplication and sum  with the n+1 gradient and n+1 respective weights excluding the bias so j start from the 1
  { double temp = grdnt_temp_sum[i] * diff_l_relu(hl2_n[i+1]);
    for (int j = 0; j < 15; ++j)
    {
      // for compansaiting with the hl2 there
      hl2_grdnt[i][j] = temp * (hl1_n[j]);                  //  hl2 i+1 for removing the bias input
      // cout << hl2_grdnt[i][j]<<" ";
    }
    // cout<<endl;
  }
}

void grdnt_lyr_1()
{
  double grdnt_temp_sum[14] = {0};
  for (int j = 1; j < 15; ++j) // colum wise multiplication and sum  with the n+1 gradient and n+1 respective weights excluding the bias so j start from the 1
  {
    for (int i = 0; i < 14; ++i)
    {
      grdnt_temp_sum[j - 1] += (hl2_grdnt[i][j]/(hl1_n[j])) * hl2_wts[i][j]; // save column sum of point to point multiplication
    }
  }
  for (int i = 0; i < 14; ++i) // colum wise multiplication and sum  with the n+1 gradient and n+1 respective weights excluding the bias so j start from the 1
  {
    double temp = grdnt_temp_sum[i] * diff_l_relu(hl1_n[i+1]);
    for (int j = 0; j < 226; ++j)
    {
      hl1_grdnt[i][j] = temp * (data[size][j + 1]); //  hl2 i+1 for removing the bias input j+1 for removing data name
                                                    //   cout << hl1_grdnt[i][j]<<" ";
    }
    // cout<<endl;
  }
}

void back_prpgtn()
{
  // calculating gradient in order output----> hidden layer 2------> hiiden 1 wts gradient

  double true_op[10] = {0};
  int true_indx = data[size][0] - 1;
  true_op[true_indx] = 1; 
  for (int i = 0; i < 10; ++i) // less than size of output
  {
    for (int j = 0; j < 15; ++j) // less yhan size of hidden 2 neuranss matrix
    {
      ol_grdnt[i][j] = (output_layer[i] - true_op[i]) * hl2_n[j]; // soft max and log loss after calculation
      
    }
    
  }
  grdnt_lyr_2();
  
  grdnt_lyr_1();
  

  return;
}

void learn_wts()
{
  // with momentum
  for (int i = 0; i < 10; ++i)
  {
    for (int j = 0; j < 15; ++j)

    {
      ol_vel[i][j] = (mmntm_coff * ol_vel[i][j]) + ((1 - mmntm_coff) * ol_grdnt[i][j]);
       ol_wts[i][j] = (ol_wts[i][j]) - (lrn_rate * ol_vel[i][j]);
      
    }
  }
  for (int i = 0; i < 14; ++i)
  {
    for (int j = 0; j < 15; ++j)
    { 
      hl2_vel[i][j] = (mmntm_coff * hl2_vel[i][j]) + ((1 - mmntm_coff) * hl2_grdnt[i][j]);
      hl2_wts[i][j] = (hl2_wts[i][j]) - (lrn_rate * hl2_vel[i][j]);
  
    }
   
  }
 
  for (int i = 0; i < 14; ++i)
  {
    for (int j = 0; j < 226; ++j)
    {
    
      hl1_vel[i][j] = (mmntm_coff * hl1_vel[i][j]) + ((1 - mmntm_coff) * hl1_grdnt[i][j]);
       hl1_wts[i][j] = (hl1_wts[i][j]) - (lrn_rate * hl1_vel[i][j]);
      
    }
  }
}

  //*****************************driver function********************************************************************************************
  int main()
  {
    printf("i did this . i am so proud of me\n");
    
    for (int epoch = 0; epoch <orj; ++epoch)
    {
      for (size = 0; size <3609; ++size)
      {

        forward();

        Log_error();
        bacth_error[epoch] += se; // summission of sq of error

        back_prpgtn();
        learn_wts();
      }

      bacth_error[epoch] = bacth_error[epoch] / 3609; // mean of error

      {
        cout << " Logerror for batch :" << epoch<< " "<<bacth_error[epoch] << endl;
      }
    }

    cout<<"storing weights"<<endl;
    store_weights(hl1_wts, hl2_wts, ol_wts, "weights.dat");
   return 0;
  
}
