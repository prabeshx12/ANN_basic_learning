A repo where I do my learning stuffs here

<h2>Day1: Artificial Neural Network</h2>

<h3> 1.1 Adaline Learning Unipolar and bipolar case</h3>

-> These help in solving only the linearly separable
    problems such as OR, AND, NAND, NOR gates

-> For non-linearly separable problems such as XOR and
    XNOR they can't predict well.

-> The activator function here will be: if value>0.5 return 1 else return 0


<i>bias(new) = bias(old) + learning_rate * error

weights(new) = weight(old) + learning_rate * inputs

error = t - predicted_value

predicted_value = &Sigma;(weights * inputs) + bias</i>


<h3> 1.2 Backpropagation</h3>

-> This will solve the non-linearity problem by introducing
non-linearity through the hidden neurons.

-> Say for the two input XOR gate it will use two hidden neurons
to create non-linearity which will aid in solving the problem.

-> This learning uses the activator function such as Sigmoid function. i.e.
<i>1/(1+e<sup>-x</sup></i>)

<h4> Forward Pass </h4>
<i>hidden_layer_input = &Sigma;(inputs * weights) + bias_input_hidden<br>
   hidden_layer_output = Sigmoid(hidden_layer_input)<br>
   final_layer_input = &Sigma;(hidden_layer_output * weights) + bias_hidden_output<br>
   final_layer_output = Sigmoid(final_layer_input)<br></i>

<h4>Back propagation</h4>
<i>&delta;<sub>k</sub> = (t - y<sub>k</sub>) * Sigmoid_derivative(y<sub>ink</sub>)<br>
   &delta;<sub>j</sub> = &delta;y<sub>inj</sub> * Sigmoid_derivative(Z<sub>inj</sub>)<br>
   &delta;y<sub>inj</sub> = &Sigma;(&delta;<sub>k</sub>* w<sub>jk</sub>)
   where &delta;<sub>k</sub> = delta output error.
</i>

<h4>Updating Weights</h4>
<i>&Delta;w<sub>ij</sub> = &alpha; * &delta;<sub>j</sub> * x<sub>i</sub><br>
<i>&Delta;w<sub>jk</sub> = &alpha; * &delta;<sub>k</sub> * z<sub>j</sub><br>
<i>w<sub>new</sub> = w<sub>old</sub> + &Delta;w<br></i>






