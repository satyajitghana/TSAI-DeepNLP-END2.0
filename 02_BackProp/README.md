# 02 Back Propagation

## Assignment

1.  Rewrite the whole excel sheet showing backpropagation. Explain each major step, and write it on Github.  
    1.  **Use exactly the same values for all variables as used in the class**
    2.  Take a screenshot, and show that screenshot in the readme file
    3.  Excel file must be there for us to cross-check the image shown on readme (no image = no score)
    4.  Explain each major step
    5.  Show what happens to the error graph when you change the learning rate from [0.1, 0.2, 0.5, 0.8, 1.0, 2.0]
2.  Submit the GitHub link. Github link must be public (check the link without logging in, if it opens, then only share the link as an assignment).
    1.  Sanity, please.

![enter image description here](https://github.com/extensive-nlp/TSAI-DeepNLP-END2.0/blob/main/02_BackProp/network.png?raw=true)


## Solution

Link to Sheets: [Click Here](https://docs.google.com/spreadsheets/d/e/2PACX-1vQt6ZBcXJcHRZbYVWIIXLG6qheyKbx8ac7vDG8geqoeCQdOTJI3kZPe6pyAVdxCYb0paFkR8jknNKGb/pubhtml)

See the Sheets named after each LR rate assigned, along with a LR Comparison sheet.

Or see the below embed

<iframe src="https://docs.google.com/spreadsheets/d/e/2PACX-1vQt6ZBcXJcHRZbYVWIIXLG6qheyKbx8ac7vDG8geqoeCQdOTJI3kZPe6pyAVdxCYb0paFkR8jknNKGb/pubhtml?widget=true&amp;headers=false" width="1000" height="1000"></iframe>

![LR Comparison](https://github.com/extensive-nlp/TSAI-DeepNLP-END2.0/blob/main/02_BackProp/LR_Comparison.png?raw=true)

## Explanation

To solve/train a neuron network, we are using gradient descent, which is

$$W_{new}=W_{old}-\eta\nabla{E_{total}}$$

where

$$W_{new} = \text{new weights} \\ W_{old}= \text{old weights} \\ \eta = \text{learning rate} \\ E_{total} = \text{error total}$$

Now we have weights $$W = \lbrack w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8 \rbrack$$

and

$$\nabla E_{total} = \nabla E_t = \lbrack \frac{\partial E_t}{\partial w_1}, \frac{\partial E_t}{\partial w_2}, \frac{\partial E_t}{\partial w_3}, \frac{\partial E_t}{\partial w_4}, \frac{\partial E_t}{\partial w_5}, \frac{\partial E_t}{\partial w_6}, \frac{\partial E_t}{\partial w_7}, \frac{\partial E_t}{\partial w_8} \rbrack$$

Now see carefully, the below equations, along with our network diagram above somewhere, very important!

$$h_1=w_1 i_i+w2 i_2 \\ h_2=w_3i_1+w4i_2 $$

$$
a_{h1} = \sigma({h_1}) \\  a_{h2} = \sigma({h_2})
$$

$$
o_1 = w_5a_{h_1}+w_6a_{h_2} \\ o_2 = w_7a_{h_1}+w_8a_{h_2}
$$

$$
a_{o_1} = \sigma(o_1) \\ a_{o_2} = \sigma(o_2)
$$

$$
E_1=\frac{1}{2}(t_1-a_{o_1})^2 \\ .\\ E_2=\frac{1}{2}(t_2-a_{o_1})^2
$$

Where $$[i_1, i_2] \text{ and }[t_1, t_2]$$ are the inputs and target outputs respectively

Now we can start to compute the partial derivatives w.r.t to the weights, and also remember to use chain rule wherever applicable and necessary.

$$\frac{\partial E_t}{\partial w_5} = \frac{\partial E_1}{\partial w_5} = \frac{\partial E_1}{\partial a_{o_1}} \frac{\partial a_{o_1}}{\partial o_1} \frac{\partial o_1}{\partial w_5}$$

We didn't consider E2 above, because it does no contribution to w5

$$\frac{\partial E_1}{\partial a_{o_1}}  = \frac{\partial \frac{1}{2}(t_1-a_{o_1})^2}{\partial a_{o_1}}  = a_{o_1}-t_1$$

$$\frac{\partial a_{o_1}}{\partial o_1}=\frac{\partial \sigma{(o_1)} }{\partial o_1} = \sigma(o_1)(1-\sigma(o_1)) = (a_{o_1})(1-a_{o_1})$$

and
$$\frac{\partial o_1}{\partial w_5} = a-o_1$$

similarly we can complete it for other weights in the last hidden layer

$$\frac{\partial E_t}{\partial w_5} = (a_{o_1}-t_1)(a_{o_1})(1-a_{o_1})a_{h_{1}}$$
$$\frac{\partial E_t}{\partial w_6} = (a_{o_1}-t_1)(a_{o_1})(1-a_{o_1})a_{h_{2}}$$
$$\frac{\partial E_t}{\partial w_7} = (a_{o_2}-t_2)(a_{o_2})(1-a_{o_2})a_{h_{1}}$$
$$\frac{\partial E_t}{\partial w_8} = (a_{o_2}-t_2)(a_{o_2})(1-a_{o_2})a_{h_{2}}$$

Now coming to the first hidden layer weights

$$\frac{\partial E_t}{\partial w_1}  = \frac{\partial E_t}{\partial a_{o_1}} \frac{\partial a_{o_1}}{\partial o_1 } \frac{\partial o_1}{\partial a_{h_1}} \frac{\partial a_{h_1}}{\partial h_1} \frac{\partial{h_1}}{\partial{w_1}}$$

Seen this somewhere, right, we did this before

$$\frac{\partial E_t}{\partial a_{o_1}} \frac{\partial a_{o_1}}{\partial o_1 } \frac{\partial o_1}{\partial a_{h_1}} = \frac{\partial (E_1+E_2)}{\partial a_{h_1}}= \frac{\partial E_1}{\partial a_{h_1}}  \frac{\partial E_2}{\partial a_{h_1}} \\ . \\ = (a_{o_1}-t_1)(a_{o_1})(1-a_{o_1})w_5 \\ + (a_{o_2}-t_2)(a_{o_2})(1-a_{o_2})w_7$$

Now we can compute all the gradient of the first layer

$$\frac{\partial E_t}{\partial w_1} = \frac{\partial E_t}{\partial a_{h_1}} (a-h_1)(1-a_{h_1})i_1$$
$$\frac{\partial E_t}{\partial w_2} = \frac{\partial E_t}{\partial a_{h_1}} (a-h_1)(1-a_{h_1})i_2$$
$$\frac{\partial E_t}{\partial w_3} = \frac{\partial E_t}{\partial a_{h_2}} (a-h_2)(1-a_{h_2})i_1$$
$$\frac{\partial E_t}{\partial w_4} = \frac{\partial E_t}{\partial a_{h_2}} (a-h_2)(1-a_{h_2})i_2$$

Done :) Now we have all the gradients, just need to update :))))))  maza aaya







