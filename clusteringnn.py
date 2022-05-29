import sys; args = sys.argv[1:]
import math, random, re

# t_funct is symbol of transfer functions: 'T1', 'T2', 'T3', or 'T4'
# input is a list of input (summation) values of the current layer
# returns a list of output values of the current layer
def transfer(t_funct, input):
   if t_funct == 'T3': return [1 / (1 + math.e**-x) for x in input]
   elif t_funct == 'T4': return [-1+2/(1+math.e**-x) for x in input]
   elif t_funct == 'T2': return [x if x > 0 else 0 for x in input]
   else: return [x for x in input]

def dot_product(input, weights):
    new_input = []
    len_sol = len(weights) // len(input)

    for i in range(len_sol):
        sum = 0
        for j in range(len(input)):
            sum += float(weights[i*len(input)+j])*float(input[j])
        new_input.append(sum)

    return new_input

def final_transform(input, weights):
    return [input[i]*weights[i] for i in range(len(input))]

# forward feeding for one input(training) set
# return updated x_vals and error of the one forward feeding
def ff(ts, xv, weights, t_funct):
   
   for i in range(len(weights)-1):
       xv[i+1] = dot_product(xv[i], weights[i])
       xv[i+1] = transfer(t_funct, xv[i+1])

   xv[-1] = final_transform(xv[-2], weights[-1])
   output_len = len(xv[-1])
   err = sum([(ts[-output_len+i] - xv[-1][i])**2 for i in range(len(xv[-1]))]) / 2
   return xv, err

# back propagation with one training set and corresponding x_vals and weights
# update E_vals (ev) and negative_grad, and then return those two lists
def bp(ts, xv, weights, ev, negative_grad):   

    for i in reversed(range(len(xv))):
        layer = xv[i]
        if i != len(xv)-1:
            for j in range(len(layer)):
                error = 0.0
                if i == len(xv)-2:
                    error += (ev[i+1][j]*layer[j]*(1-layer[j])*weights[i][j])
                    negative_grad[i][j] = ev[i+1][j]*xv[i][j]
                else:
                    for k in range(len(ev[i + 1])):
                        error += (ev[i+1][k]*layer[j]*(1-layer[j])*weights[i][k*len(ev[i])+j])
                        negative_grad[i][k*len(ev[i])+j] = ev[i+1][k]*xv[i][j]
                if i != 0:
                    ev[i][j] = error
        else:
            for j in range(len(layer)):
                output_len = len(xv[-1])
                ev[i][j] = (ts[-output_len+j] - layer[j])

    return ev, negative_grad

# update all weights and return the new weights
def update_weights(weights, negative_grad, alpha):

   for i in range(len(weights)):
       for j in range(len(weights[i])):
           weights[i][j] += negative_grad[i][j]*alpha

   return weights

def find_output(x, y, equality, radius):
    sol = x*x+y*y
    bool = False
    if equality == "<": bool = sol < radius 
    elif equality == "<=": bool = sol <= radius 
    elif equality == ">": bool = sol > radius 
    elif equality == ">=": bool = sol >= radius 
    else: return None
    return 1 if bool else 0

def rand_train(radius):
    range = math.sqrt(math.pi/2) * radius 
    return random.uniform(-range, range), random.uniform(-range, range)

def spec_train(r, n, eq):
    to_ret = []
    for x in range(0, n+1):
        x = math.cos(2*math.pi/n*x)*r
        y = math.sin(2*math.pi/n*x)*r
        sol = find_output(x, y, eq, r)
        to_ret.append([x, y, sol])
    
    return to_ret

def spec_helper(r, eq):
    spec_list = []
    while len(spec_list) < 1000:
        diff = random.uniform(-0.05, 0.05)
        spec_list += spec_train(r+diff, random.randint(1, 10), eq)

    return spec_list

def main():
   t_funct = 'T3' # we default the transfer(activation) function as 1 / (1 + math.e**(-x))

   equation = args[0]
   radius = float(re.search('[\d.]+', equation).group())
   equality = re.search('(?<=x\*x\+y\*y)(.+)', equation.replace(str(radius), "")).group()

   output_len = 1 
   
   layer_counts = [3, 3, 2, output_len, output_len] # set the number of layers

   # by using the layer counts, set initial weights [3, 2, 1, 1] => 3*2 + 2*1 + 1*1: Total 6, 2, and 1 weights are needed
   weights = [[round(random.uniform(-1.0, 1.0), 2) for j in range(layer_counts[i]*layer_counts[i+1])] for i in range(len(layer_counts)-2)]
   weights.append([round(random.uniform(-1.0, 1.0), 2) for j in range(output_len)])

   # build the structure of BP NN: E nodes and negative_gradients 
   alpha = 0.1

   for i in range(4):
    count = 0
    
    training_set = []  

    training_set.append([0, 0, find_output(0, 0, equality, radius)])
    training_set.append([radius, radius, find_output(radius, radius, equality, radius)])
    training_set.append([-radius, -radius, find_output(-radius, -radius, equality, radius)])
    training_set.append([-radius, radius, find_output(-radius, radius, equality, radius)])
    training_set.append([radius, -radius, find_output(radius, -radius, equality, radius)])

    training_set += spec_helper(radius, equality)

    x_vals = [[temp[0:len(temp)-output_len]] for temp in training_set] # x_vals starts with first input values
    # make the x value structure of the NN by putting bias and initial value 0s.
    for i in range(len(training_set)):
        for j in range(len(layer_counts)):
            if j == 0: x_vals[i][j].append(1.0)
            else: x_vals[i].append([0 for temp in range(layer_counts[j])])

    E_vals = [[*i] for i in x_vals]  #copy elements from x_vals, E_vals has the same structures with x_vals
    negative_grad = [[*i] for i in weights]  #copy elements from weights, negative gradients has the same structures with weights
    errors = [10]*len(training_set)  # Whenever FF is done once, error will be updated. Start with 10 (a big num)

    # calculate the initail error sum. After each forward feeding (# of training sets), calculate the error and store at error list
    for k in range(len(training_set)):
        x_vals[k], errors[k] = ff(training_set[k], x_vals[k], weights, t_funct)
        E_vals[k], negative_grad = bp(training_set[k], x_vals[k], weights, E_vals[k], negative_grad)
        weights = update_weights(weights, negative_grad, alpha)
    err = sum(errors)
    
    while err > 2 and count < 100:
        if err > 150:
            weights = [[round(random.uniform(-1.0, 1.0), 2) for j in range(layer_counts[i]*layer_counts[i+1])] for i in range(len(layer_counts)-2)]
            weights.append([round(random.uniform(-1.0, 1.0), 2) for j in range(output_len)])
            count = 0
        for k in range(len(training_set)):
                x_vals[k], errors[k] = ff(training_set[k], x_vals[k], weights, t_funct)
                E_vals[k], negative_grad = bp(training_set[k], x_vals[k], weights, E_vals[k], negative_grad)
                weights = update_weights(weights, negative_grad, alpha)
        err = sum(errors)
        count+=1

    training_set = []
   
    for i in range(995):
        x, y = rand_train(radius)
        training_set.append([x, y, find_output(x, y, equality, radius)])

    x_vals = [[temp[0:len(temp)-output_len]] for temp in training_set] # x_vals starts with first input values
    # make the x value structure of the NN by putting bias and initial value 0s.
    for i in range(len(training_set)):
        for j in range(len(layer_counts)):
            if j == 0: x_vals[i][j].append(1.0)
            else: x_vals[i].append([0 for temp in range(layer_counts[j])])

    errors = [10]*len(training_set)  # Whenever FF is done once, error will be updated. Start with 10 (a big num)
    E_vals = [[*i] for i in x_vals]  #copy elements from x_vals, E_vals has the same structures with x_vals

    for k in range(len(training_set)):
        x_vals[k], errors[k] = ff(training_set[k], x_vals[k], weights, t_funct)
        E_vals[k], negative_grad = bp(training_set[k], x_vals[k], weights, E_vals[k], negative_grad)
        weights = update_weights(weights, negative_grad, alpha)
    err = sum(errors)

    count = 0

    while err > 2 and count < 200:
        if err > 150:
            weights = [[round(random.uniform(-1.0, 1.0), 2) for j in range(layer_counts[i]*layer_counts[i+1])] for i in range(len(layer_counts)-2)]
            weights.append([round(random.uniform(-1.0, 1.0), 2) for j in range(output_len)])
            count = 0
        for k in range(len(training_set)):
                x_vals[k], errors[k] = ff(training_set[k], x_vals[k], weights, t_funct)
                E_vals[k], negative_grad = bp(training_set[k], x_vals[k], weights, E_vals[k], negative_grad)
                weights = update_weights(weights, negative_grad, alpha)
        err = sum(errors)
        count+=1
   print ('Layer counts:', layer_counts)
   print ('Weights:')
   for w in weights: print (w)
if __name__ == '__main__': main()
